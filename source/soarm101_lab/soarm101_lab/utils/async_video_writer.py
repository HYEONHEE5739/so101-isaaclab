"""Bounded RGB streaming and exact CFR frame reads shared by existing dataset paths."""
from __future__ import annotations
import json
import queue
import time
import shutil
import threading
import uuid
import weakref
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


class AsyncVideoWriter:
    """Copy into a bounded queue; backpressure never drops observation frames."""
    def __init__(self, video_path, codec='libx264', fps=30, width=640, height=480,
                 crf=18, queue_size=8):
        self.video_path=Path(video_path);self.codec=codec;self.fps=Fraction(str(fps)).limit_denominator(1000000)
        self.width=int(width);self.height=int(height);self.crf=int(crf)
        if self.fps<=0 or not 0<=self.crf<=51 or queue_size<1:
            raise ValueError('Invalid video FPS/CRF/queue size')
        self._queue=queue.Queue(maxsize=queue_size);self._error=None
        self.encoding_seconds=0.;self.queue_peak=0
        self._frame_index=0;self.frames=0;self._closed=False;self._cancel=threading.Event()
        self._thread=threading.Thread(target=self._worker,name='video-'+self.video_path.stem,daemon=True)
        self._thread.start()

    def write(self,image):
        self._raise_if_failed()
        if self._closed:raise RuntimeError('Video writer is closed: '+str(self.video_path))
        if image.dtype!=np.uint8 or image.shape!=(self.height,self.width,3):
            raise ValueError('RGB video requires uint8 [height,width,3]')
        # Observations/render buffers may be reused as soon as this call returns.
        image=np.array(image,copy=True,order='C')
        while True:
            self._raise_if_failed()
            try:self._queue.put(image,timeout=.05);self.frames+=1;return
            except queue.Full:continue

    def close(self):
        if not self._closed:
            self._closed=True
            while self._thread.is_alive():
                self._raise_if_failed()
                try:self._queue.put(None,timeout=.05);break
                except queue.Full:continue
            self._thread.join()
        self._raise_if_failed()
        if self._frame_index!=self.frames:raise RuntimeError('Video frame count mismatch')

    def abort(self):
        self._closed=True;self._cancel.set();self._thread.join()
        self.video_path.unlink(missing_ok=True)

    def _worker(self):
        try:
            self.video_path.parent.mkdir(parents=True,exist_ok=True)
            with av.open(str(self.video_path),'w',format='mp4') as container:
                stream=container.add_stream(self.codec,rate=self.fps,
                    options={'crf':str(self.crf),'preset':'veryfast','threads':'1'})
                stream.width=self.width;stream.height=self.height;stream.pix_fmt='yuv420p'
                stream.time_base=1/self.fps
                while not self._cancel.is_set():
                    try:image=self._queue.get(timeout=.05)
                    except queue.Empty:continue
                    if image is None:break
                    begin=time.perf_counter();self.queue_peak=max(self.queue_peak,self._queue.qsize()+1)
                    frame=av.VideoFrame.from_ndarray(image,format='rgb24')
                    frame.pts=self._frame_index;frame.time_base=1/self.fps
                    for packet in stream.encode(frame):container.mux(packet)
                    self.encoding_seconds+=time.perf_counter()-begin
                    self._frame_index+=1
                if not self._cancel.is_set():
                    begin=time.perf_counter()
                    for packet in stream.encode():container.mux(packet)
                    self.encoding_seconds+=time.perf_counter()-begin
        except Exception as exc:self._error=exc

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f'Video encoding failed: {self.video_path}: {self._error}') from self._error


class VideoFrames:
    """Array-like metadata and exact scalar seek. One decoder, no full-episode RGB array."""
    _open=OrderedDict()
    _lock=threading.RLock()

    def __init__(self,path,frames=None,fps=None):
        self.path=Path(path);self._container=None;self._decoder=None;self._next=None
        with av.open(str(self.path)) as container:
            s=container.streams.video[0]
            self.fps=float(s.average_rate);count=s.frames
            if not count:raise ValueError('Video has no finalized frame index: '+str(path))
            self.shape=(count,s.height,s.width,3);self.dtype=np.dtype('uint8');self.ndim=4
            if frames is not None and count!=frames:raise ValueError('Video/numeric frame count mismatch: '+str(path))
            if fps is not None and abs(self.fps-fps)>1e-6:raise ValueError('Video/control FPS mismatch: '+str(path))

    def __len__(self):return self.shape[0]

    def __getitem__(self,index):
        with self._lock:return self._read(index)

    def _read(self,index):
        if not isinstance(index,(int,np.integer)):raise TypeError('VideoFrames only decodes one explicit frame index')
        index=int(index)
        if not 0<=index<len(self):raise IndexError(index)
        self._open[id(self)]=weakref.ref(self);self._open.move_to_end(id(self))
        while len(self._open)>2:
            _,ref=self._open.popitem(last=False)
            old=ref()
            if old is not None:old.close()
        if self._container is None:
            self._container=av.open(str(self.path));self._stream=self._container.streams.video[0]
            self._stream.thread_type='SLICE';self._stream.codec_context.thread_count=1
        if self._next!=index:
            pts=round(index/self.fps/float(self._stream.time_base))
            self._container.seek(pts,stream=self._stream,backward=True,any_frame=False)
            self._decoder=self._container.decode(self._stream)
        for frame in self._decoder:
            seconds=float(frame.pts*frame.time_base)
            actual=round(seconds*self.fps)
            if abs(seconds-actual/self.fps)>1e-5:raise ValueError('Non-CFR video timestamp: '+str(self.path))
            if actual<index:continue
            if actual!=index:raise ValueError(f'Exact frame {index} missing in {self.path}, got {actual}')
            self._next=index+1
            return frame.to_ndarray(format='rgb24')
        raise ValueError(f'Video truncated: {self.path}, frame {index}')

    def close(self):
        with self._lock:self._close()

    def _close(self):
        self._open.pop(id(self),None)
        if self._container is not None:self._container.close();self._container=None
        self._next=None
    def __del__(self):self.close()


class VideoEpisode:
    """Private staging folder; publish only fully finalized and count-checked videos."""
    def __init__(self,parent,fps,crf=18):
        self.folder=Path(parent)/('.video-'+uuid.uuid4().hex)
        self.folder.mkdir(parents=True,exist_ok=False)
        self.fps=fps;self.crf=crf;self.writers={}

    def append(self,images):
        if self.writers and set(images)!=set(self.writers):raise ValueError('Episode camera set changed')
        for name,image in images.items():
            if not name or Path(name).name!=name:raise ValueError('Invalid camera ID')
            if name not in self.writers:
                self.writers[name]=AsyncVideoWriter(self.folder/(name+'.mp4'),fps=self.fps,
                    height=image.shape[0],width=image.shape[1],crf=self.crf)
            self.writers[name].write(image)

    def finish(self,frames):
        specs={}
        for name,writer in self.writers.items():
            writer.close()
            video=VideoFrames(writer.video_path,frames,self.fps)
            specs[name]={'file':name+'.mp4','frames':frames,'fps':self.fps,
                'shape':list(video.shape[1:]),'codec':'h264','crf':self.crf,
                'timestamp_semantics':'frame index maps to recorded observation timestamp'}
        return specs

    def abort(self):
        for writer in self.writers.values():writer.abort()
        if self.folder.exists():shutil.rmtree(self.folder)


def hdf5_cameras(group):
    """External specs coexist with legacy internal cameras; never duplicate a camera silently."""
    specs=json.loads(group.attrs.get('external_videos','{}'))
    for prefix in ('obs/policy','obs','obs/camera'):
        if prefix not in group:continue
        for name,value in group[prefix].items():
            if getattr(value,'ndim',0)==4 and (name.endswith('_cam') or prefix=='obs/camera'):
                key=name if name.endswith('_cam') else name+'_cam'
                if key in specs:raise ValueError('Duplicate internal/external camera: '+key)
                specs[key]={'path':prefix+'/'+name}
    return specs


def external_video_path(dataset,spec):
    root=Path(dataset).resolve().parent;path=(root/spec['file']).resolve()
    if not path.is_relative_to(root) and not (Path(dataset).name=='conversion_input.hdf5' and spec.get('source_reference')):raise ValueError('External video outside dataset parent')
    if not path.is_file():raise FileNotFoundError(path)
    return path


def copy_hdf5_videos(source_group,dest_group,destination):
    """Subset/Resume copies only referenced videos; new names never overwrite prior episodes."""
    specs=json.loads(source_group.attrs.get('external_videos','{}'))
    if not specs:return
    if Path(source_group.file.filename).resolve()==Path(destination).resolve():return
    parent=Path(destination).resolve().parent
    folder=parent/(Path(destination).stem+'.videos')/uuid.uuid4().hex
    folder.mkdir(parents=True,exist_ok=False)
    try:
        copied={}
        for name,spec in specs.items():
            old=external_video_path(source_group.file.filename,spec)
            target=folder/old.name
            shutil.copyfile(old,target)
            copied[name]={**spec,'file':str(target.relative_to(parent))}
        dest_group.attrs['external_videos']=json.dumps(copied)
    except BaseException:
        shutil.rmtree(folder);raise


def external_camera(dataset,spec,frames=None,fps=None):
    path=external_video_path(dataset,spec)
    if spec.get('format')=='npy':
        value=np.load(path,mmap_mode='r',allow_pickle=False)
        if value.dtype!=np.uint8 or value.ndim!=4 or value.shape[-1]!=3 or (frames is not None and len(value)!=frames):
            raise ValueError('External NPY camera shape/count mismatch')
        return value
    return VideoFrames(path,frames if frames is not None else spec.get('frames'),fps if fps is not None else spec.get('fps'))
