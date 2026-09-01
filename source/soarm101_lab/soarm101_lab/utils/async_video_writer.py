from __future__ import annotations

import queue
import threading
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


class AsyncVideoWriter:
    """
    numpy RGB frame을 백그라운드에서 MP4로 인코딩하는 writer.

    주의:
        queue가 가득 차면 frame을 버리지 않고 기다린다.
        따라서 영상 frame 수와 state/action frame 수가 항상 일치한다.
    """

    def __init__(
        self,
        video_path: Path,
        codec: str,
        fps: int,
        width: int,
        height: int,
        crf: int = 30,
        queue_size: int = 30,
    ):
        self.video_path = Path(video_path)
        self.codec = codec
        self.fps = fps
        self.width = width
        self.height = height
        self.crf = crf

        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=queue_size
        )

        self._error: Exception | None = None
        self._frame_index = 0
        self._closed = False

        self._thread = threading.Thread(
            target=self._worker,
            name=f"video-writer-{self.video_path.stem}",
            daemon=True,
        )
        self._thread.start()

    def write(self, image: np.ndarray) -> None:
        """
        frame을 encoder queue에 넣는다.

        queue가 가득 차면 encoder가 따라올 때까지 기다린다.
        """
        self._raise_if_failed()

        if self._closed:
            raise RuntimeError(
                f"Cannot write to closed video writer: {self.video_path}"
            )

        image = np.ascontiguousarray(image)

        while True:
            self._raise_if_failed()

            try:
                self._queue.put(image, timeout=0.1)
                return
            except queue.Full:
                # worker가 처리할 때까지 기다리되,
                # worker 예외가 발생했는지는 계속 확인한다.
                continue

    def close(self) -> None:
        """
        queue에 남은 모든 frame을 처리한 후 worker를 종료한다.
        """
        if self._closed:
            self._raise_if_failed()
            return

        self._closed = True
        self._queue.put(None)
        self._thread.join()

        self._raise_if_failed()

    def _worker(self) -> None:
        try:
            self.video_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            codec_options = {
                "crf": str(self.crf),
            }

            with av.open(str(self.video_path), mode="w") as container:
                stream = container.add_stream(
                    self.codec,
                    rate=self.fps,
                    options=codec_options,
                )

                stream.width = self.width
                stream.height = self.height
                stream.pix_fmt = "yuv420p"
                stream.time_base = Fraction(1, self.fps)

                while True:
                    image = self._queue.get()

                    try:
                        if image is None:
                            break

                        video_frame = av.VideoFrame.from_ndarray(
                            image,
                            format="rgb24",
                        )

                        video_frame.pts = self._frame_index
                        video_frame.time_base = Fraction(1, self.fps)

                        for packet in stream.encode(video_frame):
                            container.mux(packet)

                        self._frame_index += 1

                    finally:
                        self._queue.task_done()

                # encoder 내부 buffer flush
                for packet in stream.encode():
                    container.mux(packet)

        except Exception as exc:
            self._error = exc

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"Video encoding failed: {self.video_path}"
            ) from self._error