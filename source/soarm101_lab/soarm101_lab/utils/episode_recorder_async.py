from __future__ import annotations

import json
import logging
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image
from .async_video_writer import AsyncVideoWriter

logger = logging.getLogger(__name__)


class EpisodeRecorderAsync:
    """
    IsaacLab에서 LeRobot 변환용 중간 데이터를 기록하는 클래스.

    기록 중:

    dataset_root/
    ├── manifest.json
    ├── episode_000000/              # 저장 완료 episode
    └── .staging/
        └── episode_000001/          # 현재 기록 중인 episode
            ├── images/
            │   ├── side/
            │   │   ├── frame-000000.png
            │   │   └── ...
            │   └── wrist/
            │       └── ...
            ├── state.npy            # save_episode() 때 생성
            ├── action.npy           # save_episode() 때 생성
            └── timestamps.npy       # save_episode() 때 생성

    save_episode() 이후:

    episode_000001/
    ├── cameras/
    │   ├── side.mp4
    │   └── wrist.mp4
    ├── state.npy
    ├── action.npy
    ├── timestamps.npy
    └── episode.json

    LeRobotDataset은 이 recorder에서 사용하지 않습니다.

    Methods:
        ├── start_episode()
            현재 episode의 임시 저장 디렉터리를 생성합니다.
        
        ├── add_frame(observation, action, timestamp=None) <-- 실제 사용
        
        ├── next_episode(task=None) <-- 실제 사용
            현재 episode를 저장하고 다음 episode를 시작합니다.
        
        ├── restart_episode() <-- 실제 사용
            현재 episode를 폐기하고 같은 episode 번호로 다시 시작합니다.
        
        ├── save_episode(task=None)
            PNG를 MP4로 변환하고 state/action/timestamp와 metadata를 저장합니다.
        
        ├── close(discard_current=True) <-- 실제 사용
            Recorder 종료 시 현재 저장 중인 staging episode를 정리합니다.
        
    """

    def __init__(
        self,
        repo_id: str,
        fps: int = 30,
        root: str | Path | None = None,
        camera_names: tuple[str, ...] = ("side_came", "wrist_came"),
        task: str | None = None,
        resume: bool = False,
        video_codec: str = "libsvtav1",
        video_crf: int = 30,
    ):
        self.repo_id = repo_id
        self.fps = fps
        self.task = task
        self.camera_names = tuple(camera_names)
        self.video_codec = video_codec
        self.video_crf = video_crf

        if root is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[4]
            root = project_root / "dataset"

        self.root = Path(root)
        self.save_dir = self.root / repo_id
        self.manifest_path = self.save_dir / "manifest.json"
        self.staging_dir = self.save_dir / ".staging"

        # resume=False일 때에는 dataset 디렉터리가 이미 있으면 중단한다.
        # 단, 아직 아무 파일도 없는 새 디렉터리는 생성 가능하게 한다.
        if resume:
            if not self.save_dir.exists():
                raise FileNotFoundError(
                    f"Cannot resume. Dataset directory does not exist: {self.save_dir}"
                )

            self.episode_id = self._get_next_episode_id()
        else:
            if self.save_dir.exists() and any(self.save_dir.iterdir()):
                raise FileExistsError(
                    f"Dataset directory is not empty: {self.save_dir}. "
                    "Use resume=True to continue recording."
                )

            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.episode_id = 0

        self.staging_dir.mkdir(parents=True, exist_ok=True)

        self.start_episode()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        """
        현재 episode의 임시 저장 디렉터리를 생성한다.
        """

        self.episode_dir = (
            self.staging_dir / f"episode_{self.episode_id:06d}"
        )
        self.images_dir = self.episode_dir / "images"
        self.cameras_dir = self.episode_dir / "cameras"

        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.cameras_dir.mkdir(parents=True, exist_ok=True)

        self.camera_dirs: dict[str, Path] = {}

        for camera_name in self.camera_names:
            camera_dir = self.images_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            self.camera_dirs[camera_name] = camera_dir

        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.timestamps: list[float] = []

        self.frame_idx = 0

        # 새 episode 마다 to writer를 생성
        self.video_writers: dict[str, AsyncVideoWriter] = {}

    def next_episode(self, task: str | None = None) -> None:
        """
        현재 episode를 저장하고 다음 episode를 시작한다.
        """

        self.save_episode(task=task)

        self.episode_id += 1
        self.start_episode()

    def restart_episode(self) -> None:
        """
        현재 episode를 폐기하고 같은 episode 번호로 다시 시작한다.

        현재 episode는 .staging 아래에 있으므로,
        정상 저장된 episode에는 영향을 주지 않는다.
        """
        self._close_video_writers()

        if self.episode_dir.exists():
            logger.info("Discarding episode %06d: %s", self.episode_id, self.episode_dir)
            shutil.rmtree(self.episode_dir)

        self.start_episode()

    # ------------------------------------------------------------------
    # Frame writing
    # ------------------------------------------------------------------

    def add_frame(
        self,
        observation: dict[str, Any],
        action: Any,
        timestamp: float | None = None,
    ) -> None:
        """
        한 frame을 기록한다.

        현재는 num_envs 중 첫 번째 환경 env[0]만 기록한다.
        """

        state = self._extract_first_env(observation["policy"]["joint_pos"])
        action = self._extract_first_env(action)

        state = self._to_numpy(state, dtype=np.float32)
        action = self._to_numpy(action, dtype=np.float32)

        self.states.append(state)
        self.actions.append(action)


        if timestamp is None:
            timestamp = self.frame_idx / self.fps

        self.timestamps.append(float(timestamp))

        policy_observations = observation["policy"]

        for camera_name in self.camera_names:
            if camera_name not in policy_observations:
                raise KeyError(
                    f"Camera '{camera_name}' was not found in "
                    f"observation['policy']. "
                    f"Available cameras: {list(policy_observations.keys())}"
                )

            image = self._extract_first_env(policy_observations[camera_name])
            image = self._to_uint8_hwc(image)

            if camera_name not in self.video_writers:
                height, width, channels = image.shape

                if channels != 3:
                    raise ValueError(
                        f"Expected RGB image for '{camera_name}', "
                        f"but got shape {image.shape}"
                    )

                video_path = (
                    self.cameras_dir / f"{camera_name}.mp4"
                )

                self.video_writers[camera_name] = AsyncVideoWriter(
                    video_path=video_path,
                    codec=self.video_codec,
                    fps=self.fps,
                    width=width,
                    height=height,
                    crf=self.video_crf,
                    queue_size=30,
                )

            self.video_writers[camera_name].write(image)

        self.frame_idx += 1

    # ------------------------------------------------------------------
    # Episode saving
    # ------------------------------------------------------------------

    def save_episode(self, task: str | None = None) -> None:
        """
        현재 episode의 PNG들을 MP4로 변환하고,
        state/action/timestamp와 metadata를 저장한다.

        모든 저장 작업이 성공하면 .staging/episode_xxxxxx를
        최종 episode_xxxxxx로 이동한다.
        """

        if self.frame_idx == 0:
            logger.warning(
                "Episode %06d has no frames. Nothing will be saved.",
                self.episode_id,
            )
            return

        episode_task = task if task is not None else self.task

        states = np.stack(self.states).astype(np.float32)
        actions = np.stack(self.actions).astype(np.float32)
        timestamps = np.asarray(self.timestamps, dtype=np.float64)

        if len(states) != self.frame_idx:
            raise RuntimeError(
                f"State count mismatch: {len(states)} != {self.frame_idx}"
            )

        if len(actions) != self.frame_idx:
            raise RuntimeError(
                f"Action count mismatch: {len(actions)} != {self.frame_idx}"
            )

        if len(timestamps) != self.frame_idx:
            raise RuntimeError(
                f"Timestamp count mismatch: "
                f"{len(timestamps)} != {self.frame_idx}"
            )

        logger.info(
            "Encoding episode %06d videos from PNG files...",
            self.episode_id,
        )

        # queue에 남아 있는 frame을 모두 MP4에 기록
        self._close_video_writers()

        # 2. 수치 데이터를 저장한다.
        np.save(self.episode_dir / "state.npy", states)
        np.save(self.episode_dir / "action.npy", actions)
        np.save(self.episode_dir / "timestamps.npy", timestamps)

        # 3. PNG를 삭제한다.
        # MP4와 npy 저장이 모두 끝난 뒤에 삭제한다.
        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)

        episode_metadata = {
            "episode_index": self.episode_id,
            "repo_id": self.repo_id,
            "fps": self.fps,
            "num_frames": self.frame_idx,
            "state_shape": list(states.shape),
            "action_shape": list(actions.shape),
            "camera_names": list(self.camera_names),
            "task": episode_task,
            "video_codec": self.video_codec,
            "video_crf": self.video_crf,
            "video_files": {
                camera_name: f"cameras/{camera_name}.mp4"
                for camera_name in self.camera_names
            },
        }

        with open(
            self.episode_dir / "episode.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(episode_metadata, f, indent=2)

        # 4. staging episode를 최종 episode로 이동한다.
        final_episode_dir = (
            self.save_dir / f"episode_{self.episode_id:06d}"
        )

        if final_episode_dir.exists():
            raise FileExistsError(
                f"Final episode directory already exists: "
                f"{final_episode_dir}"
            )

        final_episode_dir.parent.mkdir(parents=True, exist_ok=True)

        # 같은 파일시스템 안에서 이름만 변경하므로 빠르게 이동된다.
        self.episode_dir.replace(final_episode_dir)

        # 5. 저장 완료된 episode만 manifest에 등록한다.
        episode_metadata["episode_dir"] = final_episode_dir.name
        self._update_manifest(episode_metadata)

        logger.info(
            "Saved episode %06d: %d frames",
            self.episode_id,
            self.frame_idx,
        )

    # ------------------------------------------------------------------
    # PNG -> MP4
    # ------------------------------------------------------------------

    def _close_video_writers(self) -> None:
        """
        현재 episode의 모든 video writer를 종료한다.
        queue에 남은 frame도 모두 MP4에 기록한 뒤 반환한다.
        """
        writers = self.video_writers

        for camera_name, writer in writers.items():
            print(
                f"[RECORDER] Finalizing {camera_name}.mp4",
                flush=True,
            )
            writer.close()

        self.video_writers = {}
        
    # def _encode_video_frames(
    #     self,
    #     imgs_dir: Path,
    #     video_path: Path,
    # ) -> None:
    #     """
    #     frame-000000.png, frame-000001.png, ...
    #     파일들을 하나의 MP4로 인코딩한다.
    #     """

    #     image_paths = sorted(
    #         imgs_dir.glob("frame-*.png"),
    #         key=lambda path: int(path.stem.split("-")[-1]),
    #     )

    #     if not image_paths:
    #         raise FileNotFoundError(
    #             f"No PNG frames found in {imgs_dir}"
    #         )

    #     with Image.open(image_paths[0]) as first_image:
    #         width, height = first_image.size

    #     video_path.parent.mkdir(parents=True, exist_ok=True)

    #     codec_options = {
    #         "crf": str(self.video_crf),
    #     }

    #     with av.open(str(video_path), mode="w") as container:
    #         stream = container.add_stream(
    #             self.video_codec,
    #             rate=self.fps,
    #             options=codec_options,
    #         )

    #         stream.width = width
    #         stream.height = height
    #         stream.pix_fmt = "yuv420p"
    #         stream.time_base = Fraction(1, self.fps)

    #         for frame_index, image_path in enumerate(image_paths):
    #             with Image.open(image_path) as image:
    #                 image = image.convert("RGB")
    #                 video_frame = av.VideoFrame.from_image(image)

    #             video_frame.pts = frame_index
    #             video_frame.time_base = Fraction(1, self.fps)

    #             for packet in stream.encode(video_frame):
    #                 container.mux(packet)

    #         # Encoder buffer flush
    #         for packet in stream.encode():
    #             container.mux(packet)

    #     if not video_path.exists():
    #         raise OSError(
    #             f"Video encoding failed: {video_path}"
    #         )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _update_manifest(
        self,
        episode_metadata: dict[str, Any],
    ) -> None:
        if self.manifest_path.exists():
            with open(
                self.manifest_path,
                "r",
                encoding="utf-8",
            ) as f:
                manifest = json.load(f)
        else:
            manifest = {
                "dataset_version": "1.0",
                "repo_id": self.repo_id,
                "fps": self.fps,
                "episodes": [],
            }

        manifest["episodes"] = [
            episode
            for episode in manifest["episodes"]
            if episode["episode_index"]
            != episode_metadata["episode_index"]
        ]

        manifest["episodes"].append(episode_metadata)
        manifest["episodes"].sort(
            key=lambda episode: episode["episode_index"]
        )

        manifest["num_episodes"] = len(manifest["episodes"])
        manifest["total_frames"] = sum(
            episode["num_frames"]
            for episode in manifest["episodes"]
        )

        if manifest["episodes"]:
            manifest["next_episode_id"] = (
                max(
                    episode["episode_index"]
                    for episode in manifest["episodes"]
                )
                + 1
            )
        else:
            manifest["next_episode_id"] = 0

        with open(
            self.manifest_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(manifest, f, indent=2)

    def _get_next_episode_id(self) -> int:
        if not self.manifest_path.exists():
            return 0

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        return int(manifest.get("next_episode_id", 0))

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_first_env(value: Any) -> Any:
        """
        num_envs 차원이 있는 경우 첫 번째 환경만 가져온다.
        """

        if isinstance(value, torch.Tensor):
            return value[0] if value.ndim > 0 else value

        if isinstance(value, np.ndarray):
            return value[0] if value.ndim > 0 else value

        if isinstance(value, (list, tuple)):
            return value[0] if len(value) > 0 else value

        return value

    @staticmethod
    def _to_numpy(
        value: Any,
        dtype: np.dtype | None = None,
    ) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)

        if dtype is not None:
            value = value.astype(dtype)

        return value

    @classmethod
    def _to_uint8_hwc(cls, image: Any) -> np.ndarray:
        image = cls._to_numpy(image)

        if image.ndim != 3:
            raise ValueError(
                f"Expected a 3D image, but got shape {image.shape}"
            )

        # CHW -> HWC
        if (
            image.shape[0] in (1, 3, 4)
            and image.shape[-1] not in (1, 3, 4)
        ):
            image = np.transpose(image, (1, 2, 0))

        # grayscale -> RGB
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # RGBA -> RGB
        if image.shape[-1] == 4:
            image = image[..., :3]

        if image.dtype != np.uint8:
            image = image.astype(np.float32)

            if image.max() <= 1.0:
                image = image * 255.0

            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB image with 3 channels, got {image.shape}"
            )

        return np.ascontiguousarray(image)

    def close(self, discard_current: bool = True) -> None:
        """
        recorder 종료 시 현재 저장 중인 staging episode를 정리한다.

        discard_current=True:
            아직 save_episode()되지 않은 현재 episode 삭제

        discard_current=False:
            현재 staging episode를 그대로 남김
        """
        self._close_video_writers()

        if not discard_current:
            return

        if self.episode_dir.exists():
            logger.info(
                "Removing unfinished staging episode %06d: %s",
                self.episode_id,
                self.episode_dir,
            )
            shutil.rmtree(self.episode_dir)

        # staging 디렉터리가 비었으면 제거
        if self.staging_dir.exists() and not any(self.staging_dir.iterdir()):
            self.staging_dir.rmdir()

    def __del__(self):
        try:
            self.close(discard_current=True)
        except Exception:
            pass