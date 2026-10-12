"""Single-environment contract: named model-radian targets and workspace-frame poses."""
from abc import ABC, abstractmethod


class SimBackend(ABC):
    @abstractmethod
    def reset(self, seed=None, initial_state=None): ...

    @abstractmethod
    def step(self, joint_target_rad): ...

    @abstractmethod
    def get_state(self): ...

    @abstractmethod
    def get_rgb(self, camera_name='side'): ...

    @abstractmethod
    def get_eef_pose(self): ...

    @abstractmethod
    def get_object_pose(self, object_name): ...

    @abstractmethod
    def set_object_pose(self, object_name, pose): ...

    @abstractmethod
    def get_initial_state(self): ...

    @abstractmethod
    def restore_state(self, state): ...

    @abstractmethod
    def is_success(self): ...

    @abstractmethod
    def close(self): ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
