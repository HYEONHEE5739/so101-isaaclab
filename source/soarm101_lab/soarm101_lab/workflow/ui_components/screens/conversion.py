"""LeRobot conversion; no camera or interaction control panel."""
from .base import StageScreen


class ConversionScreen(StageScreen):
    action = '데이터셋 변환'
    basic_keys = {'repo_id'}
