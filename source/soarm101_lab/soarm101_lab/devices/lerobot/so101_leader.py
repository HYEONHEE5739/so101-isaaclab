# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO101 leader arm teleoperation device."""
"""keyboard for Isaacsim episode recording option"""
import json
import os
import logging
import math

from dataclasses import dataclass
import numpy as np
import torch

#from soarm101_lab.assets import SO101_FOLLOWER_MOTOR_LIMITS

from ..device_base import DeviceBase
from soarm101_lab.devices.lerobot.utils import check_if_not_connected, check_if_already_connected
from soarm101_lab.devices.lerobot.motors import (
    Motor,
    MotorCalibration,
    MotorNormMode,
)
from .motors.feetech import FeetechMotorsBus, OperatingMode
from soarm101_lab.assets import SO101_FOLLOWER_USD_JOINT_LIMLITS

@dataclass
class SO101LeaderCfg:
    """Configuration for SO101 Leader device."""
    port: str = "/dev/so101_leader"
    """Serial port for the motor bus (e.g., /dev/ttyACM0)."""

    recalibrate: bool = False
    """If True, run calibration before connecting."""

    calibration_file_name: str = "so101_leader.json"
    """Name of calibration JSON file."""

# SO101의 모터 구성
SO101_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

SO101_MOTOR_NAMES = list(SO101_MOTORS.keys())

logger = logging.getLogger(__name__)

class SO101Leader(DeviceBase):
    """SO101 leader arm for teleoperation.

    Reads physical SO101 robot joint positions and returns them as torch.Tensor
    for controlling a virtual SO101 follower robot in IsaacSim.
    Example:
        >>> cfg = SO101LeaderCfg(port="/dev/ttyACM0")
        >>> leader = SO101Leader(cfg)
        >>> leader.connect()
        >>> for step in range(1000):
        ...     action = leader.advance()  # torch.Tensor shape (6,)
        ...     env.set_joint_targets(action)
        >>> leader.disconnect()
    """

    def __init__(self, env, cfg: SO101LeaderCfg | None = None):
        """Initialize SO101 Leader device.

        Args:
            env: IsaacSim environment
            cfg: Configuration for the device. If None, uses default SO101LeaderCfg.
        """
        super().__init__()

        if cfg is None:
            cfg = SO101LeaderCfg()
        self.cfg = cfg
        self.env = env
        
        
        # Setup calibration file path
        self.calibration_path = os.path.join(
            os.path.dirname(__file__),
            "calibration",
            self.cfg.calibration_file_name
        )

        # Load calibration
        if not os.path.exists(self.calibration_path) or self.cfg.recalibrate:
            logger.info("Running calibration to create new calibration file.")
            self.calibrate()
        calibration = self._load_calibration()

        # Create motor bus
        self._bus = FeetechMotorsBus(
            port=self.cfg.port,
            motors=SO101_MOTORS,
            calibration=calibration,
        )
        self._motor_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS


    def reset(self):
        """Reset the device state."""
        if self.is_connected:
            self.configure()


    def _get_raw_data(self) -> dict:
        """Read raw joint positions from the physical robot.

        Returns:
            Raw result from the motor bus `sync_read("Present_Position")`.
        """
        try:
            # sync_read with normalize=True returns -100 to 100 range
            raw_data = self._bus.sync_read("Present_Position")
            return raw_data
        except Exception as e:
            logger.error(f"Error reading from motor bus: {e}")
            return {}


    @check_if_not_connected
    def advance(self) -> torch.Tensor:
        """Read joint positions and return as torch.Tensor."""
        # DEGREES 모드: 도 단위로 반환됨
        raw_data = self._get_raw_data() 
        
        values = []
        for name in SO101_MOTOR_NAMES:
            if name in raw_data:
                degree = raw_data[name]  # 이미 도!
                # 🔴 도 → 라디안으로만 변환
                radian = degree / 180.0 * math.pi
                values.append(radian)
            else:
                values.append(0.0)
        
        action = torch.tensor(values, dtype=torch.float32)
        return action

     
    
    @property
    def motor_limits(self) -> dict[str, tuple[float, float]]:
        """Get motor limits (USD joint limits)."""
        return self._motor_limits
    
    @property
    def is_connected(self) -> bool:
        """Check if motor bus is connected."""
        return self._bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration file exists and matches motors."""
        return self._bus.is_calibrated

    @check_if_not_connected
    def disconnect(self):
        """Disconnect the motor bus."""
        self._bus.disconnect()
        logger.info("SO101-Leader disconnected.")

    @check_if_already_connected
    def connect(self) -> None:
        """Connect the motor bus and configure motors."""
        self._bus.connect()
        self.configure()
        logger.info("SO101-Leader connected.")

    def configure(self):
        """Configure motors for position control."""
        try:
            self._bus.disable_torque()
            self._bus.configure_motors()
            for motor in self._bus.motors:
                self._bus.write(
                    "Operating_Mode",
                    motor,
                    OperatingMode.POSITION.value
                )
            logger.info("✓ Motors configured")
        except Exception as e:
            logger.error(f"✗ Failed to configure: {e}")

    def calibrate(self):
        """Run interactive calibration and save results."""
        self._bus = FeetechMotorsBus(
            port=self.cfg.port,
            motors=SO101_MOTORS,
            calibration=None,  # ← calibration 없음
        )
        
        if self._bus.calibration:
                user_input = input(
                    "Press ENTER to use existing calibration or type 'c' to run new calibration: "
                )
                if user_input.strip().lower() != "c":
                    logger.info("Using existing calibration.")
                    self._bus.write_calibration(self._bus.calibration)
                    return

        logger.info("="*60)
        logger.info("          SO101 LEADER CALIBRATION")
        logger.info("="*60 + "\n")
        self.connect()
        self._bus.disable_torque()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # Step 1: Set middle position
        logger.info("[Step 1/2] Setting middle position for all joints")
        logger.info("Move all joints to approximately the middle of their range.")
        input("Press ENTER when ready...")

        homing_offset = self._bus.set_half_turn_homings()
        logger.info("✓ Middle position set")

        # Step 2: Record range
        logger.info("[Step 2/2] Recording joint ranges")
        logger.info("Move each joint through its entire range of motion.")
        logger.info("Move all joints through their entire ranges of motion.\nRecording positions. Press ENTER to stop...")
        range_mins, range_maxes = self._bus.record_ranges_of_motion()
        logger.info("✓ Ranges recorded")

        # Build calibration dict
        calibration = {}
        for motor, m in self._bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offset[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        # Save calibration
        self._bus.write_calibration(calibration)
        self._save_calibration(calibration)

        logger.info(f"✓ Calibration saved to {self.calibration_path}")
        logger.info("="*60 + "\n")
        
        self.disconnect()
  
    def _load_calibration(self) -> dict[str, MotorCalibration]:
        """Load calibration from JSON file."""
        try:
            with open(self.calibration_path) as f:
                json_data = json.load(f)

            calibration = {}
            for motor_name, motor_data in json_data.items():
                calibration[motor_name] = MotorCalibration(
                    id=int(motor_data["id"]),
                    drive_mode=int(motor_data["drive_mode"]),
                    homing_offset=int(motor_data["homing_offset"]),
                    range_min=int(motor_data["range_min"]),
                    range_max=int(motor_data["range_max"]),
                )
            return calibration
        except FileNotFoundError:
            logger.error(f"Calibration file not found: {self.calibration_path}")
            raise

    def _save_calibration(self, calibration: dict[str, MotorCalibration]):
        """Save calibration to JSON file."""
        save_data = {
            k: {
                "id": v.id,
                "drive_mode": v.drive_mode,
                "homing_offset": v.homing_offset,
                "range_min": v.range_min,
                "range_max": v.range_max,
            }
            for k, v in calibration.items()
        }

        # Create directory if needed
        calibration_dir = os.path.dirname(self.calibration_path)
        if not os.path.exists(calibration_dir):
            os.makedirs(calibration_dir)

        # Save JSON
        with open(self.calibration_path, "w") as f:
            json.dump(save_data, f, indent=4)
        logger.info(f"✓ Calibration saved to {self.calibration_path}")
