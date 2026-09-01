# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for teleoperation interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch


class DeviceBase(ABC):
    """An interface class for teleoperation devices.

    Derived classes have two implementation options:

    1. Override _get_raw_data() and use the base advance() implementation:
       This approach is suitable for devices that want to leverage the built-in
       retargeting logic but only need to customize the raw data acquisition.

    2. Override advance() completely:
       This approach gives full control over the command generation process,
       and _get_raw_data() can be ignored entirely.
    """

    def __init__(self):
        """Initialize the teleoperation interface."""
        pass


    def __str__(self) -> str:
        """Returns: A string identifier for the device."""
        return f"{self.__class__.__name__}"

    """
    Operations
    """

    @abstractmethod
    def reset(self):
        """Reset the internals."""
        raise NotImplementedError


    @abstractmethod
    def _get_raw_data(self) -> Any:
        """Internal method to get the raw data from the device.

        This method is intended for internal use by the advance() implementation.
        Derived classes can override this method to customize raw data acquisition
        while still using the base class's advance() implementation.

        Returns:
            Raw device data in a device-specific format

        Note:
            This is an internal implementation detail. Clients should call advance()
            instead of this method.
        """
        raise NotImplementedError("Derived class must implement _get_raw_data() or override advance()")
    
    
    def advance(self) -> torch.Tensor:
        """Process current device state and return control commands."""
        return self._get_raw_data()

    def add_callback(self, key: Any, func: Callable):
        """Add additional functions to bind keyboard.

        Args:
            key: The button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        raise NotImplementedError