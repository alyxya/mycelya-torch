# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch

from ._logging import get_logger

log = get_logger(__name__)


def register(registry: Dict[str, Callable]) -> Callable[[Callable], Callable]:
    """Decorator to register functions in a registry dictionary.

    This decorator adds the decorated function to the provided registry
    using the function's name as the key. Used for registering driver
    commands that can be called by name.

    Args:
        registry: Dictionary to register the function in

    Returns:
        Decorator function that registers the wrapped function
    """

    def decorator(func: Callable) -> Callable:
        registry[func.__name__] = func
        return func

    return decorator


class DeviceRegistry:
    """
    Registry for device and stream management in mycelya-torch.

    Handles:
    - Current device tracking
    - Stream management for each device
    - Device count and validation
    """

    def __init__(self) -> None:
        # Current device tracking
        self._current_device: int = 0

        # Stream management - no torch.Stream objects, only stream IDs
        self._current_streams: Dict[int, int] = defaultdict(
            lambda: 0
        )  # device_idx -> current stream_id
        self._stream_registry: Dict[int, List[int]] = defaultdict(
            lambda: [0]
        )  # device_idx -> list of stream_ids

    def get_device_count(self) -> int:
        """Return number of devices"""
        # Get actual device count from the device registry
        from mycelya_torch.device import get_device_registry

        registry = get_device_registry()
        return len(registry._devices)

    def get_device(self) -> int:
        """Get current device index"""
        return self._current_device

    def set_device(self, device_idx: int) -> None:
        """Set current device index"""
        device_count = self.get_device_count()
        if device_idx < 0 or device_idx >= device_count:
            raise ValueError(f"Invalid device index: {device_idx}")
        old_device = self._current_device
        self._current_device = device_idx
        log.info(f"Device set from {old_device} to {device_idx}")

    def has_primary_context(self, device_idx: int) -> bool:
        """Check if device has primary context"""
        device_count = self.get_device_count()
        return device_idx >= 0 and device_idx < device_count

    def exchange_device(self, device_idx: int) -> int:
        """Exchange current device and return previous device index"""
        device_count = self.get_device_count()
        if device_idx < 0 or device_idx >= device_count:
            raise ValueError(f"Invalid device index: {device_idx}")

        old_device_idx = self._current_device
        self._current_device = device_idx
        log.info(f"Device exchanged from {old_device_idx} to {device_idx}")
        return old_device_idx

    # Stream management methods
    def get_stream(self, device_idx: int) -> int:
        """Get current stream ID for device"""
        current_stream_id = self._current_streams[
            device_idx
        ]  # defaultdict returns 0 if not set
        return current_stream_id

    def get_new_stream(self, device_idx: int, priority: int = 0) -> int:
        """Create new stream ID for device and add to registry"""
        # Get next stream ID (start from 1, since 0 is default)
        registry = self._stream_registry[
            device_idx
        ]  # defaultdict returns [0] if not set
        new_stream_id = len(
            registry
        )  # This will be 1 for first new stream (since [0] exists)

        # Add to registry
        registry.append(new_stream_id)

        # Set as current stream for this device
        self._current_streams[device_idx] = new_stream_id

        return new_stream_id

    def exchange_stream(self, stream_id: int, device_idx: int) -> int:
        """Exchange current stream ID and return previous stream ID"""
        # Get the previous current stream ID
        previous_stream_id = self._current_streams[
            device_idx
        ]  # defaultdict returns 0 if not set

        # Set the new stream as current
        self._current_streams[device_idx] = stream_id

        # Make sure this stream ID is in our registry
        registry = self._stream_registry[
            device_idx
        ]  # defaultdict returns [0] if not set
        if stream_id not in registry:
            registry.append(stream_id)

        return previous_stream_id


class Driver:
    """Driver that manages device operations with registry-based dispatch"""

    registry: Dict[str, Callable] = {}

    def __init__(self) -> None:
        self.registry_obj = DeviceRegistry()

    def exec(self, cmd: str, *args: Any) -> Any:
        """Execute a command using the registry pattern"""
        log.info(f"Executing command: {cmd}(*{args[:2]}...)")  # Limit args in log

        if cmd in Driver.registry:
            res = Driver.registry[cmd](self, *args)
        else:
            raise RuntimeError(f"Unknown command: {cmd}")

        log.info(f"Command {cmd} result: {res}")
        return res

    # Storage operations - delegate to _storage module
    @register(registry)
    def create_storage(self, nbytes: int, device_index: int) -> int:
        from ._storage import create_storage

        storage_id = create_storage(nbytes, device_index)
        if storage_id == 0:
            raise RuntimeError(
                f"Failed to create storage ({nbytes} bytes) on device {device_index}"
            )
        return storage_id

    @register(registry)
    def free_storage_with_id(self, storage_id: int) -> bool:
        from ._storage import free_storage_with_id

        return free_storage_with_id(storage_id)

    @register(registry)
    def get_storage_device(self, storage_id: int) -> Optional[int]:
        from ._storage import get_storage_device

        return get_storage_device(storage_id)

    @register(registry)
    def resize_storage_by_id(self, storage_id: int, nbytes: int) -> bool:
        from ._storage import resize_storage_by_id

        return resize_storage_by_id(storage_id, nbytes)

    # Device operations
    @register(registry)
    def device_count(self, *args: Any) -> int:
        assert len(args) == 0
        return self.registry_obj.get_device_count()

    @register(registry)
    def get_device(self) -> int:
        return self.registry_obj.get_device()

    @register(registry)
    def set_device(self, device_idx: int) -> None:
        return self.registry_obj.set_device(device_idx)

    @register(registry)
    def unchecked_set_device(self, device_idx: int) -> None:
        return self.registry_obj.set_device(device_idx)

    @register(registry)
    def exchange_device(self, device_idx: int) -> int:
        return self.registry_obj.exchange_device(device_idx)

    @register(registry)
    def has_primary_context(self, device_idx: int) -> bool:
        return self.registry_obj.has_primary_context(device_idx)

    # Stream operations
    @register(registry)
    def get_stream(self, device_idx: int) -> int:
        return self.registry_obj.get_stream(device_idx)

    @register(registry)
    def get_new_stream(self, device_idx: int, priority: int = 0) -> int:
        return self.registry_obj.get_new_stream(device_idx, priority)

    @register(registry)
    def exchange_stream(self, stream_id: int, device_idx: int) -> int:
        return self.registry_obj.exchange_stream(stream_id, device_idx)

    # Event operations (placeholders for now)
    @register(registry)
    def create_event(self, device_idx: int, flag: int) -> int:
        """Create event - placeholder implementation"""
        return 1

    @register(registry)
    def destroy_event(self, event: int, device_idx: int) -> None:
        """Destroy event - placeholder implementation"""
        pass

    @register(registry)
    def record(
        self, event: int, stream: torch.Stream, device_idx: int, flag: int
    ) -> None:
        """Record event - placeholder implementation"""
        pass

    @register(registry)
    def block(self, event: int, stream: torch.Stream) -> None:
        """Block on event - placeholder implementation"""
        pass

    @register(registry)
    def query_event(self, event: int) -> bool:
        """Query event - placeholder implementation"""
        return True

    @register(registry)
    def synchronize_event(self, event: int) -> None:
        """Synchronize event - placeholder implementation"""
        pass

    @register(registry)
    def query_stream(self, stream: torch.Stream) -> bool:
        """Query stream - placeholder implementation"""
        return True

    @register(registry)
    def synchronize_stream(self, stream: torch.Stream) -> None:
        """Synchronize stream - placeholder implementation"""
        pass

    @register(registry)
    def record_data_ptr_on_stream(self, data_ptr: int, stream: torch.Stream) -> None:
        """Record data pointer on stream - placeholder implementation"""
        pass

    @register(registry)
    def elapsed_time(self, event1: int, event2: int, device_idx: int) -> float:
        """Get elapsed time between events - placeholder implementation"""
        return 0.0


# Global driver instance
driver = Driver()
