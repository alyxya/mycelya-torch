# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Callable, Dict, Optional

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
    Registry for device and stream management in torch-remote.

    Handles:
    - Current device tracking
    - Stream management for each device
    - Device count and validation
    """

    def __init__(self) -> None:
        # Current device tracking
        self._current_device: int = 0

        # Stream management
        self._current_streams: Dict[int, int] = {}  # device_idx -> stream_id
        self._stream_counter: Dict[int, int] = {}  # device_idx -> counter

    def get_device_count(self) -> int:
        """Return number of devices"""
        # Get actual device count from the device registry
        from torch_remote.device import get_device_registry

        registry = get_device_registry()
        return len(registry._devices)

    def get_device(self) -> int:
        """Get current device index"""
        return self._current_device

    def set_device(self, device_idx: int) -> int:
        """Set current device index"""
        device_count = self.get_device_count()
        if device_idx < 0 or device_idx >= device_count:
            raise ValueError(f"Invalid device index: {device_idx}")
        old_device = self._current_device
        self._current_device = device_idx
        log.info(f"Device set from {old_device} to {device_idx}")
        return old_device

    def has_primary_context(self, device_idx: int) -> bool:
        """Check if device has primary context"""
        device_count = self.get_device_count()
        return device_idx >= 0 and device_idx < device_count

    # Stream management methods
    def get_stream(self, device_idx: Optional[int] = None) -> int:
        """Get current stream for device"""
        if device_idx is None:
            device_idx = self._current_device

        return self._current_streams.get(device_idx, 0)

    def get_default_stream(self, device_idx: int) -> int:
        """Get default stream for device"""
        # Default stream is always stream 0 for remote devices
        return 0

    def get_stream_from_global_pool(self, device_idx: int, high_priority: bool = False) -> int:
        """Get stream from global pool"""
        # For simplicity, just return a counter-based stream ID
        if device_idx not in self._stream_counter:
            self._stream_counter[device_idx] = 1

        stream_id = self._stream_counter[device_idx]
        self._stream_counter[device_idx] += 1
        return stream_id

    def get_new_stream(self, device_idx: int, priority: int = 0) -> int:
        """Get new stream for device"""
        if device_idx not in self._stream_counter:
            self._stream_counter[device_idx] = 1

        stream_id = self._stream_counter[device_idx]
        self._stream_counter[device_idx] += 1
        return stream_id

    def exchange_stream(self, stream_id: int) -> int:
        """Exchange current stream"""
        old_stream = self._current_streams.get(self._current_device, 0)
        self._current_streams[self._current_device] = stream_id
        return old_stream


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
    def generate_storage_id(self) -> int:
        from ._storage import generate_storage_id
        return generate_storage_id()

    @register(registry)
    def create_storage_with_id(
        self, storage_id: int, nbytes: int, device_index: int
    ) -> bool:
        from ._storage import create_storage_with_id
        result = create_storage_with_id(storage_id, nbytes, device_index)
        if not result:
            raise RuntimeError(
                f"Failed to create storage with ID {storage_id} ({nbytes} bytes) on device {device_index}"
            )
        return result

    @register(registry)
    def free_storage_with_id(self, storage_id: int) -> bool:
        from ._storage import free_storage_with_id
        result = free_storage_with_id(storage_id)
        if not result:
            raise RuntimeError(f"Failed to free storage with ID {storage_id}")
        return result

    @register(registry)
    def get_storage_device(self, storage_id: int) -> Optional[int]:
        from ._storage import get_storage_device
        return get_storage_device(storage_id)

    @register(registry)
    def copy_data_by_id(self, dest_id: int, src_id: int, count: int) -> None:
        from ._storage import copy_data_by_id
        return copy_data_by_id(dest_id, src_id, count)

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
    def set_device(self, device_idx: int) -> int:
        return self.registry_obj.set_device(device_idx)

    @register(registry)
    def unchecked_set_device(self, device_idx: int) -> int:
        return self.registry_obj.set_device(device_idx)

    @register(registry)
    def exchange_device(self, device_idx: int) -> int:
        old_device = self.registry_obj.get_device()
        self.registry_obj.set_device(device_idx)
        return old_device

    @register(registry)
    def has_primary_context(self, device_idx: int) -> bool:
        return self.registry_obj.has_primary_context(device_idx)

    # Stream operations
    @register(registry)
    def get_stream(self, device_idx: Optional[int] = None) -> int:
        return self.registry_obj.get_stream(device_idx)

    @register(registry)
    def get_default_stream(self, device_idx: int) -> int:
        return self.registry_obj.get_default_stream(device_idx)

    @register(registry)
    def get_stream_from_global_pool(self, device_idx: int, high_priority: bool = False) -> int:
        return self.registry_obj.get_stream_from_global_pool(device_idx, high_priority)

    @register(registry)
    def get_new_stream(self, device_idx: int, priority: int = 0) -> int:
        return self.registry_obj.get_new_stream(device_idx, priority)

    @register(registry)
    def exchange_stream(self, stream_id: int) -> int:
        return self.registry_obj.exchange_stream(stream_id)

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
    def record(self, event: int, stream: int, device_idx: int, flag: int) -> None:
        """Record event - placeholder implementation"""
        pass

    @register(registry)
    def block(self, event: int, stream: int) -> None:
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
    def query_stream(self, stream: int) -> bool:
        """Query stream - placeholder implementation"""
        return True

    @register(registry)
    def synchronize_stream(self, stream: int) -> None:
        """Synchronize stream - placeholder implementation"""
        pass

    @register(registry)
    def record_data_ptr_on_stream(self, data_ptr: int, stream: int) -> None:
        """Record data pointer on stream - placeholder implementation"""
        pass

    @register(registry)
    def elapsed_time(self, event1: int, event2: int, device_idx: int) -> float:
        """Get elapsed time between events - placeholder implementation"""
        return 0.0


# Global driver instance
driver = Driver()
