# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import atexit
import contextvars
import random
from typing import Any, Callable, Dict, Optional, Set

from ._logging import get_logger

# Context variable to track when lazy allocation should be used
_lazy_allocation_context = contextvars.ContextVar("lazy_allocation", default=False)


def lazy_allocation_context():
    """Context manager to enable lazy allocation for storage creation within the context"""
    from contextlib import contextmanager

    @contextmanager
    def _context():
        token = _lazy_allocation_context.set(True)
        try:
            yield
        finally:
            _lazy_allocation_context.reset(token)

    return _context()


log = get_logger(__name__)

# Simple storage ID tracking for remote storages
# No local device simulation - remote storages exist purely as IDs

# Constants for storage ID generation
MIN_STORAGE_ID = 1
MAX_STORAGE_ID = 2**64 - 1
MAX_ID_GENERATION_ATTEMPTS = 1000


def register(registry: Dict[str, Callable]) -> Callable[[Callable], Callable]:
    """Decorator to register functions in a registry dictionary.

    This decorator adds the decorated function to the provided registry
    using the function's name as the key. Used for registering driver
    commands that can be called by name.

    Args:
        registry: Dictionary to register the function in

    Returns:
        Decorator function that registers and returns the original function
    """

    def func(fn: Callable) -> Callable:
        registry[fn.__name__] = fn
        return fn

    return func


class RemoteStorageRegistry:
    """
    Simplified registry to track remote storage IDs only.

    Architecture:
    - storage_id: Identifies remote memory allocation on clients
    - PyTorch tensors: Handle local identity and metadata naturally
    - Remote operations: Receive storage_id + tensor metadata (shape, stride, offset, storage_id)
    - Storage cleanup: Handles remote storage cleanup when tensors are freed
    """

    def __init__(self) -> None:
        # Storage ID tracking - maps storage to device
        self.storage_id_to_device: Dict[int, int] = {}  # storage_id -> device_index

        # Set for tracking all generated storage IDs to avoid duplicates
        self.generated_storage_ids: Set[int] = set()

        # Current device tracking
        self._current_device: int = 0

        # Stream management
        self._current_streams: Dict[int, int] = {}  # device_idx -> stream_id
        self._stream_counter: Dict[int, int] = {}  # device_idx -> counter

    def generate_storage_id(self) -> int:
        """
        Generate a unique storage ID with duplicate validation.

        Returns:
            int: A unique 64-bit storage ID
        """
        # Generate unique 64-bit integer IDs
        # Use non-zero values to avoid confusion with null pointers
        attempt = 0

        while attempt < MAX_ID_GENERATION_ATTEMPTS:
            # Generate a random 64-bit integer
            storage_id = random.randint(MIN_STORAGE_ID, MAX_STORAGE_ID)

            # Check if this ID is already used
            if storage_id not in self.generated_storage_ids:
                self.generated_storage_ids.add(storage_id)
                log.info(f"🆔 GENERATED Storage ID: {storage_id}")
                return storage_id

            attempt += 1
            log.warning(
                f"Generated duplicate storage ID {storage_id}, retrying (attempt {attempt})"
            )

        # If we couldn't generate a unique ID after MAX_ID_GENERATION_ATTEMPTS, raise an error
        raise RuntimeError(
            f"Failed to generate unique storage ID after {MAX_ID_GENERATION_ATTEMPTS} attempts"
        )

    def create_storage_with_id(
        self, storage_id: int, nbytes: int, device_index: int, lazy: bool = False
    ) -> bool:
        """Create remote storage with the given ID and return success status"""
        storage_id = int(storage_id)

        # Check if lazy allocation is enabled in the current context
        if not lazy:
            lazy = _lazy_allocation_context.get(False)

        # Always track the storage ID for all tensors
        self.storage_id_to_device[storage_id] = device_index

        # Register the storage with the client immediately for all allocations
        try:
            # Import here to avoid circular imports
            from ._remote_orchestrator import remote_orchestrator
            from .device import get_device_registry

            orchestrator = remote_orchestrator
            if orchestrator is not None:
                registry = get_device_registry()
                device = registry.get_device_by_index(device_index)

                if device is not None:
                    client = device.get_client()
                    if client and client.is_running():
                        # Create storage with exact byte size
                        # No garbage data needed
                        client.create_storage(nbytes, storage_id, lazy)
                        log.info(
                            f"Registered storage {storage_id} with client "
                            f"({nbytes} bytes)"
                        )
        except Exception as e:
            log.warning(f"Failed to register storage {storage_id} with client: {e}")

        log.info(f"Registered storage ID {storage_id} on device {device_index}")
        return True

    def get_storage_device(self, storage_id: int) -> Optional[int]:
        """Get device index for a storage ID"""
        storage_id = int(storage_id)
        return self.storage_id_to_device.get(storage_id)

    def free_storage_with_id(self, storage_id: int) -> bool:
        """Free storage by storage ID and perform remote cleanup"""
        storage_id = int(storage_id)
        if storage_id == 0:  # Empty storage
            return True

        # Get device index before cleanup
        device_idx = self.storage_id_to_device.get(storage_id)

        if storage_id in self.storage_id_to_device:
            # Clean up storage tracking
            self.storage_id_to_device.pop(storage_id, None)
            self.generated_storage_ids.discard(storage_id)

            # Call remote cleanup
            if device_idx is not None:
                log.info(f"Storage {storage_id} freed, initiating remote cleanup")
                self._cleanup_remote_storage(storage_id, device_idx)
            else:
                log.info(
                    f"No device index found for storage {storage_id}, "
                    "skipping remote cleanup"
                )

            log.info(f"Freed storage ID {storage_id}")
        else:
            log.warning(f"Attempted to free unknown storage {storage_id}")

        return True

    def _cleanup_remote_storage(self, storage_id: int, device_idx: int) -> None:
        """Clean up storage on remote GPU device"""
        try:
            # Import here to avoid circular imports
            from ._remote_orchestrator import remote_orchestrator
            from .device import get_device_registry

            orchestrator = remote_orchestrator
            if orchestrator is None:
                log.warning(
                    f"No remote orchestrator available for storage {storage_id} cleanup"
                )
                return

            registry = get_device_registry()
            device = registry.get_device_by_index(device_idx)

            if device is None:
                log.warning(
                    f"No device found for index {device_idx} during storage "
                    f"{storage_id} cleanup"
                )
                return

            # Attempt remote cleanup
            log.info(f"Calling remove_storage_from_remote for storage {storage_id}")
            success = orchestrator.remove_tensor_from_remote(storage_id, device)
            if success:
                log.info(
                    f"✅ Successfully cleaned up remote storage {storage_id} "
                    f"on device {device_idx}"
                )
            else:
                log.warning(
                    f"❌ Remote cleanup returned false for storage {storage_id} "
                    f"on device {device_idx}"
                )

        except Exception as e:
            # Log but don't fail - local cleanup already completed
            log.warning(
                f"Failed remote cleanup for storage {storage_id} on device {device_idx}: {e}"
            )
            # Continue execution since local cleanup is already done

    def copy_data_by_id(self, dest_id: int, src_id: int, count: int) -> bool:
        """Copy data between storages identified by their storage IDs"""
        # This is a placeholder - actual copy operations should go through remote execution
        dest_device = self.storage_id_to_device.get(int(dest_id))
        src_device = self.storage_id_to_device.get(int(src_id))

        if dest_device is None or src_device is None:
            raise RuntimeError(
                f"Copy failed: storage IDs {dest_id} or {src_id} not found"
            )

        # For storage ID system, copy operations should go through remote
        # aten execution. This is just a placeholder to indicate the operation
        # was requested
        log.info(f"Storage copy requested: {src_id} -> {dest_id} ({count} bytes)")
        return True

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

    def resize_storage_by_id(self, storage_id: int, nbytes: int) -> bool:
        """Resize remote storage by storage ID"""
        try:
            storage_id = int(storage_id)

            # Get device index for this storage
            device_idx = self.get_storage_device(storage_id)
            if device_idx is None:
                log.warning(f"No device found for storage {storage_id}")
                return False

            # Import here to avoid circular imports
            from ._remote_orchestrator import remote_orchestrator
            from .device import get_device_registry

            orchestrator = remote_orchestrator
            if orchestrator is None:
                log.warning(
                    f"No remote orchestrator available for storage {storage_id} resize"
                )
                return False

            registry = get_device_registry()
            device = registry.get_device_by_index(device_idx)

            if device is None:
                log.warning(
                    f"No device found for index {device_idx} during storage {storage_id} resize"
                )
                return False

            # Get client and call resize_storage
            client = device.get_client()
            if client and client.is_running():
                success = client.resize_storage(storage_id, nbytes)
                if success:
                    log.info(
                        f"✅ Successfully resized remote storage {storage_id} to {nbytes} bytes"
                    )
                else:
                    log.warning(
                        f"❌ Remote resize returned false for storage {storage_id}"
                    )
                return success
            else:
                log.warning(f"Client not available for storage {storage_id} resize")
                return False

        except Exception as e:
            log.warning(f"Failed to resize remote storage {storage_id}: {e}")
            return False

    # Stream management methods
    def get_stream(self, device_idx: int) -> int:
        """Get current stream ID for device"""
        return self._current_streams.get(device_idx, 0)

    def create_new_stream(self, device_idx: int, priority: int = 0) -> int:
        """Create a new stream ID"""
        counter = self._stream_counter.get(device_idx, 0) + 1
        self._stream_counter[device_idx] = counter
        return counter

    def exchange_stream(self, stream) -> int:
        """Exchange current stream with new stream"""
        device_idx = stream.device_index
        old_stream = self._current_streams.get(device_idx, 0)
        self._current_streams[device_idx] = stream.stream_id
        return old_stream


class Driver:
    """Simplified driver that manages storage IDs with registry-based dispatch"""

    registry: Dict[str, Callable] = {}

    def __init__(self) -> None:
        self.registry_obj = RemoteStorageRegistry()

        # Register this instance for cleanup
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Clean up storage ID mappings on exit"""
        self.registry_obj.storage_id_to_device.clear()
        self.registry_obj.generated_storage_ids.clear()

    def exec(self, cmd: str, *args: Any) -> Any:
        """Execute a command using the registry pattern"""
        log.info(f"Executing command: {cmd}(*{args[:2]}...)")  # Limit args in log

        if cmd in Driver.registry:
            res = Driver.registry[cmd](self, *args)
        else:
            raise RuntimeError(f"Unknown command: {cmd}")

        log.info(f"Command {cmd} result: {res}")
        return res

    # Storage operations
    @register(registry)
    def generate_storage_id(self) -> int:
        return self.registry_obj.generate_storage_id()

    @register(registry)
    def create_storage_with_id(
        self, storage_id: int, nbytes: int, device_index: int
    ) -> bool:
        result = self.registry_obj.create_storage_with_id(
            storage_id, nbytes, device_index
        )
        if not result:
            raise RuntimeError(
                f"Failed to create storage with ID {storage_id} ({nbytes} bytes) on device {device_index}"
            )
        return result

    @register(registry)
    def free_storage_with_id(self, storage_id: int) -> bool:
        result = self.registry_obj.free_storage_with_id(storage_id)
        if not result:
            raise RuntimeError(f"Failed to free storage with ID {storage_id}")
        return result

    @register(registry)
    def get_storage_device(self, storage_id: int) -> Optional[int]:
        return self.registry_obj.get_storage_device(storage_id)

    @register(registry)
    def copy_data_by_id(self, dest_id: int, src_id: int, count: int) -> bool:
        return self.registry_obj.copy_data_by_id(dest_id, src_id, count)

    @register(registry)
    def resize_storage_by_id(self, storage_id: int, nbytes: int) -> bool:
        return self.registry_obj.resize_storage_by_id(storage_id, nbytes)

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
        if device_idx is None:
            device_idx = self.registry_obj.get_device()
        return self.registry_obj.get_stream(device_idx)

    @register(registry)
    def get_new_stream(self, device_idx: int, priority: int = 0) -> int:
        return self.registry_obj.create_new_stream(device_idx, priority)

    @register(registry)
    def exchange_stream(self, stream) -> int:
        return self.registry_obj.exchange_stream(stream)

    @register(registry)
    def query_stream(self, stream) -> bool:
        # Always return True (stream is ready)
        return True

    @register(registry)
    def synchronize_stream(self, stream) -> None:
        # No-op for remote streams
        pass

    @register(registry)
    def get_default_stream(self, device_idx: Optional[int] = None) -> Any:
        # Return default stream (0) for device
        if device_idx is None:
            device_idx = self.registry_obj.get_device()
        import torch

        return torch.Stream(device=torch.device("remote", device_idx), priority=0)

    @register(registry)
    def get_stream_from_global_pool(
        self, device_idx: Optional[int] = None, is_high_priority: bool = False
    ) -> Any:
        # Return a stream from global pool (just use default stream for now)
        if device_idx is None:
            device_idx = self.registry_obj.get_device()
        import torch

        return torch.Stream(device=torch.device("remote", device_idx), priority=0)

    # Event operations
    @register(registry)
    def synchronize_event(self, event) -> None:
        # No-op for remote events
        pass

    @register(registry)
    def record(self, event, stream, device_index, flags) -> None:
        # No-op for event recording on remote
        pass

    @register(registry)
    def destroy_event(self, event, device_index) -> None:
        # No-op for event destruction on remote
        pass

    @register(registry)
    def block(self, event, stream) -> None:
        # No-op for event blocking on remote
        pass

    @register(registry)
    def query_event(self, event) -> bool:
        # Always return True (event is ready)
        return True

    @register(registry)
    def elapsed_time(self, event1, event2, device_index) -> float:
        # Return 0 for elapsed time between events
        return 0.0

    # Data operations
    @register(registry)
    def record_data_ptr_on_stream(self, data_ptr_int64, stream) -> None:
        # No-op for data pointer recording
        pass


# Global driver instance
driver = Driver()

# Register global cleanup for all contexts
atexit.register(driver._cleanup)
