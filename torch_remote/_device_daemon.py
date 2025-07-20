# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import atexit
import io
import logging
import random
from typing import Any, Dict, Optional, Set, Callable, List

import torch

log = logging.getLogger(__name__)

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
    - storage_id: Identifies remote memory allocation on GPU machines
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
            log.warning(f"Generated duplicate storage ID {storage_id}, retrying (attempt {attempt})")

        # If we couldn't generate a unique ID after MAX_ID_GENERATION_ATTEMPTS, raise an error
        raise RuntimeError(f"Failed to generate unique storage ID after {MAX_ID_GENERATION_ATTEMPTS} attempts")


    def create_storage_with_id(self, storage_id: int, nbytes: int, device_index: int) -> bool:
        """Create remote storage with the given ID and return success status"""
        storage_id = int(storage_id)

        # Always track the storage ID for all tensors
        self.storage_id_to_device[storage_id] = device_index

        # Register the storage with the GPU machine immediately for all allocations
        try:
                # Import here to avoid circular imports
                from ._remote_orchestrator import remote_orchestrator
                from .device import get_device_registry

                orchestrator = remote_orchestrator
                if orchestrator is not None:
                    registry = get_device_registry()
                    device = registry.get_device_by_index(device_index)

                    if device is not None:
                        gpu_machine = device.get_gpu_machine()
                        if gpu_machine and gpu_machine.is_running():
                            # Create tensor data of the right size (handle 0-byte case)
                            if nbytes == 0:
                                # For 0-byte allocations, create a minimal empty tensor
                                empty_tensor = torch.empty(0, dtype=torch.float32)
                            else:
                                # For non-zero allocations, create appropriately sized tensor
                                empty_tensor = torch.empty(nbytes // 4, dtype=torch.float32)  # Assume float32 for now
                            
                            buffer = io.BytesIO()
                            torch.save(empty_tensor, buffer)
                            tensor_data = buffer.getvalue()

                            gpu_machine.create_storage(tensor_data, storage_id)
                            log.info(f"Registered storage {storage_id} with GPU machine ({nbytes} bytes)")
        except Exception as e:
            log.warning(f"Failed to register storage {storage_id} with GPU machine: {e}")

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
                log.info(f"No device index found for storage {storage_id}, skipping remote cleanup")

            log.info(f"Freed storage ID {storage_id}")
        else:
            log.warning(f"Attempted to free unknown storage {storage_id}")

        return True


    # Note: GPU registration is now handled directly in create_storage_with_id

    def _cleanup_remote_storage(self, storage_id: int, device_idx: int) -> None:
        """Clean up storage on remote GPU device"""
        try:
            # Import here to avoid circular imports
            from ._remote_orchestrator import remote_orchestrator
            from .device import get_device_registry
            orchestrator = remote_orchestrator
            if orchestrator is None:
                log.warning(f"No remote orchestrator available for storage {storage_id} cleanup")
                return

            registry = get_device_registry()
            device = registry.get_device_by_index(device_idx)

            if device is None:
                log.warning(f"No device found for index {device_idx} during storage {storage_id} cleanup")
                return

            # Attempt remote cleanup
            log.info(f"Calling remove_storage_from_remote for storage {storage_id}")
            success = orchestrator.remove_tensor_from_remote(storage_id, device)
            if success:
                log.info(f"✅ Successfully cleaned up remote storage {storage_id} on device {device_idx}")
            else:
                log.warning(f"❌ Remote cleanup returned false for storage {storage_id} on device {device_idx}")

        except Exception as e:
            # Log but don't fail - local cleanup already completed
            log.warning(f"Failed remote cleanup for storage {storage_id} on device {device_idx}: {e}")
            # Continue execution since local cleanup is already done

    def copy_data_by_id(self, dest_id: int, src_id: int, count: int) -> bool:
        """Copy data between storages identified by their storage IDs"""
        # This is a placeholder - actual copy operations should go through remote execution
        dest_device = self.storage_id_to_device.get(int(dest_id))
        src_device = self.storage_id_to_device.get(int(src_id))

        if dest_device is None or src_device is None:
            raise RuntimeError(f"Copy failed: storage IDs {dest_id} or {src_id} not found")

        # For storage ID system, copy operations should go through remote aten execution
        # This is just a placeholder to indicate the operation was requested
        log.info(f"Storage copy requested: {src_id} -> {dest_id} ({count} bytes)")
        return True


    def device_count_method(self) -> int:
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
        device_count = self.device_count_method()
        if device_idx < 0 or device_idx >= device_count:
            raise ValueError(f"Invalid device index: {device_idx}")
        old_device = self._current_device
        self._current_device = device_idx
        log.info(f"Device set from {old_device} to {device_idx}")
        return old_device

    def has_primary_context(self, device_idx: int) -> bool:
        """Check if device has primary context"""
        device_count = self.device_count_method()
        return device_idx >= 0 and device_idx < device_count


class Driver:
    """Simplified driver that only manages storage IDs without local simulation"""

    def __init__(self) -> None:
        self.registry_obj = RemoteStorageRegistry()

        # Register this instance for cleanup
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Clean up storage ID mappings on exit"""
        self.registry_obj.storage_id_to_device.clear()
        self.registry_obj.generated_storage_ids.clear()

    def exec(self, cmd: str, *args: Any) -> Any:
        """Execute a command on the storage ID registry"""
        log.info(f"Executing command: {cmd}(*{args[:2]}...)")  # Limit args in log

        # Handle operations that need special error handling
        if cmd in ["create_storage_with_id", "free_storage_with_id"]:
            if hasattr(self.registry_obj, cmd):
                method = getattr(self.registry_obj, cmd)
                result = method(*args)
                if not result:
                    storage_id = args[0]
                    if cmd == "create_storage_with_id":
                        nbytes, device_index = args[1], args[2]
                        raise RuntimeError(f"Failed to create storage with ID {storage_id} ({nbytes} bytes) on device {device_index}")
                    elif cmd == "free_storage_with_id":
                        raise RuntimeError(f"Failed to free storage with ID {storage_id}")
                return result
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
        elif hasattr(self.registry_obj, cmd):
            method = getattr(self.registry_obj, cmd)
            result = method(*args)
            log.info(f"Command {cmd} result: {result}")
            return result
        elif cmd == "deviceCount":
            return self.registry_obj.device_count_method()
        elif cmd == "getDevice":
            return self.registry_obj.get_device()
        elif cmd == "setDevice":
            return self.registry_obj.set_device(*args)
        elif cmd == "uncheckedSetDevice":
            return self.registry_obj.set_device(*args)
        elif cmd == "exchangeDevice":
            old_device = self.registry_obj.get_device()
            self.registry_obj.set_device(*args)
            return old_device
        elif cmd == "hasPrimaryContext":
            return self.registry_obj.has_primary_context(*args)
        elif cmd == "getStream":
            # Return current stream ID for the device (default 0)
            device_idx = args[0] if args else self.registry_obj.get_device()
            return getattr(self.registry_obj, '_current_streams', {}).get(device_idx, 0)
        elif cmd == "getNewStream":
            # Create a new stream ID
            device_idx, priority = args[0], args[1] if len(args) > 1 else 0
            if not hasattr(self.registry_obj, '_stream_counter'):
                self.registry_obj._stream_counter = {}
            counter = self.registry_obj._stream_counter.get(device_idx, 0) + 1
            self.registry_obj._stream_counter[device_idx] = counter
            return counter
        elif cmd == "exchangeStream":
            # Exchange current stream with new stream
            stream = args[0]
            device_idx = stream.device_index
            if not hasattr(self.registry_obj, '_current_streams'):
                self.registry_obj._current_streams = {}
            old_stream = self.registry_obj._current_streams.get(device_idx, 0)
            self.registry_obj._current_streams[device_idx] = stream.stream_id
            return old_stream
        elif cmd == "queryStream":
            # Always return True (stream is ready)
            return True
        elif cmd == "synchronizeStream":
            # No-op for remote streams
            return None
        elif cmd == "synchronizeEvent":
            # No-op for remote events
            return None
        elif cmd == "getDefaultStream":
            # Return default stream (0) for device
            device_idx = args[0] if args else self.registry_obj.get_device()
            import torch
            return torch.Stream(device=torch.device("remote", device_idx), priority=0)
        elif cmd == "getStreamFromGlobalPool":
            # Return a stream from global pool (just use default stream for now)
            device_idx = args[0] if args else self.registry_obj.get_device()
            is_high_priority = args[1] if len(args) > 1 else False
            import torch
            return torch.Stream(device=torch.device("remote", device_idx), priority=0)
        elif cmd == "record":
            # No-op for event recording on remote
            return None
        elif cmd == "destroyEvent":
            # No-op for event destruction on remote
            return None
        elif cmd == "block":
            # No-op for event blocking on remote
            return None
        elif cmd == "queryEvent":
            # Always return True (event is ready)
            return True
        elif cmd == "elapsedTime":
            # Return 0 for elapsed time between events
            return 0.0
        elif cmd == "recordDataPtrOnStream":
            # No-op for data pointer recording (args: data_ptr_int64, stream)
            return None
        elif cmd == "resize_storage_by_id":
            # Resize remote storage by storage ID
            storage_id, new_shape, dtype = args
            return self._resize_storage_by_id(storage_id, new_shape, dtype)
        else:
            raise RuntimeError(f"Unknown command: {cmd}")

    # Registry methods for C++ driver interface
    registry: Dict[str, Callable] = {}

    @register(registry)
    def deviceCount(self, *args: Any) -> int:
        return self.registry_obj.device_count_method()

    @register(registry)
    def getDevice(self) -> int:
        return self.registry_obj.get_device()

    @register(registry)
    def setDevice(self, device_idx: int) -> int:
        return self.registry_obj.set_device(device_idx)

    @register(registry)
    def uncheckedSetDevice(self, device_idx: int) -> int:
        return self.registry_obj.set_device(device_idx)

    @register(registry)
    def exchangeDevice(self, device_idx: int) -> int:
        old_device = self.registry_obj.get_device()
        self.registry_obj.set_device(device_idx)
        return old_device

    @register(registry)
    def hasPrimaryContext(self, device_idx: int) -> bool:
        return self.registry_obj.has_primary_context(device_idx)

    @register(registry)
    def create_storage_with_id(self, storage_id: int, nbytes: int, device_index: int) -> bool:
        return self.registry_obj.create_storage_with_id(storage_id, nbytes, device_index)

    @register(registry)
    def copy_data_by_id(self, dest_id: int, src_id: int, count: int) -> bool:
        return self.registry_obj.copy_data_by_id(dest_id, src_id, count)

    # Note: register_storage_with_gpu removed as it was redundant with create_storage_with_id

    @register(registry)
    def generate_storage_id(self) -> int:
        return self.registry_obj.generate_storage_id()

    @register(registry)
    def free_storage_with_id(self, storage_id: int) -> bool:
        return self.registry_obj.free_storage_with_id(storage_id)

    def _resize_storage_by_id(self, storage_id: int, new_shape: List[int], dtype: str) -> bool:
        """Resize remote storage by storage ID"""
        try:
            storage_id = int(storage_id)
            
            # Get device index for this storage
            device_idx = self.registry_obj.get_storage_device(storage_id)
            if device_idx is None:
                log.warning(f"No device found for storage {storage_id}")
                return False
            
            # Import here to avoid circular imports
            from ._remote_orchestrator import remote_orchestrator
            from .device import get_device_registry
            
            orchestrator = remote_orchestrator
            if orchestrator is None:
                log.warning(f"No remote orchestrator available for storage {storage_id} resize")
                return False
            
            registry = get_device_registry()
            device = registry.get_device_by_index(device_idx)
            
            if device is None:
                log.warning(f"No device found for index {device_idx} during storage {storage_id} resize")
                return False
            
            # Get GPU machine and call resize_storage
            gpu_machine = device.get_gpu_machine()
            if gpu_machine and gpu_machine.is_running():
                success = gpu_machine.resize_storage(storage_id, new_shape, dtype)
                if success:
                    log.info(f"✅ Successfully resized remote storage {storage_id} to shape {new_shape}")
                else:
                    log.warning(f"❌ Remote resize returned false for storage {storage_id}")
                return success
            else:
                log.warning(f"GPU machine not available for storage {storage_id} resize")
                return False
                
        except Exception as e:
            log.warning(f"Failed to resize remote storage {storage_id}: {e}")
            return False



# Global driver instance
driver = Driver()

# Register global cleanup for all contexts
atexit.register(driver._cleanup)
