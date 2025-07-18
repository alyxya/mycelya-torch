# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import atexit
import logging
import random
import weakref
from typing import Any, Dict, Optional, Set, Callable

import torch

log = logging.getLogger(__name__)

# Simple tensor ID tracking for remote tensors
# No local device simulation - remote tensors exist purely as IDs


def register(registry: Dict[str, Callable]) -> Callable[[Callable], Callable]:
    def func(fn: Callable) -> Callable:
        registry[fn.__name__] = fn
        return fn
    return func


class RemoteTensorRegistry:
    """
    Simplified registry to track remote storage IDs only.
    
    Architecture:
    - storage_id: Identifies remote memory allocation (what was "tensor_id")
    - PyTorch tensors: Handle local identity and metadata naturally
    - Remote operations: Receive storage_id + tensor metadata (shape, stride, offset)
    """
    
    def __init__(self) -> None:
        # Storage ID tracking - maps storage to device and reference count
        self.storage_id_to_device: Dict[int, int] = {}  # storage_id -> device_index  
        self.storage_id_ref_count: Dict[int, int] = {}  # storage_id -> reference count
        
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
        max_attempts = 1000  # Prevent infinite loops
        attempt = 0
        
        while attempt < max_attempts:
            # Generate a random 64-bit integer
            storage_id = random.randint(1, 2**64 - 1)
            
            # Check if this ID is already used
            if storage_id not in self.generated_storage_ids:
                self.generated_storage_ids.add(storage_id)
                log.debug(f"Generated unique storage ID: {storage_id}")
                return storage_id
            
            attempt += 1
            log.warning(f"Generated duplicate storage ID {storage_id}, retrying (attempt {attempt})")
        
        # If we couldn't generate a unique ID after max_attempts, raise an error
        raise RuntimeError(f"Failed to generate unique storage ID after {max_attempts} attempts")
    
    # Backward compatibility - rename method but keep old name working
    def generate_tensor_id(self) -> int:
        """Legacy method name - now generates storage ID."""
        return self.generate_storage_id()

    def create_storage_with_id(self, storage_id: int, nbytes: int, device_index: int) -> bool:
        """Create remote storage with the given ID and return success status"""
        storage_id_int = int(storage_id)
        
        # For empty tensors, just return success
        if nbytes == 0:
            return True
            
        # Track which device owns this storage ID
        self.storage_id_to_device[storage_id_int] = device_index
        self.storage_id_ref_count[storage_id_int] = 1
        
        # Register the storage with the GPU machine immediately
        if nbytes > 0:
            try:
                # Import here to avoid circular imports
                from ._remote_executor import _get_remote_executor
                from .device import get_device_registry
                
                executor = _get_remote_executor()
                if executor is not None:
                    registry = get_device_registry()
                    device = registry.get_device_by_index(device_index)
                    
                    if device is not None:
                        gpu_machine = device.get_gpu_machine()
                        if gpu_machine and gpu_machine.is_running():
                            # Create empty tensor data of the right size
                            import io
                            empty_tensor = torch.empty(nbytes // 4, dtype=torch.float32)  # Assume float32 for now
                            buffer = io.BytesIO()
                            torch.save(empty_tensor, buffer)
                            tensor_data = buffer.getvalue()
                            
                            storage_id_str = str(storage_id_int)
                            gpu_machine.create_tensor(tensor_data, storage_id_str)
                            log.info(f"Pre-registered storage ID {storage_id_int} with GPU machine")
            except Exception as e:
                log.warning(f"Failed to pre-register storage {storage_id_int} with GPU machine: {e}")
        
        log.info(f"Registered storage ID {storage_id_int} on device {device_index}")
        return True
    
    def get_storage_device(self, storage_id: int) -> Optional[int]:
        """Get device index for a storage ID"""
        storage_id_int = int(storage_id)
        return self.storage_id_to_device.get(storage_id_int)
    
    # Legacy method for backward compatibility
    def create_tensor_with_id(self, tensor_id: int, nbytes: int, device_index: int) -> bool:
        """Legacy method - now creates storage."""
        return self.create_storage_with_id(tensor_id, nbytes, device_index)

    def free_storage_with_id(self, storage_id: int) -> bool:
        """Free storage by storage ID with reference counting and remote cleanup"""
        storage_id_int = int(storage_id)
        if storage_id_int == 0:  # Empty storage
            return True
            
        # Decrement reference count
        if storage_id_int in self.storage_id_ref_count:
            self.storage_id_ref_count[storage_id_int] -= 1
            
            # Only cleanup remote storage if this is the last reference
            if self.storage_id_ref_count[storage_id_int] <= 0:
                # Get device index before cleanup
                device_idx = self.storage_id_to_device.get(storage_id_int)
                
                # Clean up storage tracking
                del self.storage_id_ref_count[storage_id_int]
                self.storage_id_to_device.pop(storage_id_int, None)
                self.generated_storage_ids.discard(storage_id_int)
                
                # Call remote cleanup
                if device_idx is not None:
                    log.info(f"Last reference to storage {storage_id_int} freed, initiating remote cleanup")
                    self._cleanup_remote_tensor(storage_id_int, device_idx)
                else:
                    log.info(f"No device index found for storage {storage_id_int}, skipping remote cleanup")
                    
                log.info(f"Freed storage ID {storage_id_int}")
            else:
                log.info(f"Storage {storage_id_int} still has {self.storage_id_ref_count[storage_id_int]} references, skipping cleanup")
        else:
            log.warning(f"Attempted to free unknown storage {storage_id_int}")
        
        return True
    
    # Legacy method for backward compatibility
    def free_tensor_with_id(self, tensor_id: int) -> bool:
        """Legacy method - now frees storage."""
        return self.free_storage_with_id(tensor_id)

    def register_tensor_with_gpu(self, tensor_id: int, tensor_data: bytes) -> bool:
        """Register tensor data with GPU machine for immediate access"""
        # This ensures that newly created tensors (including outputs) 
        # are immediately available on the GPU machine
        try:
            # Import here to avoid circular imports
            from ._remote_executor import _get_remote_executor
            from .device import get_device_registry
            
            executor = _get_remote_executor()
            if executor is not None:
                # Get the current remote device (assumes single device for now)
                registry = get_device_registry()
                device = registry.get_device_by_index(0)  # Use device 0
                
                if device is not None:
                    gpu_machine = device.get_gpu_machine()
                    if gpu_machine and gpu_machine.is_running():
                        storage_id_str = str(tensor_id)
                        gpu_machine.create_tensor(tensor_data, storage_id_str)
                        log.info(f"Registered storage ID {tensor_id} with GPU machine")
                        return True
        except Exception as e:
            log.warning(f"Failed to register storage {tensor_id} with GPU machine: {e}")
        return False

    def _cleanup_remote_tensor(self, tensor_id: int, device_idx: int) -> None:
        """Clean up storage on remote GPU device"""
        try:
            # Import here to avoid circular imports
            from ._remote_executor import _get_remote_executor
            from .device import get_device_registry
            
            executor = _get_remote_executor()
            if executor is None:
                log.warning(f"No remote executor available for storage {tensor_id} cleanup")
                return
            
            registry = get_device_registry()
            device = registry.get_device_by_index(device_idx)
            
            if device is None:
                log.warning(f"No device found for index {device_idx} during storage {tensor_id} cleanup")
                return
            
            # Attempt remote cleanup
            log.info(f"Calling remove_tensor_from_remote for storage {tensor_id}")
            success = executor.remove_tensor_from_remote(str(tensor_id), device)
            if success:
                log.info(f"✅ Successfully cleaned up remote storage {tensor_id} on device {device_idx}")
            else:
                log.warning(f"❌ Remote cleanup returned false for storage {tensor_id} on device {device_idx}")
                
        except Exception as e:
            # Log but don't fail - local cleanup already completed
            log.warning(f"Failed remote cleanup for storage {tensor_id} on device {device_idx}: {e}")
            # Continue execution since local cleanup is already done

    def copy_data_by_id(self, dest_id: int, src_id: int, count: int) -> bool:
        """Copy data between tensors identified by their IDs"""
        # This is a placeholder - actual copy operations should go through remote execution
        dest_device = self.storage_id_to_device.get(int(dest_id))
        src_device = self.storage_id_to_device.get(int(src_id))
        
        if dest_device is None or src_device is None:
            raise RuntimeError(f"Copy failed: tensor IDs {dest_id} or {src_id} not found")
        
        # For tensor ID system, copy operations should go through remote execution
        # This is just a placeholder to indicate the operation was requested
        log.info(f"Copy requested: {src_id} -> {dest_id} ({count} bytes)")
        return True
    
    # Placeholder methods for backward compatibility (these may not be fully implemented)
    def register_tensor_mapping(self, tensor_id: int, tensor: torch.Tensor, meta: Any) -> bool:
        """Placeholder method for backward compatibility"""
        return True
    
    def get_tensor_by_id(self, tensor_id: int) -> Optional[torch.Tensor]:
        """Placeholder method for backward compatibility"""
        return None
    
    def get_meta_by_id(self, tensor_id: int) -> Optional[Any]:
        """Placeholder method for backward compatibility"""
        return None
    
    def create_view_tensor_id(self, base_tensor_id: int, new_shape: Any, new_stride: Any, new_storage_offset: int) -> int:
        """Placeholder method for backward compatibility"""
        return self.generate_storage_id()
    
    def get_storage_id(self, tensor_id: int) -> int:
        """Placeholder method for backward compatibility"""
        return tensor_id
    
    def get_tensor_ids_for_storage(self, storage_id: int) -> list:
        """Placeholder method for backward compatibility"""
        return [storage_id]
    
    def is_view_tensor(self, tensor_id: int) -> bool:
        """Placeholder method for backward compatibility"""
        return False
    
    def get_base_tensor_id(self, tensor_id: int) -> int:
        """Placeholder method for backward compatibility"""
        return tensor_id
    
    def get_view_tensor_ids(self, base_tensor_id: int) -> list:
        """Placeholder method for backward compatibility"""
        return []

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
    """Simplified driver that only manages tensor IDs without local simulation"""
    
    def __init__(self) -> None:
        self.registry_obj = RemoteTensorRegistry()
        
        # Register this instance for cleanup
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Clean up storage ID mappings on exit"""
        self.registry_obj.storage_id_to_device.clear()
        self.registry_obj.storage_id_ref_count.clear()
        self.registry_obj.generated_storage_ids.clear()

    def exec(self, cmd: str, *args: Any) -> Any:
        """Execute a command on the tensor ID registry"""
        log.info(f"Executing command: {cmd}(*{args[:2]}...)")  # Limit args in log
        
        # Handle operations that need special error handling
        if cmd in ["create_tensor_with_id", "free_tensor_with_id"]:
            if hasattr(self.registry_obj, cmd):
                method = getattr(self.registry_obj, cmd)
                result = method(*args)
                if not result:
                    storage_id = args[0]
                    if cmd == "create_tensor_with_id":
                        nbytes, device_index = args[1], args[2] 
                        raise RuntimeError(f"Failed to create tensor with ID {storage_id} ({nbytes} bytes) on device {device_index}")
                    elif cmd == "free_tensor_with_id":
                        raise RuntimeError(f"Failed to free tensor with ID {storage_id}")
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
        else:
            raise RuntimeError(f"Unknown command: {cmd}")

    # Registry methods (for backward compatibility)
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
    def create_tensor_with_id(self, tensor_id: int, nbytes: int, device_index: int) -> bool:
        return self.registry_obj.create_tensor_with_id(tensor_id, nbytes, device_index)

    @register(registry)
    def register_tensor_mapping(self, tensor_id: int, tensor: torch.Tensor, meta: Any) -> bool:
        return self.registry_obj.register_tensor_mapping(tensor_id, tensor, meta)

    @register(registry)
    def get_tensor_by_id(self, tensor_id: int) -> Optional[torch.Tensor]:
        return self.registry_obj.get_tensor_by_id(tensor_id)

    @register(registry)
    def get_meta_by_id(self, tensor_id: int) -> Optional[Any]:
        return self.registry_obj.get_meta_by_id(tensor_id)

    @register(registry)
    def free_tensor_with_id(self, tensor_id: int) -> bool:
        return self.registry_obj.free_tensor_with_id(tensor_id)

    @register(registry)
    def copy_data_by_id(self, dest_id: int, src_id: int, count: int) -> bool:
        return self.registry_obj.copy_data_by_id(dest_id, src_id, count)

    @register(registry)
    def register_tensor_with_gpu(self, tensor_id: int, tensor_data: bytes) -> bool:
        return self.registry_obj.register_tensor_with_gpu(tensor_id, tensor_data)
    
    @register(registry)
    def create_view_tensor_id(self, base_tensor_id: int, new_shape: Any, new_stride: Any, new_storage_offset: int) -> int:
        return self.registry_obj.create_view_tensor_id(base_tensor_id, new_shape, new_stride, new_storage_offset)
    
    @register(registry)
    def get_storage_id(self, tensor_id: int) -> int:
        return self.registry_obj.get_storage_id(tensor_id)
    
    @register(registry)
    def get_tensor_ids_for_storage(self, storage_id: int) -> list:
        return self.registry_obj.get_tensor_ids_for_storage(storage_id)
    
    @register(registry)
    def is_view_tensor(self, tensor_id: int) -> bool:
        return self.registry_obj.is_view_tensor(tensor_id)
    
    @register(registry)
    def get_base_tensor_id(self, tensor_id: int) -> int:
        return self.registry_obj.get_base_tensor_id(tensor_id)
    
    @register(registry)
    def get_view_tensor_ids(self, base_tensor_id: int) -> list:
        return self.registry_obj.get_view_tensor_ids(base_tensor_id)


# Global driver instance
driver = Driver()

# Register global cleanup for all contexts
atexit.register(driver._cleanup)