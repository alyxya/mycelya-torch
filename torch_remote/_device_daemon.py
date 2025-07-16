# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import atexit
import logging
import weakref

import torch

log = logging.getLogger(__name__)

# Simple tensor ID tracking for remote tensors
# No local device simulation - remote tensors exist purely as IDs


def register(registry):
    def func(fn):
        registry[fn.__name__] = fn
        return fn
    return func


class RemoteTensorRegistry:
    """
    Simple registry to track remote tensor IDs.
    No local data storage - remote tensors exist purely as tensor IDs.
    """
    
    def __init__(self):
        # Tensor ID to metadata mapping for efficient tensor operations
        self.tensor_id_to_meta = {}  # tensor_id -> RemoteTensorMeta
        self.tensor_id_to_tensor = {}  # tensor_id -> torch.Tensor (weak references)
        
        # Track which remote device owns each tensor ID
        self.tensor_id_to_device = {}  # tensor_id -> device_index
        
        # Device count for validation
        self.device_count = 2

    def create_tensor_with_id(self, tensor_id, nbytes, device_index):
        """Create a tensor with the given ID and return success status"""
        tensor_id_int = int(tensor_id)
        
        # For empty tensors, just return success
        if nbytes == 0:
            return True
            
        # Track which device owns this tensor ID
        self.tensor_id_to_device[tensor_id_int] = device_index
        
        log.info(f"Registered tensor ID {tensor_id_int} on device {device_index}")
        return True
    
    def register_tensor_mapping(self, tensor_id, tensor, meta):
        """Register a tensor ID to tensor and metadata mapping"""
        tensor_id_int = int(tensor_id)
        self.tensor_id_to_meta[tensor_id_int] = meta
        # Store weak reference to avoid circular references
        self.tensor_id_to_tensor[tensor_id_int] = weakref.ref(tensor)
    
    def get_tensor_by_id(self, tensor_id):
        """Get tensor by its ID"""
        tensor_id_int = int(tensor_id)
        if tensor_id_int in self.tensor_id_to_tensor:
            tensor_ref = self.tensor_id_to_tensor[tensor_id_int]
            return tensor_ref()  # Dereference weak reference
        return None
    
    def get_meta_by_id(self, tensor_id):
        """Get tensor metadata by its ID"""
        tensor_id_int = int(tensor_id)
        return self.tensor_id_to_meta.get(tensor_id_int)

    def free_tensor_with_id(self, tensor_id):
        """Free tensor by tensor ID"""
        tensor_id_int = int(tensor_id)
        if tensor_id_int == 0:  # Empty tensor
            return True
            
        # Clean up all mappings
        self.tensor_id_to_device.pop(tensor_id_int, None)
        self.tensor_id_to_meta.pop(tensor_id_int, None)
        self.tensor_id_to_tensor.pop(tensor_id_int, None)
        
        log.info(f"Freed tensor ID {tensor_id_int}")
        return True

    def copy_data_by_id(self, dest_id, src_id, count):
        """Copy data between tensors identified by their IDs"""
        # This is a placeholder - actual copy operations should go through remote execution
        dest_device = self.tensor_id_to_device.get(int(dest_id))
        src_device = self.tensor_id_to_device.get(int(src_id))
        
        if dest_device is None or src_device is None:
            raise RuntimeError(f"Copy failed: tensor IDs {dest_id} or {src_id} not found")
        
        # For tensor ID system, copy operations should go through remote execution
        # This is just a placeholder to indicate the operation was requested
        log.info(f"Copy requested: {src_id} -> {dest_id} ({count} bytes)")
        return True

    def device_count_method(self):
        """Return number of devices"""
        return self.device_count

    def get_device(self):
        """Get current device index"""
        return 0  # Default device

    def set_device(self, device_idx):
        """Set current device index"""
        if device_idx < 0 or device_idx >= self.device_count:
            raise ValueError(f"Invalid device index: {device_idx}")
        # For tensor ID system, device selection is handled by remote execution
        log.info(f"Device set to {device_idx}")

    def has_primary_context(self, device_idx):
        """Check if device has primary context"""
        return device_idx >= 0 and device_idx < self.device_count


class Driver:
    """Simplified driver that only manages tensor IDs without local simulation"""
    
    def __init__(self):
        self.registry_obj = RemoteTensorRegistry()
        
        # Register this instance for cleanup
        atexit.register(self._cleanup)

    def _cleanup(self):
        """Clean up tensor ID mappings on exit"""
        self.registry_obj.tensor_id_to_meta.clear()
        self.registry_obj.tensor_id_to_tensor.clear()
        self.registry_obj.tensor_id_to_device.clear()

    def exec(self, cmd, *args):
        """Execute a command on the tensor ID registry"""
        log.info(f"Executing command: {cmd}(*{args[:2]}...)")  # Limit args in log
        
        if hasattr(self.registry_obj, cmd):
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
    registry = {}

    @register(registry)
    def deviceCount(self, *args):
        return self.registry_obj.device_count_method()

    @register(registry)
    def getDevice(self):
        return self.registry_obj.get_device()

    @register(registry)
    def setDevice(self, device_idx):
        return self.registry_obj.set_device(device_idx)

    @register(registry)
    def uncheckedSetDevice(self, device_idx):
        return self.registry_obj.set_device(device_idx)

    @register(registry)
    def exchangeDevice(self, device_idx):
        old_device = self.registry_obj.get_device()
        self.registry_obj.set_device(device_idx)
        return old_device

    @register(registry)
    def hasPrimaryContext(self, device_idx):
        return self.registry_obj.has_primary_context(device_idx)

    @register(registry)
    def create_tensor_with_id(self, tensor_id, nbytes, device_index):
        return self.registry_obj.create_tensor_with_id(tensor_id, nbytes, device_index)

    @register(registry)
    def register_tensor_mapping(self, tensor_id, tensor, meta):
        return self.registry_obj.register_tensor_mapping(tensor_id, tensor, meta)

    @register(registry)
    def get_tensor_by_id(self, tensor_id):
        return self.registry_obj.get_tensor_by_id(tensor_id)

    @register(registry)
    def get_meta_by_id(self, tensor_id):
        return self.registry_obj.get_meta_by_id(tensor_id)

    @register(registry)
    def free_tensor_with_id(self, tensor_id):
        return self.registry_obj.free_tensor_with_id(tensor_id)

    @register(registry)
    def copy_data_by_id(self, dest_id, src_id, count):
        return self.registry_obj.copy_data_by_id(dest_id, src_id, count)


# Global driver instance
driver = Driver()

# Register global cleanup for all contexts
atexit.register(driver._cleanup)