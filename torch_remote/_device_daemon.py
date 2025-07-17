# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import atexit
import logging
import random
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
        
        # Set for tracking all generated tensor IDs to avoid duplicates
        self.generated_tensor_ids = set()
        
        # Current device tracking
        self._current_device = 0

    def generate_tensor_id(self):
        """
        Generate a unique tensor ID with duplicate validation.
        
        Returns:
            int: A unique 64-bit tensor ID
        """
        # Generate unique 64-bit integer IDs
        # Use non-zero values to avoid confusion with null pointers
        max_attempts = 1000  # Prevent infinite loops
        attempt = 0
        
        while attempt < max_attempts:
            # Generate a random 64-bit integer
            tensor_id = random.randint(1, 2**64 - 1)
            
            # Check if this ID is already used
            if tensor_id not in self.generated_tensor_ids:
                self.generated_tensor_ids.add(tensor_id)
                log.debug(f"Generated unique tensor ID: {tensor_id}")
                return tensor_id
            
            attempt += 1
            log.warning(f"Generated duplicate tensor ID {tensor_id}, retrying (attempt {attempt})")
        
        # If we couldn't generate a unique ID after max_attempts, raise an error
        raise RuntimeError(f"Failed to generate unique tensor ID after {max_attempts} attempts")

    def create_tensor_with_id(self, tensor_id, nbytes, device_index):
        """Create a tensor with the given ID and return success status"""
        tensor_id_int = int(tensor_id)
        
        # For empty tensors, just return success
        if nbytes == 0:
            return True
            
        # Track which device owns this tensor ID
        self.tensor_id_to_device[tensor_id_int] = device_index
        
        # Also register the tensor with the GPU machine immediately
        # Create empty tensor data for the allocation
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
                            
                            tensor_id_str = str(tensor_id_int)
                            gpu_machine.create_tensor(tensor_data, tensor_id_str)
                            log.info(f"Pre-registered tensor ID {tensor_id_int} with GPU machine")
            except Exception as e:
                log.warning(f"Failed to pre-register tensor {tensor_id_int} with GPU machine: {e}")
        
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
        """Free tensor by tensor ID with remote cleanup"""
        tensor_id_int = int(tensor_id)
        if tensor_id_int == 0:  # Empty tensor
            return True
            
        # Get device index before cleaning up mappings
        device_idx = self.tensor_id_to_device.get(tensor_id_int)
        
        # Clean up all local mappings
        self.tensor_id_to_device.pop(tensor_id_int, None)
        self.tensor_id_to_meta.pop(tensor_id_int, None)
        self.tensor_id_to_tensor.pop(tensor_id_int, None)
        self.generated_tensor_ids.discard(tensor_id_int)
        
        # Call remote cleanup if device is known
        if device_idx is not None:
            log.info(f"Initiating remote cleanup for tensor {tensor_id_int} on device {device_idx}")
            self._cleanup_remote_tensor(tensor_id_int, device_idx)
        else:
            log.info(f"No device index found for tensor {tensor_id_int}, skipping remote cleanup")
        
        log.info(f"Freed tensor ID {tensor_id_int}")
        return True

    def register_tensor_with_gpu(self, tensor_id, tensor_data):
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
                        tensor_id_str = str(tensor_id)
                        gpu_machine.create_tensor(tensor_data, tensor_id_str)
                        log.info(f"Registered tensor ID {tensor_id} with GPU machine")
                        return True
        except Exception as e:
            log.warning(f"Failed to register tensor {tensor_id} with GPU machine: {e}")
        return False

    def _cleanup_remote_tensor(self, tensor_id: int, device_idx: int):
        """Clean up tensor on remote GPU device"""
        try:
            # Import here to avoid circular imports
            from ._remote_executor import _get_remote_executor
            from .device import get_device_registry
            
            executor = _get_remote_executor()
            if executor is None:
                log.warning(f"No remote executor available for tensor {tensor_id} cleanup")
                return
            
            registry = get_device_registry()
            device = registry.get_device_by_index(device_idx)
            
            if device is None:
                log.warning(f"No device found for index {device_idx} during tensor {tensor_id} cleanup")
                return
            
            # Attempt remote cleanup
            log.info(f"Calling remove_tensor_from_remote for tensor {tensor_id}")
            success = executor.remove_tensor_from_remote(str(tensor_id), device)
            if success:
                log.info(f"✅ Successfully cleaned up remote tensor {tensor_id} on device {device_idx}")
            else:
                log.warning(f"❌ Remote cleanup returned false for tensor {tensor_id} on device {device_idx}")
                
        except Exception as e:
            # Log but don't fail - local cleanup already completed
            log.warning(f"Failed remote cleanup for tensor {tensor_id} on device {device_idx}: {e}")
            # Continue execution since local cleanup is already done

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
        # Get actual device count from the device registry
        from torch_remote.device import get_device_registry
        registry = get_device_registry()
        return len(registry._devices)

    def get_device(self):
        """Get current device index"""
        return self._current_device

    def set_device(self, device_idx):
        """Set current device index"""
        device_count = self.device_count_method()
        if device_idx < 0 or device_idx >= device_count:
            raise ValueError(f"Invalid device index: {device_idx}")
        old_device = self._current_device
        self._current_device = device_idx
        log.info(f"Device set from {old_device} to {device_idx}")
        return old_device

    def has_primary_context(self, device_idx):
        """Check if device has primary context"""
        device_count = self.device_count_method()
        return device_idx >= 0 and device_idx < device_count


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

    @register(registry)
    def register_tensor_with_gpu(self, tensor_id, tensor_data):
        return self.registry_obj.register_tensor_with_gpu(tensor_id, tensor_data)


# Global driver instance
driver = Driver()

# Register global cleanup for all contexts
atexit.register(driver._cleanup)