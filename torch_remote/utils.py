import torch

def _remote(self, device=None, dtype=None, non_blocking=False, copy=False):
    """Move tensor to remote device."""
    from .device import BackendDevice, get_device_registry
    
    if device is None:
        device = torch.device("remote", 0)
    elif isinstance(device, BackendDevice):
        # Get device index from registry
        registry = get_device_registry()
        device_index = registry.get_device_index(device)
        if device_index is None:
            # Register the device if not already registered
            device_index = registry.register_device(device)
        device = torch.device("remote", device_index)
    elif isinstance(device, int):
        device = torch.device("remote", device)
    elif isinstance(device, str) and not device.startswith("remote"):
        device = torch.device("remote", 0)  # Default to remote:0 if just "remote"
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Move to device
    tensor = self.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
    
    # If moving to a BackendDevice, store the device ID in the tensor
    if isinstance(device, torch.device) and device.type == "remote":
        registry = get_device_registry()
        backend_device = registry.get_device_by_index(device.index)
        if backend_device is not None:
            tensor._device_id = backend_device.device_id
    
    return tensor

def _add_tensor_methods():
    """Add .remote() method to torch.Tensor."""
    torch.Tensor.remote = _remote


def _patch_torch_factory_functions():
    """Patch torch factory functions to support BackendDevice."""
    from .device import BackendDevice, get_device_registry
    
    # Store original functions
    _original_randn = torch.randn
    _original_zeros = torch.zeros
    _original_ones = torch.ones
    _original_empty = torch.empty
    _original_tensor = torch.tensor
    
    def _process_device_arg(device):
        """Process device argument to handle BackendDevice."""
        if isinstance(device, BackendDevice):
            registry = get_device_registry()
            device_index = registry.get_device_index(device)
            if device_index is None:
                device_index = registry.register_device(device)
            return torch.device("remote", device_index), device.device_id
        return device, None
    
    def _attach_device_id(tensor, device_id):
        """Attach device ID to tensor if applicable."""
        if device_id is not None:
            tensor._device_id = device_id
        return tensor
    
    def patched_randn(*args, **kwargs):
        device = kwargs.get('device')
        if device is not None:
            device, device_id = _process_device_arg(device)
            kwargs['device'] = device
            tensor = _original_randn(*args, **kwargs)
            return _attach_device_id(tensor, device_id)
        return _original_randn(*args, **kwargs)
    
    def patched_zeros(*args, **kwargs):
        device = kwargs.get('device')
        if device is not None:
            device, device_id = _process_device_arg(device)
            kwargs['device'] = device
            tensor = _original_zeros(*args, **kwargs)
            return _attach_device_id(tensor, device_id)
        return _original_zeros(*args, **kwargs)
    
    def patched_ones(*args, **kwargs):
        device = kwargs.get('device')
        if device is not None:
            device, device_id = _process_device_arg(device)
            kwargs['device'] = device
            tensor = _original_ones(*args, **kwargs)
            return _attach_device_id(tensor, device_id)
        return _original_ones(*args, **kwargs)
    
    def patched_empty(*args, **kwargs):
        device = kwargs.get('device')
        if device is not None:
            device, device_id = _process_device_arg(device)
            kwargs['device'] = device
            tensor = _original_empty(*args, **kwargs)
            return _attach_device_id(tensor, device_id)
        return _original_empty(*args, **kwargs)
    
    def patched_tensor(*args, **kwargs):
        device = kwargs.get('device')
        if device is not None:
            device, device_id = _process_device_arg(device)
            kwargs['device'] = device
            tensor = _original_tensor(*args, **kwargs)
            return _attach_device_id(tensor, device_id)
        return _original_tensor(*args, **kwargs)
    
    # Replace torch functions
    torch.randn = patched_randn
    torch.zeros = patched_zeros
    torch.ones = patched_ones
    torch.empty = patched_empty
    torch.tensor = patched_tensor