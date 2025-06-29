import torch
import torch_modal._C

def device_count():
    """Return the number of modal devices available."""
    return 1

def current_device():
    """Return the current modal device index."""
    return 0

def set_device(device):
    """Set the current modal device."""
    if device != 0:
        raise RuntimeError(f"Modal device {device} not available. Only device 0 is supported.")

def is_available():
    """Return True if modal device is available."""
    return True

def get_device_name(device=None):
    """Get the name of a modal device."""
    return "Modal Device"

def get_device_properties(device=None):
    """Get properties of a modal device."""
    class Properties:
        def __init__(self):
            self.name = "Modal Device"
            self.total_memory = 0
            
    return Properties()

# Tensor types for modal device
FloatTensor = torch.Tensor
DoubleTensor = torch.Tensor
HalfTensor = torch.Tensor
BFloat16Tensor = torch.Tensor
ByteTensor = torch.Tensor
CharTensor = torch.Tensor
ShortTensor = torch.Tensor
IntTensor = torch.Tensor
LongTensor = torch.Tensor
BoolTensor = torch.Tensor