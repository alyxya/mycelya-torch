# torch-modal

A minimalistic PyTorch extension that adds a custom "modal" device type. The modal device functions identically to CPU but enforces that operations can only be performed between tensors on the same modal device.

## Features

- Custom "modal" device type using PyTorch's PrivateUse1 backend
- `.modal()` method to move tensors to modal device
- Device compatibility checking - operations require all tensors to be on modal device
- Fallback to CPU implementations for actual computation

## Installation

```bash
cd pytorch-modal
pip install -e .
```

## Usage

```python
import torch
import torch_modal

# Create tensors and move to modal device
x = torch.randn(2, 2).modal()
y = torch.randn(2, 2).modal()

# Operations work between modal tensors
z = x.mm(y)  # Success

# Operations fail between modal and CPU tensors
cpu_tensor = torch.randn(2, 2)
# z = x.mm(cpu_tensor)  # Raises error!
```

## Testing

```bash
python test_modal.py
```

## Architecture

- `torch_modal/__init__.py` - Main package initialization and device registration
- `torch_modal/modal/__init__.py` - Modal device module with device management functions
- `torch_modal/utils.py` - Tensor method additions (`.modal()`)
- `torch_modal/csrc/modal_extension.cpp` - C++ extension with device compatibility checking