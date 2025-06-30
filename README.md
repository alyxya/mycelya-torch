# torch-modal

A PyTorch extension that implements a custom "modal" device type using PyTorch's PrivateUse1 backend. The modal device enforces strict device compatibility - operations can only be performed between tensors on the same modal device, while falling back to CPU for actual computation.

## Features

- **Custom Device Backend**: Modal device type using PyTorch's PrivateUse1 infrastructure
- **Tensor Method Extension**: `.modal()` method to move tensors to modal device  
- **Device Compatibility Enforcement**: Operations require all tensors to be on modal device
- **CPU Fallback**: Actual computation performed on CPU with device tracking
- **Comprehensive C++ Integration**: Full PrivateUse1HooksInterface implementation
- **Memory Management**: Custom storage and device daemon for tensor lifecycle

## Installation

```bash
git clone <repository-url>
cd pytorch-modal
pip install -e .
```

## Usage

### Basic Usage

```python
import torch
import torch_modal

# Create tensors and move to modal device
x = torch.randn(2, 2).modal()
y = torch.randn(2, 2).modal()

print(f"x device: {x.device}")  # modal:0
print(f"y device: {y.device}")  # modal:0

# Operations work between modal tensors
z = x.mm(y)  # Success - both on modal device
result = x + y  # Success - element-wise operations

# Operations fail between modal and CPU tensors
cpu_tensor = torch.randn(2, 2)
# z = x.mm(cpu_tensor)  # Raises RuntimeError!
```

### Device Management

```python
import torch_modal

# Check device availability
print(f"Modal available: {torch_modal.modal.is_available()}")
print(f"Device count: {torch_modal.modal.device_count()}")
print(f"Device name: {torch_modal.modal.get_device_name()}")

# Move tensors with options
x = torch.randn(3, 3, dtype=torch.float32)
y = x.modal(dtype=torch.float64, copy=True)  # Convert dtype and copy
```

## Testing

### Quick Test (Recommended)
```bash
python test_torch_modal.py
```

### Debug Mode
```bash
python test_torch_modal.py --debug
```

### Verbose Testing
```bash
python test_torch_modal.py --verbose
```

The test suite includes comprehensive coverage of:
- Import and C extension functionality
- Device availability and properties
- Modal tensor creation and operations
- Mixed device operation handling
- Parameter validation (dtype, copy, etc.)
- Error handling and edge cases

## Architecture

### Core Components

- **`torch_modal/__init__.py`** - Package initialization and PrivateUse1 device registration
- **`torch_modal/modal.py`** - Main modal device implementation and tensor operations
- **`torch_modal/modal/__init__.py`** - Device management functions (is_available, device_count, etc.)
- **`torch_modal/utils.py`** - Tensor method extensions (`.modal()` method injection)

### Backend Implementation

- **`torch_modal/_aten_impl.py`** - ATen operator implementations for modal device
- **`torch_modal/_meta_parser.py`** - Metadata parsing for tensor operations
- **`torch_modal/_device_daemon.py`** - Device lifecycle and memory management

### C++ Extension

- **`torch_modal/csrc/modal_extension.cpp`** - Main C++ extension entry point
- **`torch_modal/csrc/ModalHooks.cpp`** - PrivateUse1HooksInterface implementation
- **`torch_modal/csrc/ModalMem.cpp`** - Custom memory management for modal tensors
- **`torch_modal/csrc/Modal.h`** - C++ header definitions

### Build System

- **`setup.py`** - Package configuration with CppExtension build support
- **`TESTING.md`** - Comprehensive testing documentation and guidelines

## Development

The project uses a clean separation between Python device management and C++ backend implementation. The modal device leverages PyTorch's PrivateUse1 hooks to provide a fully integrated custom device experience while maintaining compatibility with PyTorch's tensor ecosystem.

Key design principles:
- **Device Isolation**: Strict enforcement of device compatibility
- **CPU Fallback**: Transparent computation delegation to CPU
- **Memory Safety**: Proper tensor lifecycle management
- **PyTorch Integration**: Native integration with PyTorch's device infrastructure