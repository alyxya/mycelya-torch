# torch-modal

A PyTorch extension that implements a custom "modal" device type using PyTorch's PrivateUse1 backend. The modal device automatically routes computations to Modal's A100 GPUs for high-performance tensor operations.

## Features

- **Modal Device**: A new PyTorch device type that automatically routes computations to Modal's A100 GPUs
- **Automatic Dispatch**: Operations on modal tensors are automatically executed on A100 GPUs
- **Seamless Integration**: Works with existing PyTorch code with minimal changes
- **High Performance**: Leverages A100 GPUs for compute-intensive operations
- **Private Package Architecture**: Remote execution code is isolated in a separate package to prevent Modal job import conflicts
- **Configurable Execution**: Enable/disable remote execution as needed
- **Comprehensive C++ Integration**: Full PrivateUse1HooksInterface implementation
- **Memory Management**: Custom storage and device daemon for tensor lifecycle

## Installation

```bash
pip install -e .
```

This will install both the main `torch_modal` package and the private `torch_modal_remote` package needed for remote execution.

## Usage

### Basic Usage

```python
import torch
import torch_modal

# Create tensors on the modal device
x = torch.randn(3, 3, device="modal")
y = torch.randn(3, 3, device="modal")

# Operations automatically use A100 GPU via Modal
result = x + y  # Executed on A100 GPU
print(result)
```

### Remote Execution Configuration

You can control whether operations are executed remotely on A100 GPUs:

```python
import torch_modal

# Enable remote execution (default)
torch_modal.enable_remote_execution()

# Disable remote execution (use local execution)
torch_modal.disable_remote_execution()

# Check current setting
print(torch_modal.is_remote_execution_enabled())
```

### Device Management

```python
import torch
import torch_modal

# Check device availability
print(f"Modal available: {torch.modal.is_available()}")
print(f"Device count: {torch.modal.device_count()}")

# Move tensors with options
x = torch.randn(3, 3, dtype=torch.float32)
y = x.to("modal", dtype=torch.float64, copy=True)  # Convert dtype and copy
```

## Testing

Run the test suite:

```bash
pytest test_torch_modal.py -v
```

Test remote execution manually:

```bash
python demo_manual_remote.py
```

The test suite includes comprehensive coverage of:
- Import and C extension functionality
- Device availability and properties
- Modal tensor creation and operations
- Remote execution functionality
- Mixed device operation handling
- Parameter validation (dtype, copy, etc.)
- Error handling and edge cases

## Architecture

### Core Components

- **`torch_modal/__init__.py`** - Package initialization and PrivateUse1 device registration
- **`torch_modal/modal/__init__.py`** - Device management functions (is_available, device_count, etc.)
- **`torch_modal/utils.py`** - Tensor method extensions (`.modal()` method injection)

### Backend Implementation

- **`torch_modal/_aten_impl.py`** - ATen operator implementations for modal device with remote execution
- **`torch_modal/_modal_remote.py`** - Remote execution system for A100 GPU operations
- **`torch_modal/_meta_parser.py`** - Metadata parsing for tensor operations
- **`torch_modal/_device_daemon.py`** - Device lifecycle and memory management

### C++ Extension

- **`torch_modal/csrc/modal_extension.cpp`** - Main C++ extension entry point
- **`torch_modal/csrc/ModalHooks.cpp`** - PrivateUse1HooksInterface implementation
- **`torch_modal/csrc/ModalMem.cpp`** - Custom memory management for modal tensors
- **`torch_modal/csrc/Modal.h`** - C++ header definitions

### Private Package Structure

- **`torch_modal_remote/`** - Private package containing Modal app for remote execution
  - `app.py` - Modal application with A100 GPU execution functions
  - `setup.py` - Separate installation configuration

### Build System

- **`setup.py`** - Package configuration with CppExtension build support

## Development

The project uses a clean separation between Python device management, C++ backend implementation, and remote execution. The modal device leverages PyTorch's PrivateUse1 hooks to provide a fully integrated custom device experience while automatically routing computations to Modal's A100 GPUs.

Key design principles:
- **Remote Execution**: Automatic dispatch of operations to A100 GPUs via Modal
- **Package Isolation**: Private package structure prevents Modal job import conflicts
- **Memory Safety**: Proper tensor lifecycle management across local and remote execution
- **PyTorch Integration**: Native integration with PyTorch's device infrastructure
- **Configurable Execution**: Runtime control over local vs remote execution
