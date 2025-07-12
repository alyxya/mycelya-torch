# torch-remote

A PyTorch extension that implements a custom "remote" device type using PyTorch's PrivateUse1 backend. The remote device automatically routes computations to cloud GPU providers for high-performance tensor operations, with Modal as the first supported provider.

## Features

- **Remote Device**: A new PyTorch device type that automatically routes computations to cloud GPU providers
- **Multi-Provider Support**: Designed to support multiple cloud providers (Modal, RunPod, etc.) with Modal as the first implementation
- **Multi-GPU Support**: Supports a wide range of GPU types including T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, and B200
- **Automatic Dispatch**: Operations on remote tensors are automatically executed on cloud GPUs
- **Seamless Integration**: Works with existing PyTorch code with minimal changes
- **High Performance**: Leverages high-end cloud GPUs for compute-intensive operations
- **Private Package Architecture**: Remote execution code is isolated in a separate package to prevent import conflicts
- **Device-Specific GPU Routing**: Automatically selects appropriate GPU type based on workload requirements
- **Comprehensive C++ Integration**: Full PrivateUse1HooksInterface implementation
- **Memory Management**: Custom storage and device daemon for tensor lifecycle

## Installation

```bash
pip install -e .
```

This will install both the main `torch_remote` package and the private `torch_remote_execution` package needed for remote execution.

## Usage

### Basic Usage

```python
import torch
import torch_remote
from torch_remote import create_modal_device

# Create a Modal device with A100-40GB GPU
device = create_modal_device("A100-40GB")

# Create tensors on the remote device
x = torch.randn(3, 3, device=device.device())
y = torch.randn(3, 3, device=device.device())

# Operations automatically use cloud GPU
result = x + y  # Executed on A100-40GB GPU on Modal
print(result)
```

### Device API

The `BackendDevice` provides a `.device()` method to get a PyTorch device object:

```python
import torch
import torch_remote

# Create a backend device
backend_device = torch_remote.create_modal_device("A100-40GB")

# Get PyTorch device object
torch_device = backend_device.device()
print(f"Device: {torch_device}")  # Device: remote:0

# Use with PyTorch factory functions
x = torch.randn(3, 3, device=backend_device.device())
y = torch.zeros(5, 5, device=backend_device.device())

# Use with .to() method (BackendDevice directly supported)
z = torch.ones(2, 2).to(backend_device)
```

### Available GPU Types

```python
from torch_remote import create_modal_device, GPUType

# Create devices with different GPU types
t4_device = create_modal_device("T4")           # Entry-level GPU
l4_device = create_modal_device("L4")           # Mid-range GPU
a100_40_device = create_modal_device("A100-40GB")  # High-end GPU
a100_80_device = create_modal_device("A100-80GB")  # High-memory GPU
h100_device = create_modal_device("H100")       # Latest high-end GPU

# Or use the GPUType enum
device = create_modal_device(GPUType.A100_40GB)

# Supported GPU types: T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200
```

### Device Management and Validation

```python
import torch
import torch_remote
from torch_remote import create_modal_device

# Check device availability
print(f"Remote available: {torch.remote.is_available()}")
print(f"Device count: {torch.remote.device_count()}")

# Create device and get device info
device = create_modal_device("A100-40GB")
print(f"Device: {device}")
print(f"Device name: {device.device_name}")
print(f"GPU type: {device.gpu_type.value}")

# Device validation - tensors must be on the same device instance
device1 = create_modal_device("A100-40GB")
device2 = create_modal_device("A100-40GB")  # Different device instance

x = torch.randn(3, 3, device=device1.device())
y = torch.randn(3, 3, device=device2.device())
# Operations between different device instances will fail

# Move tensors with options
x = torch.randn(3, 3, dtype=torch.float32)
y = x.to(device, dtype=torch.float64, copy=True)  # Convert dtype and copy
```

## Testing

Run the test suite:

```bash
pytest test/test_torch_remote.py -v
```

Test remote execution manually:

```bash
python example_device_usage.py
```

The test suite includes comprehensive coverage of:
- Import and C extension functionality
- Device availability and properties
- Remote tensor creation and operations
- Remote execution functionality
- Mixed device operation handling
- Parameter validation (dtype, copy, etc.)
- Error handling and edge cases

## Architecture

### Core Components

- **`torch_remote/__init__.py`** - Package initialization and PrivateUse1 device registration
- **`torch_remote/backends/modal/__init__.py`** - Modal backend device management functions
- **`torch_remote/utils.py`** - Tensor method patches (`.to()` method enhancement for BackendDevice)

### Backend Implementation

- **`torch_remote/_aten_impl.py`** - ATen operator implementations for remote device with cloud execution
- **`torch_remote/_remote_executor.py`** - Remote execution system for cloud GPU operations
- **`torch_remote/_meta_parser.py`** - Metadata parsing for tensor operations
- **`torch_remote/_device_daemon.py`** - Device lifecycle and memory management

### C++ Extension

- **`torch_remote/csrc/remote_extension.cpp`** - Main C++ extension entry point
- **`torch_remote/csrc/RemoteHooks.cpp`** - PrivateUse1HooksInterface implementation
- **`torch_remote/csrc/RemoteMem.cpp`** - Custom memory management for remote tensors
- **`torch_remote/csrc/Remote.h`** - C++ header definitions

### Private Package Structure

- **`torch_remote_execution/`** - Private package containing cloud provider apps for remote execution
  - `modal_app.py` - Modal application with multi-GPU support (T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200)
  - `setup.py` - Separate installation configuration

### Build System

- **`setup.py`** - Package configuration with CppExtension build support

## Development

The project uses a clean separation between Python device management, C++ backend implementation, and remote execution. The remote device leverages PyTorch's PrivateUse1 hooks to provide a fully integrated custom device experience while automatically routing computations to cloud GPU providers.

Key design principles:
- **Multi-Provider Support**: Extensible architecture supporting multiple cloud GPU providers
- **Remote Execution**: Automatic dispatch of operations to cloud GPUs
- **Package Isolation**: Private package structure prevents import conflicts during remote execution
- **Memory Safety**: Proper tensor lifecycle management across local and remote execution
- **PyTorch Integration**: Native integration with PyTorch's device infrastructure
