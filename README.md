# torch-remote

A PyTorch extension that implements a custom "remote" device type using PyTorch's PrivateUse1 backend. The remote device automatically routes computations to cloud GPU providers for high-performance tensor operations, with Modal as the first supported provider.

## Features

- **Remote Device**: A new PyTorch device type that automatically routes computations to cloud GPU providers
- **Multi-Provider Support**: Designed to support multiple cloud providers (Modal, RunPod, etc.) with Modal as the first implementation
- **Automatic Dispatch**: Operations on remote tensors are automatically executed on cloud GPUs
- **Seamless Integration**: Works with existing PyTorch code with minimal changes
- **High Performance**: Leverages high-end cloud GPUs for compute-intensive operations
- **Private Package Architecture**: Remote execution code is isolated in a separate package to prevent import conflicts
- **Configurable Execution**: Enable/disable remote execution as needed
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

# Create tensors on the remote device
x = torch.randn(3, 3, device="remote")
y = torch.randn(3, 3, device="remote")

# Operations automatically use cloud GPU
result = x + y  # Executed on cloud GPU
print(result)
```

### Remote Execution Configuration

You can control whether operations are executed remotely on cloud GPUs:

```python
import torch_remote

# Enable remote execution (default)
torch_remote.enable_remote_execution()

# Disable remote execution (use local execution)
torch_remote.disable_remote_execution()

# Check current setting
print(torch_remote.is_remote_execution_enabled())
```

### Device Management

```python
import torch
import torch_remote

# Check device availability
print(f"Remote available: {torch.remote.is_available()}")
print(f"Device count: {torch.remote.device_count()}")

# Move tensors with options
x = torch.randn(3, 3, dtype=torch.float32)
y = x.to("remote", dtype=torch.float64, copy=True)  # Convert dtype and copy
```

## Testing

Run the test suite:

```bash
pytest test_torch_remote.py -v
```

Test remote execution manually:

```bash
python demo_manual_remote.py
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
- **`torch_remote/utils.py`** - Tensor method extensions (`.remote()` method injection)

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
  - `app.py` - Modal application with A100 GPU execution functions
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
- **Configurable Execution**: Runtime control over local vs remote execution
