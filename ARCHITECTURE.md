# torch-modal Architecture Documentation

This document provides a detailed explanation of the torch-modal codebase architecture, including the purpose and functionality of each source file.

## Overview

torch-modal implements a custom PyTorch device that automatically routes computations to Modal's A100 GPUs. The system uses a three-layer architecture with private package isolation to prevent Modal job import conflicts.

## Architecture Layers

### 1. PyTorch Integration Layer (C++ Extension)
Registers "modal" as a native PyTorch PrivateUse1 device with full integration into PyTorch's dispatch system.

### 2. Local Device Simulation Layer (Python)
Manages modal tensors using CPU memory with threading-based device simulation for local operations.

### 3. Remote Execution Layer (Modal)
Automatically dispatches compute-intensive operations to A100 GPUs via Modal's infrastructure.

## Package Structure

```
torch_modal/                    # Main PyTorch extension package
├── __init__.py                 # Package initialization & device registration
├── _aten_impl.py              # Operation dispatch & remote execution logic  
├── _modal_remote.py           # Remote execution system
├── _meta_parser.py            # Tensor metadata & data structures
├── _device_daemon.py          # Device management & process communication
├── utils.py                   # Tensor method extensions
├── modal/
│   └── __init__.py            # Device management functions
└── csrc/                      # C++ extension
    ├── modal_extension.cpp    # Python extension entry point
    ├── ModalHooks.cpp         # PrivateUse1 backend implementation
    ├── ModalMem.cpp           # Memory management
    └── Modal.h                # C++ header definitions

torch_modal_remote/            # Private package for remote execution
├── __init__.py               # Private package marker
├── app.py                    # Modal A100 execution app
└── setup.py                  # Private package installation
```

## Source File Details

### torch_modal Package (Main Package)

#### Core Module Files

**`torch_modal/__init__.py`** - Package Initialization & Device Registration
- Registers "modal" as a PyTorch PrivateUse1 backend device
- Creates the `torch.modal` module with device management functions:
  - `device()` - Context manager for device selection
  - `device_count()` - Number of available modal devices
  - `current_device()` - Get current device index
  - `is_available()` - Check device availability
- Sets up random number generation, streams, and device context management
- Exposes remote execution configuration APIs:
  - `enable_remote_execution()` - Enable A100 GPU execution
  - `disable_remote_execution()` - Use local execution only
  - `is_remote_execution_enabled()` - Check current setting
- Imports and initializes the C++ extension

**`torch_modal/_aten_impl.py`** - Operation Dispatch & Remote Execution Logic
- **Primary dispatch system**: Handles all ATen operations on modal tensors
- **Remote execution decision logic**: `_should_use_remote_execution()` determines which operations should run on A100 GPUs vs locally
- **Operation filtering**: 
  - Skip lists for memory ops (copy_, resize_, etc.)
  - Factory functions (empty, zeros, ones)
  - View operations (reshape, transpose, etc.) that should stay local
- **Compute-intensive operation routing**: Automatically routes to remote A100s:
  - Matrix operations (mm, bmm, addmm)
  - Neural network operations (conv2d, linear, relu, softmax)
  - Reduction operations (sum, mean, var, std)
  - Large tensor operations (>1000 elements)
- **Fallback mechanisms**: Local execution when remote is unavailable
- **Factory function support**: Handles tensor creation operations on modal device
- **Library registration**: Registers modal device implementations for specific PyTorch operations

**`torch_modal/_modal_remote.py`** - Remote Execution System
- **ModalRemoteExecutor class**: Manages remote execution on Modal's A100 GPUs
- **Tensor serialization/deserialization**: Converts modal tensors to/from bytes for network transport
- **Modal app integration**: Interfaces with the `torch_modal_remote` package
- **Error handling and fallbacks**: Graceful degradation when Modal is unavailable
- **Ephemeral execution**: Uses Modal's run contexts for one-off operations without persistent deployments
- **Device conversion helpers**: Converts between modal tensors and CPU tensors for transport

**`torch_modal/_meta_parser.py`** - Tensor Metadata & Data Structures
- **ModalTensorMeta class**: Captures tensor metadata (shape, dtype, strides, storage info) for serialization
- **ModalTensorData class**: Custom tensor subclass that:
  - Reports "modal" device but stores data on CPU
  - Overrides `.device` property to return `torch.device("modal", index)`
  - Provides proper `.cpu()` method that returns regular torch.Tensor
- **Serialization helpers**: Convert tensors to/from metadata for inter-process communication
- **Device spoofing**: Makes CPU tensors appear as modal device tensors to PyTorch
- **Validation functions**: Ensures only valid data types pass through device boundaries

**`torch_modal/_device_daemon.py`** - Device Management & Process Communication
- **Driver class**: Main coordinator for device operations and memory management
- **Threading-based execution**: Uses Python threads instead of multiprocessing to avoid hanging issues
- **Memory allocators**: 
  - `Allocator` - Base allocator class with malloc/free interface
  - `HostAllocator` - Manages pinned host memory
  - `DeviceAllocator` - Manages device memory and tensor reconstruction
- **Device simulation**: Simulates 2 modal devices using CPU memory
- **Stream and event management**: PyTorch CUDA-like stream semantics for modal device
- **Cleanup handling**: Signal handlers and atexit hooks for proper resource cleanup
- **Operation execution**: Routes operations to worker threads via queue-based communication
- **_Executor class**: Worker thread that actually performs tensor operations

#### Utility Files

**`torch_modal/utils.py`** - Tensor Method Extensions
- Adds `.modal()` method to `torch.Tensor` class
- Enables `tensor.modal()` to move tensors to modal device
- Simple wrapper around the C++ modal conversion function

**`torch_modal/modal/__init__.py`** - Device Management Functions
- Device availability, count, and property queries
- RNG state management for modal device
- Synchronization primitives
- Tensor type aliases (FloatTensor, DoubleTensor, etc.)

#### C++ Extension

**`torch_modal/csrc/modal_extension.cpp`** - Python Extension Entry Point
- PyTorch C++ extension initialization using PyBind11
- Exposes `_init()` function to initialize PrivateUse1 device
- Provides `_get_default_generator()` for random number generation
- Links Python factory functions to C++ implementation
- Sets up the bridge between Python and C++ components

**`torch_modal/csrc/ModalHooks.cpp`** - PrivateUse1 Backend Implementation
- Implements PyTorch's `PrivateUse1HooksInterface` for full device integration
- **Device management**: device count, current device, device guard implementation
- **Generator management**: Random number generators for modal device
- **Stream management**: Stream creation, synchronization, and querying
- **Memory management**: Host allocator integration
- **Event system**: Event creation, recording, and synchronization
- Integrates with Python-based device driver through method calls

**`torch_modal/csrc/ModalMem.cpp`** - Memory Management
- **ModalAllocator class**: Handles device memory allocation/deallocation
- Integrates with Python-based memory management system
- Routes allocation requests through Python driver
- Registers allocator with PyTorch's memory management system
- Handles memory cleanup and error reporting

**`torch_modal/csrc/Modal.h`** - C++ Header Definitions
- Common types and utilities for the C++ extension
- `modal_ptr_t` - Pointer type for modal device memory
- Python GIL management helpers
- Template functions for cleanup and error reporting
- Method lookup utilities for calling Python functions from C++

### torch_modal_remote Package (Private Package)

**`torch_modal_remote/__init__.py`** - Private Package Marker
- Simple package initialization with version information
- Documentation warning against direct use
- Marks package as internal to torch_modal

**`torch_modal_remote/app.py`** - Modal A100 Execution App
- **Modal application definition**: Creates Modal app "torch-modal-extension" with A100 GPU configuration
- **Docker image setup**: 
  - Debian slim base with Python 3.11
  - Installs PyTorch with CUDA 12.1 support
  - Includes torchvision, torchaudio, and nvidia-ml-py3
- **execute_aten_operation function**: The core function decorated with `@app.function` that runs on A100 GPUs:
  - Receives serialized tensors, metadata, args, and kwargs
  - Deserializes tensors and moves them to CUDA device
  - Processes tensor placeholders in arguments
  - Executes the requested ATen operation on A100 GPU
  - Serializes results and returns them
- **GPU utilization**: Automatically detects and uses CUDA when available
- **Error handling**: Comprehensive error reporting and traceback printing

**`torch_modal_remote/setup.py`** - Private Package Installation
- Separate setuptools configuration for the remote execution package
- Dependencies: modal>=0.60.0, torch>=2.0.0
- Marked as development status to discourage standalone installation
- Classifiers indicate it's for internal use

## Operation Flow

Here's how a typical operation flows through the system:

1. **User Code**: `result = torch.add(modal_tensor_a, modal_tensor_b)`

2. **PyTorch Dispatch**: PyTorch's dispatch system routes to modal device implementation

3. **_aten_impl.py**: 
   - `_modal_kernel_fallback` or `_kernel_fallback` receives the operation
   - `_should_use_remote_execution()` decides if this should run remotely
   - For compute-intensive ops: routes to remote execution
   - For simple ops: handles locally

4. **Remote Execution Path** (if enabled):
   - `ModalRemoteExecutor.execute_remote_operation()` is called
   - Tensors are serialized to bytes
   - Modal app is invoked with `app.run()` context
   - `torch_modal_remote.app.execute_aten_operation` runs on A100 GPU
   - Results are serialized and returned
   - Results are deserialized back to modal tensors

5. **Local Execution Path** (fallback):
   - Operation metadata is computed
   - Output tensors are allocated on modal device
   - Operation is executed via device daemon
   - Results are returned as modal tensors

## Key Design Decisions

### Private Package Isolation
The most important architectural decision is separating the Modal app code into `torch_modal_remote`. This prevents import conflicts when Modal jobs execute, since Modal would otherwise try to import the entire `torch_modal` extension and create circular dependencies.

### Threading vs Multiprocessing
The system uses threading instead of multiprocessing for device simulation to avoid complex cleanup issues that were causing hanging processes.

### Lazy Remote Execution
Remote execution is lazy-loaded and gracefully degrades when Modal is not available, allowing the extension to work in environments without Modal.

### Operation Filtering
Smart filtering ensures that only operations that benefit from A100 acceleration are sent remotely, while keeping memory operations, views, and small tensor operations local for efficiency.

### CPU Storage with Device Spoofing
Modal tensors are stored in CPU memory but report as "modal" device to PyTorch, enabling seamless integration with PyTorch's device system while maintaining compatibility.

## Configuration

Users can control remote execution behavior:

```python
import torch_modal

# Enable remote execution (default)
torch_modal.enable_remote_execution()

# Disable remote execution (use local execution)
torch_modal.disable_remote_execution()

# Check current setting
print(torch_modal.is_remote_execution_enabled())
```

## Testing

The system includes comprehensive testing:
- `test_torch_modal.py` - Unit tests for all functionality
- `demo_manual_remote.py` - Manual testing of remote execution
- Pytest-based test suite with cleanup handling

This architecture provides a seamless PyTorch device experience while leveraging Modal's A100 GPU infrastructure for high-performance computing.