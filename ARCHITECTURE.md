# torch-remote Architecture Documentation

This document provides a detailed explanation of the torch-remote codebase architecture, including the purpose and functionality of each source file.

## Overview

torch-remote implements a custom PyTorch device that automatically routes computations to remote cloud providers. The system uses a three-layer architecture with a pluggable multi-provider backend system. Modal serves as the first supported provider, with additional providers planned for future releases. The architecture includes private package isolation to prevent provider-specific import conflicts.

## Architecture Layers

### 1. PyTorch Integration Layer (C++ Extension)
Registers "remote" as a native PyTorch PrivateUse1 device with full integration into PyTorch's dispatch system.

### 2. Local Device Simulation Layer (Python)
Manages remote tensors using CPU memory with threading-based device simulation for local operations.

### 3. Remote Execution Layer (Multi-Provider)
Automatically dispatches compute-intensive operations to cloud providers via a pluggable backend system. Currently supports Modal's A100 GPU infrastructure, with additional providers planned.

## Package Structure

```
torch_remote/                   # Main PyTorch extension package
├── __init__.py                 # Package initialization & device registration
├── _aten_impl.py              # Operation dispatch & remote execution logic  
├── _remote_execution.py       # Remote execution system
├── _meta_parser.py            # Tensor metadata & data structures
├── _device_daemon.py          # Device management & process communication
├── utils.py                   # Tensor method extensions
├── backends/                  # Multi-provider backend system
│   ├── __init__.py            # Backend registry & management
│   └── modal/                 # Modal provider implementation
│       ├── __init__.py        # Modal backend integration
│       └── _modal_remote.py   # Modal-specific remote execution
├── remote/
│   └── __init__.py            # Device management functions
└── csrc/                      # C++ extension
    ├── remote_extension.cpp   # Python extension entry point
    ├── RemoteHooks.cpp        # PrivateUse1 backend implementation
    ├── RemoteMem.cpp          # Memory management
    └── Remote.h               # C++ header definitions

torch_remote_execution/        # Private package for remote execution
├── __init__.py               # Private package marker
├── modal_app.py              # Modal multi-GPU execution app (T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200)
└── setup.py                  # Private package installation
```

## Source File Details

### torch_remote Package (Main Package)

#### Core Module Files

**`torch_remote/__init__.py`** - Package Initialization & Device Registration
- Registers "remote" as a PyTorch PrivateUse1 backend device
- Creates the `torch.remote` module with device management functions:
  - `device()` - Context manager for device selection
  - `device_count()` - Number of available remote devices
  - `current_device()` - Get current device index
  - `is_available()` - Check device availability
- Sets up random number generation, streams, and device context management
- Imports and initializes the C++ extension

**`torch_remote/_aten_impl.py`** - Operation Dispatch & Remote Execution Logic
- **Primary dispatch system**: Handles all ATen operations on remote tensors
- **Remote execution decision logic**: `_should_use_remote_execution()` determines which operations should run on cloud providers vs locally
- **Operation filtering**: 
  - Skip lists for memory ops (copy_, resize_, etc.)
  - Factory functions (empty, zeros, ones)
  - View operations (reshape, transpose, etc.) that should stay local
- **Compute-intensive operation routing**: Automatically routes to remote cloud providers:
  - Matrix operations (mm, bmm, addmm)
  - Neural network operations (conv2d, linear, relu, softmax)
  - Reduction operations (sum, mean, var, std)
  - Large tensor operations (>1000 elements)
- **Fallback mechanisms**: Local execution when remote is unavailable
- **Factory function support**: Handles tensor creation operations on remote device
- **Library registration**: Registers remote device implementations for specific PyTorch operations

**`torch_remote/_remote_execution.py`** - Remote Execution System
- **RemoteExecutor class**: Manages remote execution across cloud providers
- **Tensor serialization/deserialization**: Converts remote tensors to/from bytes for network transport
- **Provider backend integration**: Interfaces with the `torch_remote_backends` package
- **Error handling and fallbacks**: Graceful degradation when remote providers are unavailable
- **Multi-provider support**: Abstracts provider-specific execution details
- **Device conversion helpers**: Converts between remote tensors and CPU tensors for transport

**`torch_remote/_meta_parser.py`** - Tensor Metadata & Data Structures
- **RemoteTensorMeta class**: Captures tensor metadata (shape, dtype, strides, storage info) for serialization
- **RemoteTensorData class**: Custom tensor subclass that:
  - Reports "remote" device but stores data on CPU
  - Overrides `.device` property to return `torch.device("remote", index)`
  - Provides proper `.cpu()` method that returns regular torch.Tensor
- **Serialization helpers**: Convert tensors to/from metadata for inter-process communication
- **Device spoofing**: Makes CPU tensors appear as remote device tensors to PyTorch
- **Validation functions**: Ensures only valid data types pass through device boundaries

**`torch_remote/_device_daemon.py`** - Device Management & Process Communication
- **Driver class**: Main coordinator for device operations and memory management
- **Threading-based execution**: Uses Python threads instead of multiprocessing to avoid hanging issues
- **Memory allocators**: 
  - `Allocator` - Base allocator class with malloc/free interface
  - `HostAllocator` - Manages pinned host memory
  - `DeviceAllocator` - Manages device memory and tensor reconstruction
- **Device simulation**: Simulates 2 remote devices using CPU memory
- **Stream and event management**: PyTorch CUDA-like stream semantics for remote device
- **Cleanup handling**: Signal handlers and atexit hooks for proper resource cleanup
- **Operation execution**: Routes operations to worker threads via queue-based communication
- **_Executor class**: Worker thread that actually performs tensor operations

#### Utility Files

**`torch_remote/utils.py`** - Tensor Method Extensions
- Patches `.to()` method to support `BackendDevice` objects
- Enables `tensor.to(backend_device)` to move tensors to remote device
- Simple wrapper around the C++ remote conversion function

**`torch_remote/remote/__init__.py`** - Device Management Functions
- Device availability, count, and property queries
- RNG state management for remote device
- Synchronization primitives
- Tensor type aliases (FloatTensor, DoubleTensor, etc.)

#### Backend System Files

**`torch_remote/backends/__init__.py`** - Backend Registry & Management
- **Backend registry**: Manages available cloud provider backends
- **Provider interface**: Defines the standard interface all backends must implement
- **Backend loading**: Dynamically loads and initializes provider backends
- **Configuration management**: Handles provider-specific configuration and credentials
- **Fallback logic**: Manages fallback between providers when one is unavailable

**`torch_remote/backends/modal/__init__.py`** - Modal Backend Integration
- **Modal backend implementation**: Implements the standard provider interface for Modal
- **Authentication handling**: Manages Modal API tokens and authentication
- **Resource configuration**: Handles Modal-specific GPU and container configurations
- **Error translation**: Converts Modal-specific errors to standard backend errors

**`torch_remote/backends/modal/_modal_remote.py`** - Modal-Specific Remote Execution
- **ModalRemoteExecutor class**: Modal-specific implementation of remote execution
- **Modal app integration**: Interfaces with the `torch_remote_backends.modal` package
- **A100 GPU optimization**: Modal-specific optimizations for A100 GPU utilization
- **Ephemeral execution**: Uses Modal's run contexts for one-off operations

#### C++ Extension

**`torch_remote/csrc/remote_extension.cpp`** - Python Extension Entry Point
- PyTorch C++ extension initialization using PyBind11
- Exposes `_init()` function to initialize PrivateUse1 device
- Provides `_get_default_generator()` for random number generation
- Links Python factory functions to C++ implementation
- Sets up the bridge between Python and C++ components

**`torch_remote/csrc/RemoteHooks.cpp`** - PrivateUse1 Backend Implementation
- Implements PyTorch's `PrivateUse1HooksInterface` for full device integration
- **Device management**: device count, current device, device guard implementation
- **Generator management**: Random number generators for remote device
- **Stream management**: Stream creation, synchronization, and querying
- **Memory management**: Host allocator integration
- **Event system**: Event creation, recording, and synchronization
- Integrates with Python-based device driver through method calls

**`torch_remote/csrc/RemoteMem.cpp`** - Memory Management
- **RemoteAllocator class**: Handles device memory allocation/deallocation
- Integrates with Python-based memory management system
- Routes allocation requests through Python driver
- Registers allocator with PyTorch's memory management system
- Handles memory cleanup and error reporting

**`torch_remote/csrc/Remote.h`** - C++ Header Definitions
- Common types and utilities for the C++ extension
- `remote_ptr_t` - Pointer type for remote device memory
- Python GIL management helpers
- Template functions for cleanup and error reporting
- Method lookup utilities for calling Python functions from C++

### torch_remote_execution Package (Private Package)

**`torch_remote_execution/__init__.py`** - Private Package Marker
- Simple package initialization with version information
- Documentation warning against direct use
- Marks package as internal to torch_remote

**`torch_remote_execution/modal_app.py`** - Modal Multi-GPU Execution App
- **Modal application definition**: Creates Modal app "torch-remote-extension" with multi-GPU support
- **Docker image setup**: 
  - Debian slim base with Python 3.11
  - Installs PyTorch with CUDA 12.1 support
- **Multi-GPU support**: Separate execution functions for each GPU type:
  - T4: `execute_aten_operation_t4()`
  - L4: `execute_aten_operation_l4()`
  - A10G: `execute_aten_operation_a10g()`
  - A100-40GB: `execute_aten_operation_a100_40gb()`
  - A100-80GB: `execute_aten_operation_a100_80gb()`
  - L40S: `execute_aten_operation_l40s()`
  - H100: `execute_aten_operation_h100()`
  - H200: `execute_aten_operation_h200()`
  - B200: `execute_aten_operation_b200()`
- **Device-specific GPU routing**: Each function configured with appropriate GPU type, timeout, and retry settings
- **Common execution implementation**: Shared `_execute_aten_operation_impl()` function that:
  - Receives serialized tensors, metadata, args, and kwargs
  - Deserializes tensors and moves them to CUDA device
  - Processes tensor placeholders in arguments
  - Executes the requested ATen operation on specified GPU
  - Serializes results and returns them
- **GPU utilization**: Automatically detects and uses CUDA when available
- **Error handling**: Comprehensive error reporting and traceback printing
- **Function registry**: `GPU_FUNCTIONS` dictionary maps GPU types to their execution functions

**`torch_remote_execution/setup.py`** - Private Package Installation
- Main setuptools configuration for the remote execution package
- Dependencies: modal>=0.60.0, torch>=2.0.0
- Marked as development status to discourage standalone installation
- Classifiers indicate it's for internal use

## Operation Flow

Here's how a typical operation flows through the system:

1. **User Code**: `result = torch.add(remote_tensor_a, remote_tensor_b)`

2. **PyTorch Dispatch**: PyTorch's dispatch system routes to remote device implementation

3. **_aten_impl.py**: 
   - `_remote_kernel_fallback` or `_kernel_fallback` receives the operation
   - `_should_use_remote_execution()` decides if this should run remotely
   - For compute-intensive ops: routes to remote execution
   - For simple ops: handles locally

4. **Remote Execution Path** (if enabled):
   - `RemoteExecutor.execute_remote_operation()` is called
   - Current provider backend is selected (e.g., Modal)
   - Tensors are serialized to bytes
   - Provider app is invoked (e.g., Modal app with `app.run()` context)
   - Appropriate GPU-specific function is called based on device configuration
   - `torch_remote_execution.modal_app.execute_aten_operation_*` runs on cloud GPU
   - Results are serialized and returned
   - Results are deserialized back to remote tensors

5. **Local Execution Path** (fallback):
   - Operation metadata is computed
   - Output tensors are allocated on remote device
   - Operation is executed via device daemon
   - Results are returned as remote tensors

## Key Design Decisions

### Private Package Isolation
The most important architectural decision is separating the provider backend code into `torch_remote_execution`. This prevents import conflicts when cloud provider jobs execute, since providers would otherwise try to import the entire `torch_remote` extension and create circular dependencies. The execution package is isolated with minimal dependencies.

### Threading vs Multiprocessing
The system uses threading instead of multiprocessing for device simulation to avoid complex cleanup issues that were causing hanging processes.

### Lazy Remote Execution
Remote execution is lazy-loaded and gracefully degrades when cloud providers are not available, allowing the extension to work in environments without specific providers. The multi-provider system allows fallback between different backends.

### Operation Filtering
Smart filtering ensures that only operations that benefit from cloud GPU acceleration are sent remotely, while keeping memory operations, views, and small tensor operations local for efficiency.

### Multi-GPU Support
The system supports multiple GPU types through device-specific Modal functions, allowing automatic routing to the most appropriate GPU based on workload requirements and availability.

### CPU Storage with Device Spoofing
Remote tensors are stored in CPU memory but report as "remote" device to PyTorch, enabling seamless integration with PyTorch's device system while maintaining compatibility.

## Configuration

The system uses Modal as the default cloud provider backend.

## Testing

The system includes comprehensive testing:
- `test_torch_remote.py` - Unit tests for all functionality
- Provider-specific tests for backend validation
- Pytest-based test suite with cleanup handling

This architecture provides a seamless PyTorch device experience while leveraging cloud provider GPU infrastructure for high-performance computing. The multi-provider system allows users to choose their preferred cloud backend while maintaining a consistent API.