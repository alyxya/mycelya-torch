# torch-remote Architecture Documentation

This document provides a detailed explanation of the torch-remote codebase architecture, including the purpose and functionality of each source file.

## Overview

torch-remote implements a custom PyTorch device that automatically routes computations to remote cloud providers. The system uses a three-layer architecture with a pluggable multi-provider backend system. Modal serves as the first supported provider, with additional providers planned for future releases. The architecture includes private package isolation to prevent provider-specific import conflicts.

## Architecture Layers

### 1. PyTorch Integration Layer (C++ Extension)
Registers "remote" as a native PyTorch PrivateUse1 device with full integration into PyTorch's dispatch system.

### 2. Local Coordination Layer (Python)
Manages remote tensors using pure tensor ID-based coordination with local metadata tracking and device registry management.

### 3. Remote Execution Layer (Multi-Provider)
Automatically dispatches compute-intensive operations to cloud providers via a pluggable backend system. Currently supports Modal's A100 GPU infrastructure, with additional providers planned.

## Package Structure

```
torch_remote/                   # Main PyTorch extension package
├── __init__.py                 # Package initialization & device registration
├── _aten_impl.py              # Operation dispatch & remote execution logic  
├── _remote_orchestrator.py    # Remote execution orchestration
├── _meta_parser.py            # Tensor metadata & data structures
├── _device_daemon.py          # Device management & process communication
├── device.py                  # Backend device abstraction & registry
├── utils.py                   # Tensor method extensions
├── backends/                  # Multi-provider backend system
│   ├── __init__.py            # Backend registry & management
│   └── modal/                 # Modal provider implementation
│       └── __init__.py        # Modal backend integration
└── csrc/                      # C++ extension
    ├── remote_extension.cpp   # Python extension entry point
    ├── RemoteHooks.cpp        # PrivateUse1 backend implementation
    ├── RemoteMem.cpp          # Memory management
    └── Remote.h               # C++ header definitions

torch_remote_modal/            # Private package for Modal execution
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

**`torch_remote/_remote_orchestrator.py`** - Remote Execution Orchestration
- **RemoteOrchestrator class**: Manages remote execution across cloud providers using stateful ModalClient instances
- **Tensor serialization/deserialization**: Converts remote tensors to/from bytes for network transport
- **Provider backend integration**: Interfaces with the `torch_remote_modal` package
- **Error handling and fallbacks**: Graceful degradation when remote providers are unavailable
- **Device validation**: Enforces single-device operations and prevents mixed-device tensor operations
- **Device conversion helpers**: Converts between remote tensors and CPU tensors for transport
- **Stateful execution**: Manages device-specific GPU machines with caching for improved performance

**`torch_remote/_meta_parser.py`** - Tensor Metadata & Data Structures
- **RemoteTensorMeta class**: Captures tensor metadata (shape, dtype, strides, storage info) for serialization
- **C++ Integration**: Remote tensors are now regular torch.Tensor objects created via C++ TORCH_LIBRARY_IMPL
- **Serialization helpers**: Convert tensors to/from metadata for inter-process communication
- **Device handling**: Direct PyTorch device integration without wrapper classes
- **Validation functions**: Ensures only valid data types pass through device boundaries

**`torch_remote/_device_daemon.py`** - Device Management & Tensor ID Registry
- **Driver class**: Main coordinator for device operations and tensor ID management
- **RemoteTensorRegistry**: Manages local tensor ID to metadata mapping
- **Tensor ID coordination**: 
  - Maps tensor IDs to RemoteTensorMeta objects with shape/dtype information
  - Tracks tensor ID to device relationships for validation
  - Maintains weak references to local tensor objects
- **Device registry management**: Coordinates with DeviceRegistry for multi-device operations
- **Stream and event management**: PyTorch CUDA-like stream semantics for remote device
- **Cleanup handling**: Signal handlers and atexit hooks for proper resource cleanup
- **Pure ID-based architecture**: No local tensor data storage, only metadata coordination

#### Utility Files

**`torch_remote/utils.py`** - Tensor Method Extensions
- Patches `.to()` method to support `RemoteMachine` objects
- Enables `tensor.to(backend_device)` to move tensors to remote device
- Simple wrapper around the C++ remote conversion function

**`torch_remote/device.py`** - Backend Device Abstraction & Registry
- **RemoteMachine class**: Represents a remote GPU device with specific provider and GPU type
  - Unique device ID generation with provider-gpu-uuid format
  - GPU type validation for provider compatibility
  - Device equality and hashing based on unique device ID
  - Provider-specific configuration support
- **DeviceRegistry class**: Manages active RemoteMachine instances
  - Device registration with automatic index assignment
  - Device lookup by ID or index
  - Device compatibility validation for operations
  - Enforces single-device constraint for tensor operations
- **GPUType enum**: Supported GPU types (T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200)
- **BackendProvider enum**: Supported cloud providers (Modal, with future providers planned)
- **Factory functions**: `create_modal_machine()` for easy Modal device creation
- **Global registry**: Shared device registry for system-wide device management

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

### torch_remote_modal Package (Private Package)

**`torch_remote_modal/__init__.py`** - Private Package Marker
- Simple package initialization with version information
- Documentation warning against direct use
- Marks package as internal to torch_remote

**`torch_remote_modal/modal_app.py`** - Modal Multi-GPU Execution App
- **ModalClient class**: Stateful wrapper representing a remote GPU machine running on Modal
  - Encapsulates Modal app and executor with connection management
  - Context manager support for automatic resource cleanup
  - Device-specific initialization and state management
  - Machine lifecycle operations (start, stop, is_running)
- **Modal application definition**: Creates device-specific Modal apps with unique identifiers
- **Docker image setup**: 
  - Debian slim base with Python 3.11
  - Installs PyTorch with CUDA support
- **Multi-GPU support**: Dynamic creation of device-specific execution functions for each GPU type:
  - T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200
- **Device-specific GPU routing**: Each machine configured with appropriate GPU type, timeout, and retry settings
- **Stateful execution model**: 
  - `create_modal_app_for_gpu()` creates device-specific ModalClient instances
  - Machine caching prevents redundant app creation for same device
  - Context management ensures proper resource cleanup
- **Common execution implementation**: Shared execution logic that:
  - Receives serialized tensors, metadata, args, and kwargs
  - Deserializes tensors and moves them to CUDA device
  - Processes tensor placeholders in arguments
  - Executes the requested ATen operation on specified GPU
  - Serializes results and returns them
- **GPU utilization**: Automatically detects and uses CUDA when available
- **Error handling**: Comprehensive error reporting and traceback printing
- **Dynamic app creation**: Creates unique Modal apps per device to support multiple concurrent GPU machines

**`torch_remote_modal/setup.py`** - Private Package Installation
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
   - **Device validation**: Ensures all tensors belong to same device before operation
   - For compute-intensive ops: routes to remote execution
   - For simple ops: handles locally

4. **Remote Execution Path** (if enabled):
   - `RemoteOrchestrator.execute_remote_aten_operation_efficient()` is called
   - **Device detection and validation**: `_detect_device_from_tensors()` ensures single-device operation
   - Device-specific ModalClient is retrieved or created
   - Tensors are serialized to bytes
   - ModalClient context is started (Modal app)
   - Appropriate GPU-specific function is called based on device configuration
   - `PytorchServer.execute_aten_operation()` runs on cloud GPU
   - Results are serialized and returned
   - Results are deserialized back to remote tensors with original device ID preserved

5. **Local Execution Path** (fallback):
   - Operation metadata is computed
   - Output tensors are allocated on remote device
   - Operation is executed via device daemon
   - Results are returned as remote tensors

## Key Design Decisions

### Private Package Isolation
The most important architectural decision is separating the Modal backend code into `torch_remote_modal`. This prevents import conflicts when Modal jobs execute, since Modal would otherwise try to import the entire `torch_remote` extension and create circular dependencies. The Modal execution package is isolated with minimal dependencies.

### Pure Tensor ID Coordination
The system uses a pure tensor ID-based architecture without local device simulation. This eliminates the need for complex threading/multiprocessing decisions and provides cleaner coordination between local metadata tracking and remote execution.

### Lazy Remote Execution
Remote execution is lazy-loaded and gracefully degrades when cloud providers are not available, allowing the extension to work in environments without specific providers. The multi-provider system allows fallback between different backends.

### Operation Filtering
Smart filtering ensures that only operations that benefit from cloud GPU acceleration are sent remotely, while keeping memory operations, views, and small tensor operations local for efficiency.

### Stateful Remote Execution
The system uses stateful ModalClient instances for improved performance and resource management:
- **Device-specific machines**: Each RemoteMachine gets its own ModalClient instance
- **Connection caching**: Modal app contexts are reused across operations on the same device
- **Context management**: Automatic startup/shutdown of remote GPU resources
- **Machine lifecycle**: Start, stop, and running state management for remote resources

### Input/Output Tensor Separation
The system implements intelligent separation between input and output tensors for optimal data transfer efficiency:
- **Input tensors**: Data is read from local metadata and serialized for remote execution
- **Output tensors**: Only metadata is used locally, data is written directly on remote GPU
- **In-place operations**: Supported efficiently without unnecessary data transfers
- **Pre-allocated outputs**: `out=` parameter tensors are handled as output tensors to avoid redundant transfers

### Multi-GPU Support
The system supports multiple GPU types through device-specific ModalClient instances, allowing automatic routing to the most appropriate GPU based on workload requirements and availability.

### Tensor ID-Based Memory Architecture
Remote tensors use a pure tensor ID system with no local data storage. Each tensor stores only a unique 64-bit ID and metadata (shape, dtype) locally, while actual tensor data resides exclusively on remote GPUs. This provides zero local memory overhead and seamless PyTorch integration.

### Device Validation and Isolation
The system enforces strict device isolation to prevent operations between tensors on different remote devices:
- **Single-device constraint**: Operations can only be performed between tensors on the same RemoteMachine instance
- **Device ID tracking**: Each tensor maintains a `_device_id` attribute linking it to its specific remote device
- **Cross-device detection**: `_detect_device_from_tensors()` validates all tensors in an operation belong to the same device
- **Error prevention**: Mixed-device operations raise clear error messages before execution
- **Provider isolation**: Different cloud provider instances are treated as distinct devices

## Configuration

The system uses Modal as the default cloud provider backend.

## Testing

The system includes comprehensive testing:
- `test_torch_remote.py` - Unit tests for all functionality
- Provider-specific tests for backend validation
- Pytest-based test suite with cleanup handling

This architecture provides a seamless PyTorch device experience while leveraging cloud provider GPU infrastructure for high-performance computing. The multi-provider system with stateful execution, device validation, and backend abstraction allows users to choose their preferred cloud backend while maintaining a consistent API and ensuring safe, efficient operations across different remote GPU devices.