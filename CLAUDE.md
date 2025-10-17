# Mycelya-Torch: PyTorch Remote Tensor Execution System

A production-ready PyTorch extension that enables transparent remote execution of tensor operations on cloud GPU infrastructure. The system uses a sequential tensor ID architecture with custom PyTorch integration for memory-efficient distributed computing.

## Architecture Overview

### Core Architectural Principles
- **Sequential Tensor ID Architecture**: Atomic counter generates unique storage IDs (1, 2, 3...) with metadata-based FNV-1a hash identification
- **Custom PyTorch Integration**: Complete custom TensorImpl, StorageImpl, and Allocator following pytorch-npu patterns with zero local memory overhead
- **Four-Layer Architecture**: C++ Backend, Python Coordination, Client Interface, Server Implementation with clean separation of concerns
- **Multi-GPU Cloud Support**: 8 GPU types supported (T4, L4, A10G, A100, L40S, H100, H200, B200) on Modal
- **Provider Abstraction**: Pluggable client system with Modal (production), Mock (development), extensible for AWS
- **RPC Batching**: Background thread processing reduces network overhead by ~10-100x with queue-based operation dispatch
- **Remote Function Execution**: @remote decorator enables transparent remote execution of custom functions with automatic serialization
- **CPU Scalar Support**: Automatic handling of 0-dimensional CPU tensors in mixed-device operations

### Production-Scale Features
- **Thread-Safe Operations**: Background processing with proper synchronization and error handling
- **Comprehensive Test Coverage**: Complete test suite with 2-tier test strategy (fast/comprehensive)
- **Enterprise Error Handling**: Clear error messages, graceful failure modes, and automatic resource cleanup
- **Memory Efficiency**: Zero local tensor data storage, raw bytes transfer, metadata caching with automatic invalidation
- **Performance Optimizations**: Meta tensor inference, view operation handling, dynamic output support

## Development Commands

To run tests:
```bash
# Regression tests (core functionality, ~10 seconds)
pytest tests/test_regression.py -v

# Remote decorator tests
pytest tests/test_remote_decorator.py -v

# Full comprehensive test suite
pytest tests/ -v
```

To run linting:
```bash
ruff check .
```

To run formatting:
```bash
ruff format .
```

To run type checking:
```bash
# No type checking tools currently configured
# Consider adding: mypy
```

## Local Development Setup

### Prerequisites
**IMPORTANT**: PyTorch 2.9+ must be pre-installed before building mycelya-torch.

```bash
# Install latest PyTorch first (CPU-only example)
pip install --upgrade torch

# For GPU support, visit: https://pytorch.org/get-started/locally/
```

### Building C++ Extensions
```bash
# Clone the repository
git clone https://github.com/alyxya/mycelya-torch.git
cd mycelya-torch

# Build C++ extensions for development (PyTorch must already be installed)
python setup.py build_ext --inplace
```

### Mock Client for Testing

For development and testing without cloud resources, use the mock client:

```python
import torch
import mycelya_torch

# Use mock client - runs locally using Modal's .local() execution
machine = mycelya_torch.RemoteMachine("mock")
device = machine.device("cpu")  # Mock client uses CPU

# All operations run locally but through the same API
x = torch.randn(100, 100, device=device)
y = x @ x  # Executed locally
```

**Mock Client Benefits:**
- **No Cloud Dependencies**: Runs entirely locally without Modal authentication
- **Same API**: Identical interface to production Modal client for testing
- **Fast Development**: Immediate feedback without network latency
- **CI/CD Integration**: Perfect for automated testing in build pipelines

## Build Configuration

### Python Packaging
- **`setup.py`**: Legacy setuptools build (no build isolation), C++ extension compilation with platform-specific compiler flags, includes PyTorch version validation
- **`ruff.toml`**: Ruff linter and formatter configuration
- **Dependencies**: modal>=1.1.0, numpy, cloudpickle>=3.1.1
- **Pre-requisite**: PyTorch>=2.9.0 (must be pre-installed before installing mycelya-torch)
- **License**: AGPL-3.0-or-later
- **Python Support**: 3.10+
- **Note**: No pyproject.toml to avoid pip build isolation, following pytorch/xla pattern

### Code Quality Configuration
- **Ruff**: Line length 88, comprehensive rule selection (E, W, F, I, B, C4, UP)
- **Pytest**: Configuration in `pytest.ini` with custom markers (@pytest.mark.fast)
- **Build artifacts**: Compiled extensions in `build/` directory

## Codebase Structure

### Project Scale
- **Modular architecture** with Python and C++ components across comprehensive test coverage
- **27 Python modules** (~6,850 lines) including core coordination, provider clients, and operation dispatch
- **10 C++ source files** (~1,000 lines) with custom PyTorch integration following pytorch-npu patterns
- **19 comprehensive test files** with 2-tier testing strategy and fast/comprehensive markers
- **Example application** demonstrating Stable Diffusion integration
- **Production-ready codebase** with enterprise-level code quality and documentation

### Core Python Modules (mycelya_torch/)
- `__init__.py` - Public API and PyTorch PrivateUse1 backend registration with tensor ID utilities
- `_orchestrator.py` - Central coordination with RPC batching, cache management, and background thread processing
- `_client_manager.py` - Client connection management and coordination for remote execution
- `_machine.py` - RemoteMachine abstraction with multi-provider support and context management
- `_device.py` - DeviceManager for mapping local device indices to remote GPU configurations
- `_storage.py` - Sequential storage ID system with atomic counter and thread-safe generation (1, 2, 3...)
- `_backend_hooks.py` - PyTorch backend hooks and C++ interface bridge for transparent integration
- `_remote.py` - Remote function execution decorator with automatic machine inference and serialization
- `_pickle.py` - Custom cloudpickle-based serialization system for tensor and function transfer
- `_utils.py` - Internal tensor utilities and metadata handling
- `_logging.py` - Hierarchical logging configuration with tensor hash IDs for debugging
- `_package_version.py` - Package version management and version checking utilities

### ATen Operation System (aten/)
Modular operation dispatch with clean separation of concerns:
- `__init__.py` - PyTorch library registrations for comprehensive ATen operation coverage
- `dispatch.py` - Main fallback kernel with meta tensor inference and CPU scalar tensor support
- `copy.py` - Cross-device copy and transfer operations with raw bytes optimization
- `scalar.py` - Scalar operations with local execution optimization

### Provider Client System (clients/)
- `__init__.py` - Client package initialization and exports
- `base_client.py` - Abstract client interface with RPC batching, caching, and standardized provider API
- `modal/__init__.py` - Modal client module exports
- `modal/client.py` - Modal cloud provider implementation with multi-GPU support and connection management
- `mock/__init__.py` - Mock client module exports
- `mock/client.py` - Local execution provider using Modal's .local() for development and testing

### Server Implementation System (servers/)
Pluggable server architecture for different cloud providers with clean client-server separation:
- `__init__.py` - Server system package initialization and provider discovery
- `modal/__init__.py` - Modal server module exports and API
- `modal/server.py` - Modal cloud GPU server implementation with dynamic app creation and lazy/realized storage
- `mock/__init__.py` - Mock server module exports and API
- `mock/server.py` - Mock server implementation for local testing and development

The servers directory mirrors the clients directory structure, providing a clear separation between:
- **Client Layer** (`clients/`): Interface, connection management, RPC batching, local coordination
- **Server Layer** (`servers/`): Remote execution, GPU management, tensor operations, storage handling

### C++ Backend Integration (csrc/)
Complete custom PyTorch integration following pytorch-npu patterns:
- `Mycelya.h` - Core header definitions, constants, and API declarations
- `MycelyaTensorImpl.cpp/.h` - Custom tensor implementation with FNV-1a metadata hash computation
- `MycelyaStorageImpl.cpp/.h` - Custom storage implementation with sequential ID tracking and zero local memory
- `MycelyaAllocator.cpp/.h` - Enhanced allocator with storage ID management and efficient allocation
- `MycelyaHooks.cpp` - PyTorch PrivateUse1 backend hooks with custom device management
- `MycelyaMem.cpp` - Memory management utilities and cross-platform compatibility
- `mycelya_extension.cpp` - Python bindings, C++ extensions, and API exposure

### Development Resources
- `examples/` - Sample applications:
  - `tiny_sd.py` - Stable Diffusion integration example
- `tests/` - Comprehensive test coverage with fast/comprehensive markers (19 test files):
  - `test_regression.py` - Core functionality regression tests (~10 seconds)
  - `test_remote_decorator.py` - Remote function execution tests
  - `test_autograd_basic.py` - Basic autograd functionality
  - `test_autograd_complex.py` - Complex autograd scenarios
  - `test_basic_operations.py` - Basic tensor operations
  - `test_comparison_logical_operations.py` - Comparison and logical operations
  - `test_device_management.py` - Device creation and management
  - `test_error_handling.py` - Error handling and validation
  - `test_indexing_selection_operations.py` - Indexing and selection
  - `test_loss_functions.py` - Loss function computations
  - `test_mathematical_operations.py` - Mathematical operations
  - `test_mycelya_torch.py` - Core mycelya-torch functionality
  - `test_reduction_operations.py` - Reduction operations
  - `test_tensor_manipulation_operations.py` - Tensor manipulation
  - `test_tensor_transfers.py` - Tensor transfer operations
  - `test_transformer.py` - Transformer model tests
  - `test_utilities.py` - Utility functions
  - `test_view_operations.py` - View and reshape operations
  - `conftest.py` - Pytest configuration and fixtures
- Legacy build system with `setup.py` for C++ extensions and `ruff.toml` for code quality

## Current Architecture

### Key Design Principles
- **Sequential Tensor ID Architecture**: Atomic counter generates unique storage IDs (1, 2, 3...) with FNV-1a metadata hash computation for debugging
- **Custom PyTorch Integration**: Complete TensorImpl/StorageImpl/Allocator following pytorch-npu architecture patterns with zero local memory overhead
- **Clean Input/Output Separation**: Raw bytes transfer with numpy serialization, eliminating torch.save/load overhead (~2-5x faster)
- **Zero Local Memory**: No tensor data stored locally for remote tensors, only metadata maintained in custom implementations
- **RPC Batching**: Background thread processing reduces network overhead by ~10-100x with queue-based operation dispatch
- **Multi-Provider Support**: Extensible client system with Modal (production), Mock (development), ready for AWS
- **CPU Scalar Tensor Support**: Automatic handling of 0-dimensional CPU tensors in operations with mycelya tensors

### Memory Management Excellence
- **Zero local memory overhead**: Custom TensorImpl/StorageImpl store no tensor data locally, only metadata maintained
- **Sequential storage ID system**: Atomic counter generates unique IDs (1, 2, 3...) for efficient memory management across devices
- **Dual storage architecture**: Lazy allocation for meta operations, realized storage on remote GPUs for actual computation
- **Raw bytes transfer**: Direct numpy serialization bypasses torch.save/load overhead, achieving ~2-5x faster data transfer
- **Metadata-based caching**: Shape/stride/offset/dtype keys enable efficient caching without storing tensor data
- **FNV-1a hash computation**: On-demand metadata hash generation for debugging without memory allocations
- **Automatic cache invalidation**: Immediate invalidation at queue time ensures correctness with background batching

### Advanced Operation Dispatch Flow
1. **Device Validation**: Check for mixed device operations, allowing 0-dimensional CPU scalars with mycelya tensors
2. **Meta Tensor Inference**: Shape computation using PyTorch's meta device eliminates data transfer for shape operations
3. **View Operation Optimization**: Local view creation with remote propagation maximizes memory efficiency
4. **Dynamic Output Support**: Special handling for operations with data-dependent output shapes (e.g., nonzero, unique)
5. **RPC Batching Pipeline**: Operations queued in background thread, reducing network calls by ~10-100x
6. **Remote Function Execution**: @remote decorator with automatic machine inference and cloudpickle-based serialization
7. **Remote Execution**: All compute operations dispatched to cloud GPUs with proper error handling
8. **Thread-Safe Processing**: Background thread coordination with proper synchronization and error propagation
9. **Efficient Data Transfer**: Raw untyped storage bytes only when crossing device boundaries, no unnecessary serialization

## Usage Patterns

### Basic Usage
```python
import torch
import mycelya_torch

# Create remote machine with cloud GPU
machine = mycelya_torch.RemoteMachine("modal", "A100")

# Operations automatically execute on remote GPU with RPC batching
x = torch.randn(1000, 1000, device=machine.device("cuda"))  # Storage ID: 1
y = torch.randn(1000, 1000, device=machine.device("cuda"))  # Storage ID: 2
result = x @ y  # Matrix multiplication on remote A100, Storage ID: 3

# Each tensor has FNV-1a metadata hash for debugging
print(f"Result computed on {result.device}: {result.shape}")
```

### CPU Scalar Tensor Support
```python
import torch
import mycelya_torch

machine = mycelya_torch.RemoteMachine("modal", "A100")
device = machine.device("cuda")

# Create mycelya tensor
x = torch.randn(1000, 1000, device=device)

# Mix with CPU scalar tensor (0-dimensional) - automatically handled
scalar_cpu = torch.tensor(2.0)  # CPU scalar
result = x * scalar_cpu  # Works! CPU scalar auto-transferred

# Non-scalar CPU tensors still raise error
cpu_vector = torch.randn(1000)  # Non-scalar
# result = x + cpu_vector  # RuntimeError: Cannot mix cpu tensors with mycelya tensors
```

### Remote Function Execution
```python
import torch
import mycelya_torch

# Create remote machine
machine = mycelya_torch.RemoteMachine("modal", "A100")
device = machine.device("cuda")

# Define custom functions that execute remotely
@mycelya_torch.remote
def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b

@mycelya_torch.remote()
def complex_computation(x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    # Multiple operations executed remotely with automatic batching
    y = x * scale
    z = torch.relu(y)
    w = torch.softmax(z, dim=-1)
    return w.sum(dim=0)

# Create tensors on remote GPU
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Functions automatically execute on remote GPU with machine inference
result1 = matrix_multiply(x, y)  # Executes remotely on A100
result2 = complex_computation(x, scale=3.0)  # Executes remotely on A100

print(f"Results computed on {result1.device}")
```

### Production Neural Network Training
```python
import torch.nn as nn
import mycelya_torch

# Create remote machine with high-memory GPU
machine = mycelya_torch.RemoteMachine("modal", "A100")
device = machine.device("cuda")

# Model automatically uses sequential tensor IDs for all parameters
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Full training loop with remote gradient computation
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(dataloader):
        # Efficient data transfer with raw bytes optimization
        data, target = data.to(device), target.to(device)

        # All operations batched and executed remotely
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()  # Gradients computed on remote A100
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```


## Implementation Details

### Sequential Storage ID Architecture
- **Atomic Storage ID Generation**: Thread-safe counter producing unique incremental IDs (1, 2, 3, 4...)
- **FNV-1a Metadata Hash Computation**: Deterministic hash from shape/stride/dtype/offset/storage_id for debugging
- **Custom TensorImpl Integration**: MycelyaTensorImpl computes metadata hashes on-demand in C++
- **Zero Memory Overhead**: Hash generation without additional memory allocations or storage
- **Cross-Device Validation**: Storage IDs prevent operations between different remote machines
- **Internal API Access**: Metadata hash accessible via utility functions for debugging complex tensor flows
- **Process-Scoped Uniqueness**: Storage IDs unique within single Python process for memory efficiency

### CPU Scalar Tensor Support
- **Automatic Detection**: 0-dimensional CPU tensors detected in operation dispatch
- **Validation in Meta Execution**: Mixed device tensors validated, allowing CPU scalars
- **Conversion in Orchestrator**: CPU scalar tensors converted to Python scalars via .item()
- **PyTorch Compatibility**: Matches PyTorch's standard behavior where CPU scalars auto-transfer to GPU
- **Error Handling**: Non-scalar CPU tensors properly rejected with clear error messages

### Enterprise Error Handling
- **Cross-Device Operation Prevention**: Clear RuntimeError messages when mixing tensors from different machines
- **Type Safety Validation**: Descriptive errors for non-mycelya tensors accessing tensor ID APIs
- **Storage State Management**: Proper error handling for lazy vs realized storage access patterns
- **Connection Lifecycle Management**: Automatic reconnection with graceful degradation and retry logic
- **Background Thread Error Propagation**: Proper error handling in RPC batching with future-based error reporting
- **Provider-Specific Errors**: Modal connection failures, authentication errors, and resource exhaustion handling
- **Memory Management Errors**: Clear errors for storage allocation failures and cleanup issues

### Multi-Provider Architecture
- **Standardized Client Interface**: Abstract base client with consistent API across providers
- **Modal Production Client**: Complete cloud provider with multi-GPU support, dynamic app creation, connection pooling
- **Mock Development Client**: Local execution using Modal's .local() for testing without cloud dependencies
- **Extensible Provider System**: Clean interfaces ready for AWS integration
- **Provider-Agnostic Features**: RPC batching, caching, error handling work across all providers
- **Connection Management**: Proper initialization, cleanup, and resource management per provider
- **GPU Type Abstraction**: Unified interface for 8 different GPU types on Modal

## Documentation Maintenance

**IMPORTANT**: This file should be updated whenever making significant changes to the codebase.

### Update This File When:
- Adding new core modules or changing module responsibilities
- Modifying the sequential tensor ID architecture or metadata hash system
- Adding/removing provider clients (AWS, etc.) or GPU types
- Changing development commands (test, lint, typecheck) or build configuration
- Making breaking changes to the public API or internal architecture
- Major performance optimizations or C++ implementation changes
- Updates to RPC batching system or background thread processing
- Reorganizing ATen operation handling or modular dispatch system
- Updates to build system, development workflows, or testing strategies

**Recent Changes**:
- Removed pyproject.toml to eliminate pip build isolation (follows pytorch/xla pattern)
- All metadata now in setup.py, ruff configuration in ruff.toml

## Development Guidelines

### Production Code Quality Standards

#### License Compliance (AGPL-3.0-or-later)
- **All source files must maintain AGPL license headers**
- New files require: `Copyright (C) 2025 alyxya, SPDX-License-Identifier: AGPL-3.0-or-later`
- Contributions must preserve open source licensing for derivative works

#### Sequential Tensor ID Architecture Rules
- **Never store tensor data locally** for remote tensors - only metadata in custom implementations
- **Sequential storage IDs** (1, 2, 3...) generated by atomic counter for memory efficiency
- **FNV-1a metadata hash computation** from tensor properties in MycelyaTensorImpl for debugging
- **Zero memory overhead** design - hash computed on-demand without additional allocations
- **Custom PyTorch integration** following pytorch-npu patterns with complete TensorImpl/StorageImpl/Allocator
- **Process-scoped uniqueness** for storage IDs enabling efficient cross-device validation

#### Multi-Provider Architecture Implementation
- **Client-Server Separation**: Clean separation between client interfaces (`clients/`) and server implementations (`servers/`)
- **Standardized Client Interface**: Follow base_client.py pattern with RPC batching support for all providers
- **Server Implementation Pattern**: Follow Modal server structure in `mycelya_torch/servers/modal/` for new providers
- **Multi-GPU Configuration**: Support lazy/realized storage architecture across all server implementations
- **Connection Management**: Handle lifecycle, authentication, and background thread processing in client layer
- **Hierarchical Logging**: Integrate with tensor hash IDs for debugging across client-server boundaries
- **Development Support**: Provide Mock client equivalent using server .local() execution for testing

#### Enterprise-Level Code Quality
- **Ruff linting/formatting**: Line length 88, comprehensive rule selection (E,W,F,I,B,C4,UP)
- **Google C++ Style**: All C++ files with consistent formatting and comprehensive documentation
- **2-Tier Testing Strategy**: Regression (~10s), Comprehensive (~10-30min)
- **Test Markers**: Use `@pytest.mark.fast` for categorization
- **Comprehensive Error Handling**: Clear RuntimeError messages, graceful failure modes, proper error propagation
- **Thread-Safe Operations**: Background processing with proper synchronization and future-based error handling
- **Modular Organization**: Clean separation between ATen operation handlers, provider clients, and core coordination

#### Performance and Memory Optimization
- **RPC Batching**: Background thread reduces network calls by ~10-100x with queue-based dispatch
- **Raw Bytes Transfer**: Direct numpy serialization eliminating torch.save/load overhead (~2-5x faster)
- **Meta Tensor Integration**: Shape computation without data transfer using PyTorch's meta device
- **View Operation Optimization**: Local view creation with remote propagation for memory efficiency
- **Automatic Cache Invalidation**: Immediate invalidation at queue time ensuring correctness
- **CPU Scalar Optimization**: 0-dimensional CPU tensors converted to Python scalars for efficient remote execution