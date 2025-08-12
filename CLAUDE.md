# Mycelya: PyTorch Remote Tensor Execution System

A PyTorch extension that enables transparent remote execution of tensor operations on cloud GPU infrastructure. The system uses a sequential tensor ID architecture with custom PyTorch integration for memory-efficient distributed computing.

## Architecture Overview

- **Metadata Hash System**: Remote tensors have unique metadata-based hash IDs with `tensor.metadata_hash` property
- **Custom PyTorch Integration**: Complete custom TensorImpl, StorageImpl, and Allocator following pytorch-npu patterns
- **Three-Layer Architecture**: C++ Backend, Python Coordination, Remote Execution
- **Multi-GPU Support**: 9 GPU types supported (T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200)
- **Provider Abstraction**: Pluggable backend system (Modal, Mock providers, extensible for others)
- **RPC Batching**: Background thread processing for reduced network overhead
- **HuggingFace Integration**: Direct remote model loading with parameter linking

## Development Commands

To run tests:
```bash
# Minimal regression tests (every commit, <30 seconds)
pytest tests/test_regression.py::TestCriticalRegression -v

# Fast functional tests (PR reviews, ~2-5 minutes)  
pytest tests/test_regression.py -v

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

## Build Configuration

### Modern Python Packaging
- **`pyproject.toml`**: Modern build system using setuptools with PyTorch C++ extensions
- **`setup.py`**: C++ extension compilation with platform-specific compiler flags
- **Dependencies**: torch>=2.1.0, modal>=1.0.0, numpy
- **License**: AGPL-3.0-or-later
- **Python Support**: 3.8+

### Code Quality Configuration
- **Ruff**: Line length 88, comprehensive rule selection (E, W, F, I, B, C4, UP)
- **Pytest**: Configuration in `pytest.ini` with custom markers
- **Build artifacts**: Compiled extensions in `build/` directory

## Key Components

### Core Modules
- `mycelya_torch/__init__.py` - Public API and PyTorch PrivateUse1 backend registration with metadata hash monkey patching
- `mycelya_torch/_aten_impl.py` - ATen operation dispatch system with meta tensor inference and view handling
- `mycelya_torch/_remote_orchestrator.py` - Remote execution orchestration with RPC batching integration
- `mycelya_torch/_device_daemon.py` - Device registry and storage operations with centralized logging
- `mycelya_torch/_device.py` - Device registry management with thread-safe device registration
- `mycelya_torch/_machine.py` - RemoteMachine abstraction supporting Modal and Mock providers

### Utility and Support Modules
- `mycelya_torch/_tensor_utils.py` - Clean metadata classes and serialization utilities
- `mycelya_torch/_storage.py` - Storage ID lifecycle management and cross-device validation
- `mycelya_torch/_batching.py` - RPC batching system with background thread processing
- `mycelya_torch/_huggingface_utils.py` - Model loading and tensor linking utilities for HuggingFace models
- `mycelya_torch/_logging.py` - Centralized logging configuration with hierarchical loggers

### Simple Operation Dispatch
- Direct conditional logic in `_aten_impl.py` - Simple if/elif dispatch without complex patterns

### Provider Interface
- `mycelya_torch/backends/client_interface.py` - Standardized provider interface with caching and RPC batching support
- `mycelya_torch/backends/modal/client.py` - Modal provider implementation with RPC batching integration
- `mycelya_torch/backends/mock/client.py` - Local execution provider for development and testing

### Remote Execution Provider
- `_mycelya_torch_modal/modal_app.py` - Modal cloud GPU integration with multi-GPU support

### C++ Backend Integration
- `mycelya_torch/csrc/Mycelya.h` - Core header definitions and declarations
- `mycelya_torch/csrc/MycelyaTensorImpl.cpp/.h` - Custom tensor implementation with metadata hash computation
- `mycelya_torch/csrc/MycelyaStorageImpl.cpp/.h` - Custom storage implementation with storage ID tracking
- `mycelya_torch/csrc/MycelyaAllocator.cpp/.h` - Enhanced allocator with storage ID management
- `mycelya_torch/csrc/MycelyaHooks.cpp` - PyTorch PrivateUse1 backend hooks with custom implementations
- `mycelya_torch/csrc/MycelyaMem.cpp` - Memory management utilities
- `mycelya_torch/csrc/mycelya_extension.cpp` - Python bindings and extensions

## Current Architecture (2025-08-10)

### Key Design Principles
- **Metadata Hash System**: Unique metadata-based hash IDs with `tensor.metadata_hash` property for debugging
- **Custom PyTorch Integration**: Complete TensorImpl/StorageImpl following pytorch-npu architecture patterns
- **Clean Input/Output Separation**: Efficient data transfer with clear boundaries
- **Zero Local Memory**: No tensor data stored locally for remote tensors
- **RPC Batching**: Background thread processing for performance optimization
- **Multi-Provider Support**: Extensible backend system with Modal and Mock implementations

### Memory Efficiency
- **Zero local memory overhead** for remote tensors with custom TensorImpl/StorageImpl
- **Dual storage architecture**: Lazy allocation vs realized storage on remote GPUs
- **Raw bytes storage**: Direct numpy serialization eliminating torch.save/load overhead
- **Metadata-based caching**: Shape/stride/offset keys instead of tensor ID parameters
- **Automatic cache invalidation**: Proper cache semantics with batching operations
- **Thread-safe tensor ID generation**: Atomic counter for unique sequential IDs

### Operation Dispatch Flow
1. **Meta Tensor Inference**: Shape inference using PyTorch's meta device
2. **View Operation Handling**: Local view creation with remote propagation
3. **Dynamic Output Support**: Special handling for operations with data-dependent output shapes
4. **RPC Batching**: Operations queued and batched in background thread
5. **Remote Execution**: All compute operations dispatched to remote GPUs
6. **Cache Management**: Immediate invalidation at queue time for correctness
7. **Data Transfer**: Raw untyped storage bytes only when crossing device boundaries

## Usage Patterns

### Basic Usage
```python
import torch
import mycelya_torch

# Create remote machine
machine = mycelya_torch.RemoteMachine("modal", "T4")

# Operations automatically execute on remote GPU
x = torch.randn(1000, 1000, device=machine.device())
y = torch.randn(1000, 1000, device=machine.device())
result = x @ y  # Matrix multiplication on remote T4
```

### Advanced Usage
```python
# Neural network training with tensor IDs
model = nn.Linear(784, 10).to(machine.device())
optimizer = torch.optim.Adam(model.parameters())

# Full training loop on remote GPU
for data, target in dataloader:
    data, target = data.to(machine.device()), target.to(machine.device())
    print(f"Data tensor ID: {data.id}")  # Unique sequential ID
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # Gradients computed remotely
    optimizer.step()
```

### HuggingFace Integration
```python
# Direct remote model loading
import mycelya_torch

machine = mycelya_torch.RemoteMachine("modal", "T4")
model = mycelya_torch.load_huggingface_model(
    "microsoft/DialoGPT-medium", 
    machine, 
    torch_dtype=torch.float16
)

# Model parameters are already on remote GPU
for name, param in model.named_parameters():
    print(f"{name}: tensor ID {param.id}, device {param.device}")
```

## Implementation Details

### Tensor ID System
- **Sequential incremental IDs** (1, 2, 3...) generated for each mycelya tensor
- **Python API integration**: `tensor.metadata_hash` property and `tensor.get_metadata_hash()` method
- **Hash-based identification**: Uses FNV-1a hash of shape/stride/dtype/offset/storage_id
- **Custom TensorImpl integration**: Metadata hash computation in MycelyaTensorImpl
- **Zero memory overhead**: Hash computed on-demand without additional allocations
- **Debugging support**: Unique identification for complex tensor flows

### Error Handling
- **Cross-device operation prevention**: Clear errors when mixing different machines
- **Type safety**: Clear error messages for non-mycelya tensors accessing tensor IDs
- **Storage validation**: Proper error handling for lazy vs realized storage access
- **Connection management**: Automatic reconnection with RPC batching
- **Comprehensive error handling**: Descriptive RuntimeError messages for all failure modes
- **Batch error propagation**: Proper error handling in background thread processing

### Provider Interface
- **Standardized client interface** for multiple providers with RPC batching support
- **Modal implementation** as production cloud provider
- **Mock implementation** for local development and testing
- **Extensible architecture** for RunPod, Lambda Labs, etc.
- **Dual storage support**: Lazy allocation and realized storage across providers
- **Connection lifecycle management**: Proper initialization and cleanup

## Documentation Maintenance

**IMPORTANT**: This file should be updated whenever making significant changes to the codebase.

### Update This File When:
- Adding new core modules or changing module responsibilities
- Modifying the architecture (tensor ID system, execution flow, etc.)
- Adding/removing provider backends or GPU support
- Changing development commands (test, lint, typecheck)
- Making breaking changes to the public API
- Adding new important implementation details

### Development Guidelines

#### License Compliance
- **All source files must maintain AGPL license headers**
- New files require: Copyright (C) 2025 alyxya, SPDX-License-Identifier: AGPL-3.0-or-later

#### Metadata Hash Architecture Rules
- **Never store tensor data locally** for remote tensors - only metadata
- **Metadata-based hash IDs** computed from tensor properties in custom TensorImpl
- **Clean separation** between metadata hashes (debugging) and storage IDs (memory management)
- **Custom PyTorch integration** following pytorch-npu implementation patterns
- **Deterministic hash generation** based on tensor metadata

#### Provider Implementation Patterns  
- Follow Modal implementation pattern in `_mycelya_torch_modal/`
- Implement standardized client interface with RPC batching support
- Support multi-GPU configuration with lazy/realized storage
- Handle connection lifecycle and background thread processing properly
- Integrate with centralized logging system

#### Code Quality Standards
- **Use ruff for linting and formatting**
- **Run minimal regression tests on every commit**
- **Follow existing patterns** for new functionality
- **Comprehensive error handling** with clear error messages
- **Thorough testing** for all new features

#### Testing Strategy
- **Critical regression tests**: 12 essential tests covering core functionality (~30 seconds)
- **Fast functional tests**: Extended coverage for PR reviews (~2-5 minutes)
- **Full test suite**: Comprehensive validation for releases (~10-30 minutes)
- **Test markers**: Use `@pytest.mark.critical` and `@pytest.mark.fast` for categorization

### Recent Development Notes

#### Custom TensorImpl Integration (2025-08-10)
- **Complete custom tensor stack**: MycelyaTensorImpl, MycelyaStorageImpl, MycelyaAllocator
- **Metadata hash IDs**: Unique hash-based IDs with `tensor.metadata_hash` property
- **pytorch-npu compliance**: Following established integration patterns for production readiness
- **Python API enhancement**: Monkey patching torch.Tensor with mycelya-specific methods
- **FNV-1a hash algorithm**: Fast, deterministic hash of shape/stride/dtype/offset/storage_id
- **Zero memory overhead**: Custom implementations maintain existing efficiency
- **Diagnostic capabilities**: Testing functions to verify custom implementation usage

#### RPC Batching System (2025-08-08)
- **Background thread processing**: Queue-based batching reduces network overhead
- **Thread-safe operations**: `RPCBatchQueue` with automatic futures handling
- **Comprehensive error handling**: `BatchProcessor` with proper error propagation
- **Cache invalidation timing**: Invalidation at queue time for correct semantics
- **Provider integration**: Seamless integration with Modal and Mock clients

#### HuggingFace Integration (2025-08-08)
- **Direct remote model loading**: Load models directly on remote GPUs without data transfer
- **Local model skeleton**: Create parameter stubs with remote tensor linking
- **Storage ID linking**: Automatic parameter connection to remote storage
- **Tied weights support**: Proper handling of shared parameters (e.g., embedding layers)
- **Dual loading approaches**: Remote-first vs local-then-transfer strategies

#### Dual Storage Architecture (2025-08-08)
- **Lazy storage**: Integer byte counts for deferred allocation
- **Realized storage**: Actual tensor data on remote GPUs
- **Modal server support**: Both lazy and realized storage types
- **Fallback mechanisms**: Proper error handling for storage retrieval
- **Model parameter compatibility**: Improved access patterns for HuggingFace models

#### Raw Bytes Storage Refactoring (2025-07-27)
- **Eliminated torch.save/load overhead**: Replaced with direct raw untyped storage byte operations
- **Dual tensor metadata support**: Storage operations now support source + target metadata for partial updates
- **Client interface redesign**: get_storage_data() returns raw bytes, update_storage() uses dual metadata
- **Performance improvements**: Reduced serialization overhead in data transfer operations
- **Maintained backward compatibility**: Deprecated methods preserved with warnings

#### Linting Cleanup (2025-07-22)
- Fixed 22 out of 26 ruff linting errors
- Remaining 4 E402 errors are intentional for PyTorch backend registration order
- All parameter name mismatches resolved in `_remote_orchestrator.py`
- Unused variables cleaned up across test files
- Set comprehension syntax improvements

#### Minimal Regression Test Suite (2025-07-22)
- Created `tests/test_regression.py` with 12 critical tests for every commit
- Added pytest markers for test categorization (critical, fast, slow, integration)
- Regression tests cover: imports, device creation, basic operations, transfers, gradients
- Target runtime: <30 seconds for critical tests, 2-5 minutes for fast functional tests

#### Architectural Refactoring (2025-07-22)
- **Service Extraction**: Extracted TensorTransferService and StorageMachineResolver from monolithic orchestrator
- **Simple Dispatch**: Replaced over-engineered strategy pattern with straightforward if/elif logic
- **Dependency Injection**: Added ServiceContainer for managing service dependencies and reducing circular imports
- **Provider Standardization**: Enhanced client interface with simplified, focused parameters (lazy_allocation for storage)
- **Clean Boundaries**: Established early conversion boundary where tensors become metadata at PyTorch integration layer
- **Eliminated Technical Debt**: Removed deprecated fields, circular import workarounds, and large conditional logic blocks
- **Simplified Architecture**: Removed over-engineered ConnectionPoolManager in favor of direct machine.get_client() calls
- **Simplified Error Handling**: Removed custom exception hierarchy in favor of descriptive RuntimeError messages

#### Current Status
- **Production-ready architecture** with custom PyTorch integration following pytorch-npu patterns
- **Metadata hash system** with `tensor.metadata_hash` property for enhanced debugging
- **RPC batching optimization** reducing network overhead with background thread processing
- **Multi-provider support** with Modal (production) and Mock (development) implementations
- **HuggingFace integration** enabling direct remote model loading and parameter linking
- **Comprehensive test coverage** with critical regression tests and extended functional tests
- **Zero memory overhead** custom implementations maintaining efficiency
- **Thread-safe operations** throughout the entire stack
- **Clean error handling** with descriptive messages and proper error propagation
- **Extensible architecture** ready for additional cloud providers

#### Public API Updates (2025-08-12)
- **Simplified machine creation**: Use `RemoteMachine(provider, gpu_type)` instead of factory functions
- **Updated public API**: Proper `__all__` exports with clean module organization
- **HuggingFace API**: Direct import from main module (`mycelya_torch.load_huggingface_model`)
- **Logging utilities**: Comprehensive logging control functions exported in public API
- **Device management**: `get_all_machines()` utility for enumerating registered devices

#### Documentation and Examples (2025-08-12)
- **Usage examples**: Updated to reflect current API patterns
- **Example scripts**: 4 example files demonstrating HuggingFace integration and performance comparisons
- **Build documentation**: Added comprehensive build configuration details
- **File structure**: Complete mapping of all source files and their purposes

Last updated: 2025-08-12