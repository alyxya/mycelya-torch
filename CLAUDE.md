# Mycelya: PyTorch Remote Tensor Execution System

A PyTorch extension that enables transparent remote execution of tensor operations on cloud GPU infrastructure. The system uses a sequential tensor ID architecture with custom PyTorch integration for memory-efficient distributed computing.

## Architecture Overview

- **Metadata Hash System**: Remote tensors have unique metadata-based hash IDs for internal debugging and identification
- **Custom PyTorch Integration**: Complete custom TensorImpl, StorageImpl, and Allocator following pytorch-npu patterns
- **Three-Layer Architecture**: C++ Backend, Python Coordination, Remote Execution
- **Multi-GPU Support**: 10 GPU types supported (T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200)
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
- `mycelya_torch/__init__.py` - Public API and PyTorch PrivateUse1 backend registration
- `mycelya_torch/_orchestrator.py` - Remote execution orchestration with RPC batching integration
- `mycelya_torch/_backend_hooks.py` - PyTorch backend hooks and C++ interface bridge
- `mycelya_torch/_device.py` - DeviceManager for mapping local device indices to remote device info
- `mycelya_torch/_machine.py` - RemoteMachine abstraction supporting Modal and Mock providers

### ATen Operation System
- `mycelya_torch/aten/__init__.py` - PyTorch library registrations for ATen operations
- `mycelya_torch/aten/dispatch.py` - Main operation dispatch with fallback kernel and meta tensor inference
- `mycelya_torch/aten/copy.py` - Copy and transfer operations between devices
- `mycelya_torch/aten/meta.py` - Meta tensor inference and shape computation
- `mycelya_torch/aten/scalar.py` - Scalar operations and local execution
- `mycelya_torch/aten/utils.py` - Utilities for operation handling and view management

### Utility and Support Modules
- `mycelya_torch/_storage.py` - Integer-based storage ID system with thread-safe generation and cross-device validation
- `mycelya_torch/_huggingface_utils.py` - Model loading and tensor linking utilities for HuggingFace models
- `mycelya_torch/_logging.py` - Centralized logging configuration with hierarchical loggers

### Modular Operation Dispatch
- **Organized ATen handlers** in dedicated `aten/` directory with clear separation of concerns
- **Fallback kernel dispatch** in `aten/dispatch.py` with comprehensive operation coverage
- **Specialized operation handlers** for copy, meta, scalar, and utility operations
- **Clean PyTorch integration** with proper library registration and backend hooks

### Provider Interface
- `mycelya_torch/backends/base_client.py` - Standardized provider interface with caching and RPC batching support
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
- **Metadata Hash System**: Unique metadata-based hash IDs for internal debugging and identification
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
- **Thread-safe storage ID generation**: Atomic counter for unique sequential storage IDs (1, 2, 3...)

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
    # Tensors have internal metadata hash IDs for debugging
    print(f"Data tensor shape: {data.shape}")
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
    print(f"{name}: tensor shape {param.shape}, device {param.device}")
```

## Implementation Details

### Storage ID System
- **Sequential incremental storage IDs** (1, 2, 3...) generated for each remote tensor storage
- **Internal API integration**: Metadata hash computation accessible via internal utility functions
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
- Reorganizing code structure or ATen operation handling
- Major C++ implementation changes or performance optimizations
- Updates to build system or development workflows

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
- **Use ruff for linting and formatting** with line length 88 and comprehensive rule selection
- **Run minimal regression tests on every commit** with target runtime <30 seconds
- **Follow existing patterns** for new functionality, especially in ATen operation handling
- **Comprehensive error handling** with clear error messages and proper error propagation
- **Thorough testing** for all new features with appropriate pytest markers
- **Google C++ style** for all C++ source files with consistent formatting
- **Modular organization** following the established `aten/` directory structure
- **Clean separation of concerns** between C++ and Python implementations

#### Testing Strategy
- **Critical regression tests**: 20 essential tests covering core functionality (~30 seconds)
- **Fast functional tests**: Extended coverage for PR reviews (~2-5 minutes)
- **Full test suite**: Comprehensive validation for releases (~10-30 minutes)
- **Test markers**: Use `@pytest.mark.critical` and `@pytest.mark.fast` for categorization