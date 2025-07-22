# torch-remote: PyTorch Remote Tensor Execution System

A PyTorch extension that enables transparent remote execution of tensor operations on cloud GPU infrastructure. The system uses a pure tensor ID-based architecture for memory-efficient distributed computing.

## Architecture Overview

- **Pure Tensor ID System**: Remote tensors store only metadata locally (ID, shape, dtype) with actual data on remote GPUs
- **Three-Layer Architecture**: C++ Backend, Python Coordination, Remote Execution
- **Multi-GPU Support**: 9 GPU types supported (T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200)
- **Provider Abstraction**: Pluggable backend system (Modal currently, extensible for other providers)

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

## Key Components

### Core Modules
- `torch_remote/__init__.py` - Public API and PyTorch PrivateUse1 backend registration
- `torch_remote/_aten_impl.py` - ATen operation dispatch system using strategy pattern
- `torch_remote/_remote_orchestrator.py` - Remote execution orchestration with dependency injection
- `torch_remote/_device_daemon.py` - Local tensor ID registry and device daemon interface  
- `torch_remote/device.py` - RemoteMachine abstraction and device management

### Service Architecture (New)
- `torch_remote/services/tensor_transfer.py` - Extracted tensor serialization/transfer logic
- `torch_remote/services/storage_resolver.py` - Storage-to-machine mapping and validation
- `torch_remote/core/container.py` - Dependency injection container for service management

### Operation Dispatch System (New)
- `torch_remote/dispatch/operation_classifier.py` - Centralized operation categorization
- `torch_remote/dispatch/execution_strategies.py` - Strategy pattern for operation execution

### Provider Interface
- `torch_remote/backends/client_interface.py` - Standardized provider interface with agnostic parameters
- `torch_remote/backends/modal/client.py` - Modal provider implementation

### Remote Execution Provider
- `_torch_remote_modal/modal_app.py` - Modal cloud GPU integration with multi-GPU support

### C++ Backend Integration
- `torch_remote/csrc/RemoteMem.cpp` - Custom allocator storing tensor IDs as data pointers
- `torch_remote/csrc/RemoteHooks.cpp` - PyTorch PrivateUse1 backend implementation

## Current Architecture (Post-Cleanup)

### Key Design Principles
- **Clean Input/Output Separation**: Efficient data transfer with clear boundaries
- **Pure Tensor ID Coordination**: 64-bit tensor IDs for all remote coordination
- **Zero Local Memory**: No tensor data stored locally for remote tensors
- **Eliminated Fallbacks**: All remote operations must succeed remotely or fail clearly

### Memory Efficiency
- **50% memory reduction** compared to previous implementations  
- Remote tensors use meta tensors locally (no data storage)
- C++ allocator stores tensor ID as data pointer for efficient lookup
- Automatic cleanup via PyTorch's memory management system

### Operation Dispatch Flow
1. **Local Operations**: View operations executed locally with shared storage IDs
2. **Remote Operations**: All compute operations dispatched to remote GPUs
3. **Meta Execution**: Shape inference using PyTorch meta tensors
4. **Data Transfer**: Only when crossing device boundaries (CPU ↔ Remote)

## Usage Patterns

### Basic Usage
```python
import torch
import torch_remote

# Create remote machine
machine = torch_remote.create_modal_machine("T4")

# Operations automatically execute on remote GPU
x = torch.randn(1000, 1000, device=machine.device())
y = torch.randn(1000, 1000, device=machine.device())
result = x @ y  # Matrix multiplication on remote T4
```

### Advanced Usage
```python
# Neural network training
model = nn.Linear(784, 10).to(machine.device())
optimizer = torch.optim.Adam(model.parameters())

# Full training loop on remote GPU
for data, target in dataloader:
    data, target = data.to(machine.device()), target.to(machine.device())
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # Gradients computed remotely
    optimizer.step()
```

## Implementation Details

### Tensor ID System
- **64-bit random IDs** generated for each remote tensor
- **Stored as data pointer** in C++ allocator for efficient lookup
- **Collision detection** with retry mechanism
- **Automatic cleanup** when tensors are garbage collected

### Error Handling
- **Cross-device operation prevention**: Clear errors when mixing different machines
- **Stale reference detection**: Validation of tensor references
- **Connection management**: Automatic reconnection on failures
- **Comprehensive exception hierarchy**: RemoteTensorError, StaleReferenceError, etc.

### Provider Interface
- **Standardized client interface** for multiple providers
- **Modal implementation** as reference implementation
- **Extensible architecture** for RunPod, Lambda Labs, etc.

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

#### Tensor ID Architecture Rules
- **Never store tensor data locally** for remote tensors - only metadata
- **64-bit tensor IDs** stored as data pointers in C++ allocator
- **Clean separation** between local metadata and remote storage

#### Provider Implementation Patterns  
- Follow Modal implementation pattern in `_torch_remote_modal/`
- Implement standardized client interface
- Support multi-GPU configuration
- Handle connection lifecycle properly

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
- **Strategy Pattern**: Implemented operation classification and execution strategies for clean dispatch
- **Dependency Injection**: Added ServiceContainer for managing service dependencies and reducing circular imports
- **Provider Standardization**: Enhanced client interface with simplified, focused parameters (lazy_allocation for storage)
- **Clean Boundaries**: Established early conversion boundary where tensors become metadata at PyTorch integration layer
- **Eliminated Technical Debt**: Removed deprecated fields, circular import workarounds, and large conditional logic blocks
- **Simplified Architecture**: Removed over-engineered ConnectionPoolManager in favor of direct machine.get_client() calls
- **Simplified Error Handling**: Removed custom exception hierarchy in favor of descriptive RuntimeError messages

#### Current Status
- Core functionality stable and tested
- Memory efficiency optimizations in place
- Clean error handling throughout
- Modal provider fully functional
- Minimal regression test suite in place
- Clean service-oriented architecture with dependency injection
- Strategy pattern for extensible operation dispatch
- Ready for additional provider implementations

Last updated: 2025-07-22