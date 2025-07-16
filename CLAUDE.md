# torch-remote: PyTorch Remote Tensor Execution System

A PyTorch extension that enables transparent remote execution of tensor operations on cloud GPU infrastructure. The system uses a pure tensor ID-based architecture for memory-efficient distributed computing.

## Architecture Overview

- **Pure Tensor ID System**: Remote tensors store only metadata locally (ID, shape, dtype) with actual data on remote GPUs
- **Three-Layer Architecture**: PyTorch Integration (C++), Local Coordination (Python), Remote Execution (Multi-Provider)
- **Multi-GPU Support**: 9 GPU types supported (T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200)
- **Provider Abstraction**: Pluggable backend system (Modal currently, more providers planned)

## Development Commands

To run tests:
```bash
pytest tests/ -v
```

To run linting:
```bash
# Add your lint command here (e.g., flake8, black, etc.)
```

To run type checking:
```bash
# Add your typecheck command here (e.g., mypy)
```

## Key Components

### Core Modules
- `torch_remote/__init__.py` - Registers "remote" as PyTorch PrivateUse1 backend
- `torch_remote/_aten_impl.py` - Main operation dispatch system
- `torch_remote/_remote_executor.py` - Remote execution coordination
- `torch_remote/_device_daemon.py` - Local tensor ID registry and device simulation
- `torch_remote/device.py` - Device abstraction and backend management

### Remote Execution
- `torch_remote_execution/modal_app.py` - Modal cloud GPU integration with multi-GPU support
- `torch_remote/backends/modal/` - Modal provider implementation

### C++ Extension
- `torch_remote/csrc/RemoteMem.cpp` - Custom allocator with tensor ID generation
- `torch_remote/csrc/RemoteHooks.cpp` - PyTorch PrivateUse1 backend implementation

## Recent Architectural Changes

- **Clean Input/Output Separation**: Distinguishes between input tensors (read data) and output tensors (write data) for efficient data transfer
- **Pure Tensor ID Coordination**: Removed local device simulation, uses 64-bit tensor IDs for all coordination
- **Enhanced Memory Efficiency**: Zero local memory overhead for remote tensor data
- **Improved Error Handling**: Eliminated silent fallbacks, all remote operations must succeed remotely

## Memory Efficiency

- Remote tensors use meta tensors locally (no data storage)
- 50% memory reduction compared to previous implementations
- C++ allocator stores tensor ID as data pointer for efficient lookup
- Automatic cleanup via PyTorch's memory management system

## Usage Pattern

```python
import torch
import torch_remote

# Create remote device
device = torch_remote.create_modal_device(gpu_type="A100")

# Operations automatically execute on remote GPU
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
result = x @ y  # Matrix multiplication on remote A100
```

## Documentation Maintenance

**IMPORTANT**: This file should be updated whenever making significant changes to the codebase. 

### Update This File When:
- Adding new core modules or changing module responsibilities
- Modifying the architecture (tensor ID system, execution flow, etc.)
- Adding/removing provider backends or GPU support
- Changing development commands (test, lint, typecheck)
- Making breaking changes to the public API
- Adding new important implementation details

### Development Notes
Use this section to track important learnings and implementation details that should be preserved:

- **License Compliance**: All source files must maintain AGPL license headers at the top
- **Tensor ID Architecture**: The system relies on 64-bit tensor IDs stored as data pointers in the C++ allocator
- **Remote vs Local**: Never store actual tensor data locally for remote tensors - only metadata
- **Provider Patterns**: New backends should follow the Modal implementation pattern in `torch_remote/backends/`

### Recent Development Notes
(Update this section as code evolves)

Last updated: 2025-07-16