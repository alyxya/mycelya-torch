# Mycelya - PyTorch Remote Execution

A PyTorch extension that enables transparent remote execution of tensor operations on cloud GPU infrastructure. Features sequential tensor IDs, custom PyTorch integration, and RPC batching for efficient distributed computing.

## Overview

Mycelya is a PyTorch extension that uses a sequential tensor ID architecture with custom PyTorch integration to run operations on remote cloud GPUs. Each mycelya tensor has a unique incremental ID accessible via `tensor.id` property, enabling efficient debugging and monitoring.

**Key Features:**
- **Sequential tensor IDs** - Each tensor has unique incremental ID (1, 2, 3...) via `tensor.id` property
- **Custom PyTorch integration** - Complete TensorImpl/StorageImpl following pytorch-npu patterns
- **RPC batching** - Background thread processing reduces network overhead
- **Zero local memory** - Only tensor metadata stored locally, actual data stays on remote GPUs
- **Multiple GPU support** - T4, L4, A10G, A100, H100, H200, B200 across cloud providers
- **HuggingFace integration** - Direct remote model loading without data transfer
- **Multi-provider support** - Modal (production), Mock (development), extensible for others
- **Full autograd support** - Gradients work seamlessly across local/remote boundaries

## Installation

```bash
pip install torch>=2.0.0 modal>=0.60.0
pip install -e .
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- Modal account and API key for cloud GPU access

## Quick Start

```python
import torch
import mycelya_torch

# Create a remote machine with an A100 GPU
machine = mycelya_torch.create_modal_machine("A100-40GB")

# Operations automatically execute on the remote A100
x = torch.randn(1000, 1000, device=machine.device())
y = torch.randn(1000, 1000, device=machine.device())

# Each tensor has a unique sequential ID for debugging
print(f"Tensor x ID: {x.id}")  # 1
print(f"Tensor y ID: {y.id}")  # 2

result = x @ y  # Matrix multiplication happens on remote A100
print(f"Result tensor ID: {result.id}")  # 3

# Transfer result back when needed
result_cpu = result.cpu()
print(f"Result shape: {result_cpu.shape}")
```

## Supported GPU Types

```python
import mycelya_torch

# Available GPU types
gpu_types = [
    "T4", "L4", "A10G",
    "A100-40GB", "A100-80GB",
    "L40S", "H100", "H200", "B200"
]

machine = mycelya_torch.create_modal_machine("H100")
```

## Advanced Usage

### Neural Network Training

```python
import torch
import torch.nn as nn
import mycelya_torch

# Create remote machine
machine = mycelya_torch.create_modal_machine("A100-40GB")
device = machine.device()

# Define model and move to remote device
model = nn.Linear(784, 10).to(device)
optimizer = torch.optim.Adam(model.parameters())

# Training loop - all operations happen remotely
for batch_idx, (data, target) in enumerate(dataloader):
    data, target = data.to(device), target.to(device)
    
    # Debug tensor IDs for monitoring
    print(f"Batch {batch_idx}: Data ID {data.id}, Target ID {target.id}")

    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()  # Gradients computed on remote GPU
    optimizer.step()
```

### Mixed Local/Remote Operations

```python
import torch
import mycelya_torch

machine = mycelya_torch.create_modal_machine("T4")
device = machine.device()

# Create tensors on different devices
local_tensor = torch.randn(100, 100)  # CPU
remote_tensor = torch.randn(100, 100, device=device)  # Remote GPU

# Automatic transfer for operations
result = local_tensor @ remote_tensor.cpu()  # Transfer remote to CPU first

# Or transfer local to remote
local_on_remote = local_tensor.to(device)
result = local_on_remote @ remote_tensor  # Both on remote GPU
```

### HuggingFace Integration

```python
import torch
import mycelya_torch
import mycelya_torch.huggingface as mhf

# Create remote machine
machine = mycelya_torch.create_modal_machine("T4")

# Load model directly on remote GPU (no data transfer)
model = mhf.load_model_remote(
    "microsoft/DialoGPT-medium", 
    machine, 
    torch_dtype=torch.float16
)

# All parameters are already on remote GPU with unique IDs
for name, param in model.named_parameters():
    print(f"{name}: tensor ID {param.id}, device {param.device}")
```

### Mock Provider for Development

```python
# Use local execution for development/testing
machine = mycelya_torch.create_mock_machine()
device = machine.device()

# Same API, but executes locally using Modal's .local() calls
x = torch.randn(100, 100, device=device)
result = x @ x.T  # Executed locally
print(f"Local execution result ID: {result.id}")
```

## Architecture

Mycelya uses a three-layer architecture with custom PyTorch integration:

1. **C++ Layer** - Custom TensorImpl, StorageImpl, and Allocator with sequential tensor IDs
2. **Python Coordination** - RPC batching, metadata management, and operation dispatch
3. **Remote Execution** - Multi-provider system (Modal, Mock, extensible)

### Memory Efficiency

- **Zero local storage** for remote tensor data with custom TensorImpl/StorageImpl
- **Sequential tensor IDs** (1, 2, 3...) stored in TensorImpl for debugging
- **RPC batching** reduces network calls with background thread processing
- **Dual storage architecture** supports lazy allocation and realized storage
- **Meta tensor integration** for shape inference without data transfer

## Development

### Running Tests

```bash
# Minimal regression tests (run on every commit, <30 seconds)
pytest tests/test_regression.py::TestCriticalRegression -v

# Fast functional tests (run on PR reviews, ~2-5 minutes)
pytest tests/test_regression.py -v

# Full test suite (comprehensive, run on releases)
pytest tests/ -v

# Run specific test categories
pytest tests/test_basic_operations.py -v
pytest tests/test_autograd_basic.py -v
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .
```

### Project Structure

```
mycelya_torch/
├── __init__.py              # Public API and PyTorch backend registration
├── device.py               # RemoteMachine and device management
├── _aten_impl.py            # ATen operation dispatch with meta inference
├── _remote_orchestrator.py  # Remote execution with RPC batching
├── _device_daemon.py        # Device registry and storage operations
├── _batching.py            # RPC batching system
├── _huggingface_utils.py   # HuggingFace model integration
├── _logging.py             # Centralized logging system
├── backends/
│   ├── client_interface.py  # Standardized provider interface
│   ├── modal/client.py     # Modal cloud provider
│   └── mock/client.py      # Mock provider for development
└── csrc/                   # C++ backend implementation
    ├── MycelyaTensorImpl.cpp   # Custom tensor implementation
    ├── MycelyaStorageImpl.cpp  # Custom storage implementation
    ├── MycelyaAllocator.cpp    # Enhanced allocator
    └── MycelyaHooks.cpp        # PyTorch PrivateUse1 hooks

_mycelya_torch_modal/
├── modal_app.py            # Modal server with lazy/realized storage
└── client.py              # Modal client with batching
```

## Limitations

- **Cross-device operations**: Cannot operate between different remote machines directly
- **Provider dependency**: Modal provider requires Modal account for cloud access
- **Tensor ID scope**: Tensor IDs are unique within single Python process (not across processes)
- **Background batching**: RPC batching adds minor latency for individual operations

## Contributing

This project is licensed under AGPL-3.0. All contributions must maintain the AGPL license headers.

### Development Setup

1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Run critical regression tests: `pytest tests/test_regression.py::TestCriticalRegression -v`
4. Use Mock provider for local development: `mycelya_torch.create_mock_machine()`
5. Follow existing code style and patterns with ruff formatting

## License

Copyright (C) 2025 alyxya
SPDX-License-Identifier: AGPL-3.0-or-later

This project is licensed under the GNU Affero General Public License v3.0 or later.
