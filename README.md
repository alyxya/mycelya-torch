# Mycelya - PyTorch Remote Execution

A PyTorch extension that enables transparent remote execution of tensor operations on cloud GPU infrastructure. Execute PyTorch code on remote GPUs without changing your existing code.

## Overview

Mycelya is a PyTorch extension that uses a pure tensor ID-based architecture to run PyTorch operations on remote cloud GPUs while keeping only metadata (shape, dtype) stored locally. This provides memory-efficient distributed computing with zero local memory overhead for remote tensor data.

**Key Features:**
- **Transparent remote execution** - Your PyTorch code runs unchanged on remote GPUs
- **Zero local memory** - Only tensor metadata stored locally, actual data stays on remote GPUs
- **Multiple GPU support** - T4, L4, A10G, A100, H100, H200, B200 across cloud providers
- **Full autograd support** - Gradients work seamlessly across local/remote boundaries
- **Provider abstraction** - Currently supports Modal, designed for multiple cloud providers

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
result = x @ y  # Matrix multiplication happens on remote A100

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

## Architecture

Mycelya uses a three-layer architecture:

1. **C++ Layer** - Custom PyTorch PrivateUse1 backend with tensor ID allocation
2. **Python Coordination** - Local tensor metadata management and operation dispatch
3. **Remote Execution** - Cloud provider implementations (currently Modal)

### Memory Efficiency

- **Zero local storage** for remote tensor data
- **64-bit tensor IDs** stored as data pointers for efficient lookup
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
├── __init__.py          # Public API and PyTorch backend registration
├── device.py           # RemoteMachine and device management
├── _aten_impl.py        # ATen operation dispatch system
├── _remote_orchestrator.py # Remote execution coordination
├── _device_daemon.py    # Local storage ID registry
└── csrc/               # C++ backend implementation
    ├── RemoteMem.cpp   # Custom allocator
    └── RemoteHooks.cpp # PyTorch PrivateUse1 hooks

_mycelya_torch_modal/
├── modal_app.py        # Modal cloud GPU integration
└── client.py          # Modal client implementation
```

## Limitations

- **Cross-device operations**: Cannot operate between different remote machines directly
- **View operations**: Some advanced view operations may require CPU transfer
- **Provider dependency**: Currently requires Modal account for cloud access

## Contributing

This project is licensed under AGPL-3.0. All contributions must maintain the AGPL license headers.

### Development Setup

1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Run tests to verify setup: `pytest tests/`
4. Follow the existing code style and patterns

## License

Copyright (C) 2025 alyxya
SPDX-License-Identifier: AGPL-3.0-or-later

This project is licensed under the GNU Affero General Public License v3.0 or later.
