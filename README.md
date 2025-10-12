# Mycelya: Remote GPU Computing for PyTorch

Run your PyTorch code anywhere and power it with cloud GPUs. Mycelya integrates a remote GPU backend into PyTorch, allowing tensor operations to execute on cloud infrastructure with minimal code changes.

```python
import torch
import mycelya_torch

# Create a remote machine with cloud GPU
machine = mycelya_torch.RemoteMachine("modal", "A100")
cuda_device = machine.device("cuda")

# Your existing PyTorch code just works
x = torch.randn(1000, 1000, device=cuda_device)
y = torch.randn(1000, 1000).to(cuda_device)  # Move tensor to remote GPU
result = x @ y  # Computed on remote A100!

# Transfer result back to local machine
result_local = result.cpu()
print(f"Result: {result_local}")
```


## Supported GPUs (Modal)

**8 GPU Types**: T4, L4, A10G, A100, L40S, H100, H200, B200

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- Modal account (free tier available)

**Note**: Modal is currently the only supported GPU cloud provider. Support for other providers (AWS, etc.) will be added in future releases.

### Install
```bash
pip install git+https://github.com/alyxya/mycelya-torch.git
```

### Setup Modal
```bash
modal setup
```

## Quick Start

### MNIST Training
```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mycelya_torch

# Setup remote GPU
machine = mycelya_torch.RemoteMachine("modal", "T4")
device = machine.device("cuda")

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Define model - all operations run on remote GPU
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train on remote GPU
for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
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

@mycelya_torch.remote
def complex_computation(x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    # Multiple operations executed remotely
    y = x * scale
    z = torch.relu(y)
    w = torch.softmax(z, dim=-1)
    return w.sum(dim=0)

# Create tensors on remote GPU
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Functions automatically execute on remote GPU
result1 = matrix_multiply(x, y)  # Executes remotely
result2 = complex_computation(x, scale=3.0)  # Executes remotely

print(f"Results computed on {result1.device}")
```

## API Reference

### `RemoteMachine`

```python
# Create remote machine with cloud GPU
machine = mycelya_torch.RemoteMachine(
    "modal", "A100",
    gpu_count=1,                                  # 1-8 GPUs
    pip_packages=["transformers", "diffusers"],   # Pre-install for remote functions
    idle_timeout=300,                             # Pause after 5 min inactivity
    modal_timeout=3600                            # Function timeout (default: 1 hour)
)
device = machine.device("cuda")

# Install packages dynamically
machine.pip_install("numpy")

# Pause to save costs, resume when needed
machine.pause()   # Offload state and stop compute
machine.resume()  # Restart and reload state
```

### `@remote` Decorator

```python
# Execute entire function remotely
@mycelya_torch.remote
def custom_function(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x @ x.T)

result = custom_function(x)  # Runs on remote GPU

# Async execution
@mycelya_torch.remote(run_async=True)
def async_function(x: torch.Tensor) -> torch.Tensor:
    return x @ x.T

future = async_function(x)
result = future.result()
```

## License

AGPL-3.0-or-later - See LICENSE file for details.