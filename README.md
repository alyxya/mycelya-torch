# Mycelya: Remote GPU Computing for PyTorch

Run your PyTorch code anywhere and power it with cloud GPUs. Mycelya integrates a remote GPU backend into PyTorch, allowing tensor operations to execute on cloud infrastructure with minimal code changes.

```python
import torch
import mycelya_torch

# Create a remote machine with cloud GPU
machine = mycelya_torch.RemoteMachine("modal", "A100")

# Your existing PyTorch code just works
x = torch.randn(1000, 1000, device=machine.device("cuda"))
y = torch.randn(1000, 1000, device=machine.device("cuda"))
result = x @ y  # Computed on remote A100!

print(f"Result computed on {result.device}: {result.shape}")
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

### Basic Usage
```python
import torch
import mycelya_torch

# Create remote machine
machine = mycelya_torch.RemoteMachine("modal", "A100")
device = machine.device("cuda")

# Your PyTorch code runs on remote GPU
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
result = x @ y

# Transfer result back to local machine
result_local = result.cpu()
print(f"Result: {result_local}")
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

### Neural Network Training
```python
import torch
import torch.nn as nn
import mycelya_torch

# Set up remote machine
machine = mycelya_torch.RemoteMachine("modal", "A100")
device = machine.device("cuda")

# Define your model (works exactly like normal PyTorch)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop - all computations happen on remote GPU
for epoch in range(10):
    for batch_data, batch_labels in dataloader:
        # Move data to remote GPU
        data = batch_data.to(device)
        labels = batch_labels.to(device)

        # Forward pass, loss, backward pass all on remote GPU
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```


## License

AGPL-3.0-or-later - See LICENSE file for details.