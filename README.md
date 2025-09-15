# Mycelya: Remote GPU Computing for PyTorch

Run your PyTorch code on powerful cloud GPUs without changing a single line. Mycelya transparently executes tensor operations on remote cloud infrastructure while you work locally.

```python
import torch
import mycelya_torch

# Create a remote machine with cloud GPU
machine = mycelya_torch.RemoteMachine("modal", "A100")

# Your existing PyTorch code just works
x = torch.randn(1000, 1000, device=machine.device())
y = torch.randn(1000, 1000, device=machine.device())
result = x @ y  # Computed on remote A100!

print(f"Result computed on {result.device}: {result.shape}")
```

## Why Mycelya?

- **🚀 Powerful Hardware** - Access H100, A100, and other high-end GPUs instantly
- **🔧 Zero Code Changes** - Your existing PyTorch code works unchanged
- **⚡ Smart Batching** - Automatically batches operations to minimize network overhead
- **🤖 HuggingFace Ready** - Load models directly on remote GPUs without downloading
- **🎯 Remote Functions** - Execute custom functions remotely with the @remote decorator

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
device = machine.device()

# Your PyTorch code runs on remote GPU
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
result = x @ y

# Transfer result back when needed
result_local = result.cpu()
print(f"Computation done on {device}, result shape: {result.shape}")
```

### Remote Function Execution
```python
import torch
import mycelya_torch

# Create remote machine
machine = mycelya_torch.RemoteMachine("modal", "A100")
device = machine.device()

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
device = machine.device()

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

### Load HuggingFace Models
```python
import torch
import mycelya_torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create remote machine
machine = mycelya_torch.RemoteMachine("modal", "A100")

# Load model architecture (no weights yet)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    torch_dtype=torch.float16,
    device_map=None  # Don't load weights yet
)

# Load weights directly on remote GPU (no local download!)
remote_state_dicts = mycelya_torch.load_huggingface_state_dicts(
    "microsoft/DialoGPT-medium",
    machine.device()
)

# Load the remote weights into the model
model.load_state_dict(remote_state_dicts[""], strict=True)  # "" is root directory

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Generate text on remote GPU
def chat(message):
    inputs = tokenizer(message, return_tensors="pt")
    inputs = {k: v.to(machine.device()) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Use it
response = chat("Hello! How are you today?")
print(response)
```

## Local Development

```bash
# Clone the repo
git clone https://github.com/alyxya/mycelya-torch.git
cd mycelya-torch

# Build C++ extensions for development
python setup.py build_ext --inplace

# Run tests
pytest tests/test_regression.py::TestCriticalRegression -v  # Critical tests (<30s)
pytest tests/test_regression.py -v                         # Fast tests (~2-5min)
pytest tests/ -v                                           # Full test suite (~10-30min)

# Code quality
ruff check .     # Linting
ruff format .    # Formatting
```

### Mock Client for Testing

For development and testing without cloud resources, use the mock client:

```python
import mycelya_torch

# Use mock client - runs locally using Modal's .local() execution
# GPU type is ignored for mock client
machine = mycelya_torch.RemoteMachine("mock")
device = machine.device()

# All operations run locally but through the same API
x = torch.randn(100, 100, device=device)
y = x @ x  # Executed locally
```

## License

AGPL-3.0-or-later - See LICENSE file for details.

