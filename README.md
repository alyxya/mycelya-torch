# Mycelya: Remote GPU Computing for PyTorch

Run your PyTorch code on powerful cloud GPUs without changing a single line. Mycelya transparently executes tensor operations on remote cloud infrastructure while you work locally.

```python
import torch
import mycelya_torch

# Create a remote machine with cloud GPU
machine = mycelya_torch.RemoteMachine("modal", "A100-40GB")

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

## Supported GPUs

| GPU | Memory | Best For |
|-----|--------|----------|
| T4 | 16GB | Development, small models |
| L4 | 24GB | Inference, medium models |
| A10G | 24GB | Training, inference |
| A100-40GB | 40GB | Large model training |
| A100-80GB | 80GB | Very large models |
| H100 | 80GB | Latest generation, fastest |
| H200 | 141GB | Extreme memory requirements |
| B200 | 192GB | Next-gen AI workloads |

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.1+
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
machine = mycelya_torch.RemoteMachine("modal", "A100-40GB")
device = machine.device()

# Your PyTorch code runs on remote GPU
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
result = x @ y

# Transfer result back when needed
result_local = result.cpu()
print(f"Computation done on {device}, result shape: {result.shape}")
```

### Neural Network Training
```python
import torch
import torch.nn as nn
import mycelya_torch

# Set up remote machine
machine = mycelya_torch.RemoteMachine("modal", "A100-40GB")
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
import mycelya_torch
from transformers import AutoTokenizer

# Create remote machine
machine = mycelya_torch.RemoteMachine("modal", "A100-80GB")

# Load model directly on remote GPU (no local download!)
model = mycelya_torch.load_huggingface_model(
    "microsoft/DialoGPT-medium",
    machine,
    torch_dtype=torch.float16
)

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

## Common Patterns

### Mixed Local/Remote Computing
```python
# Some operations local, others remote
local_data = torch.randn(1000, 100)  # On CPU
remote_weights = torch.randn(100, 10, device=machine.device())  # On remote GPU

# Automatically handles data transfer
result = local_data @ remote_weights  # local_data moved to remote GPU
final_result = result.cpu()  # Move back to local CPU
```

### Working with Multiple GPUs
```python
# Different machines for different tasks
training_machine = mycelya_torch.RemoteMachine("modal", "A100-80GB")
inference_machine = mycelya_torch.RemoteMachine("modal", "T4")

# Train on powerful GPU
model = train_model(training_machine.device())

# Switch to cheaper GPU for inference
model = model.to(inference_machine.device())
predictions = model(test_data.to(inference_machine.device()))
```

## Getting Help

- **Examples**: Check the `examples/` directory for complete working examples
- **Issues**: Report bugs at [GitHub Issues](https://github.com/alyxya/mycelya-torch/issues)
- **Documentation**: See `CLAUDE.md` for technical details

## Local Development

```bash
# Clone the repo
git clone https://github.com/alyxya/mycelya-torch.git
cd mycelya-torch

# Build C++ extensions for development
python setup.py build_ext --inplace

# Run tests
pytest tests/test_regression.py::TestCriticalRegression -v
```

## License

AGPL-3.0-or-later - See LICENSE file for details.

---

**Ready to accelerate your PyTorch workflows?** Install Mycelya and start computing on cloud GPUs in minutes!