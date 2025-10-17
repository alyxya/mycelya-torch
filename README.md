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

### Prerequisites
- Python 3.10+
- PyTorch 2.6-2.8 (must be pre-installed)
- Modal account (free tier available)

### Quick Install

```bash
# 1. Install PyTorch 2.6-2.8 first
pip install "torch>=2.6,<2.9"

# 2. Install mycelya-torch
pip install --no-build-isolation mycelya-torch

# Or install from source
pip install --no-build-isolation git+https://github.com/alyxya/mycelya-torch.git

# 3. Setup Modal if not setup yet
modal setup
```

**Note**: Modal is currently the only supported GPU cloud provider. Support for other providers (AWS, etc.) will be added in future releases.

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

### LLM Inference
```python
import mycelya_torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define remote functions for model loading and inference
@mycelya_torch.remote
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return model, tokenizer

@mycelya_torch.remote
def generate_text(model, tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# Create remote machine with required packages
machine = mycelya_torch.RemoteMachine(
    "modal", "A100", pip_packages=["transformers", "accelerate"]
)

# Load model and generate text - all on remote GPU
model, tokenizer = load_model("Qwen/Qwen3-4B-Instruct-2507")
content = generate_text(model, tokenizer, "Explain quantum computing briefly.")
print("Response:", content)
```

### Diffusion Model Inference
```python
import torch
import mycelya_torch
from diffusers import DiffusionPipeline

# Define remote functions for pipeline loading and image generation
@mycelya_torch.remote
def load_pipeline(model_name: str):
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    return pipe

@mycelya_torch.remote
def generate_image(pipe, prompt: str, height: int, width: int, seed: int):
    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(seed)
    ).images[0]
    return image

# Create remote machine with required packages
machine = mycelya_torch.RemoteMachine(
    "modal", "H100", pip_packages=["diffusers", "transformers", "accelerate"]
)

# Load pipeline and generate image - all on remote GPU
pipe = load_pipeline("Qwen/Qwen-Image")
image = generate_image(pipe, "A cat holding a sign that says hello world",
                       height=1024, width=1024, seed=0)
image.save("cat.png")
```

## API Reference

### `RemoteMachine`

```python
# Create remote machine with cloud GPU
machine = mycelya_torch.RemoteMachine(
    "modal", "A100",
    gpu_count=1,                                  # 1-8 GPUs
    pip_packages=["transformers", "diffusers"],   # Pre-install for remote functions
    idle_timeout=300                              # Pause after 5 min inactivity
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
