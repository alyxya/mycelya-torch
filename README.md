# PyTorch Modal Labs Remote Execution Backend

A custom PyTorch compilation backend that executes models remotely on Modal Labs GPU infrastructure. This allows you to offload computation to powerful GPUs in the cloud while maintaining the familiar PyTorch API.

## Overview

This project provides:
- **Custom PyTorch Backend**: Integrates with `torch.compile()` to redirect execution to Modal
- **Remote GPU Execution**: Runs your models on Modal's A100 GPUs
- **Automatic Serialization**: Handles tensor and graph serialization/deserialization
- **Fallback Support**: Gracefully falls back to local execution on errors

## Files

- `modal_backend.py` - Main PyTorch backend implementation
- `modal_server.py` - Modal GPU server functions
- `example_usage.py` - Example models and usage patterns
- `requirements_modal.txt` - Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_modal.txt
```

### 2. Set up Modal

```bash
# Install Modal CLI
pip install modal

# Set up Modal account
modal setup

# Deploy the GPU server
modal deploy modal_server.py
```

### 3. Use the Backend

```python
import torch
from modal_backend import compile_for_modal

# Create your model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

# Compile for Modal execution
model_compiled = compile_for_modal(model)

# Use normally - execution happens on Modal GPUs!
input_data = torch.randn(32, 784)
output = model_compiled(input_data)
```

### 4. Run Examples

```bash
python example_usage.py
```

## How It Works

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your Code     │    │  Modal Backend   │    │  Modal GPU      │
│                 │    │                  │    │                 │
│ model = ...     │    │ 1. Intercept     │    │ 3. Execute      │
│ torch.compile() ├────┤    torch.compile │    │    on A100      │
│ output = model()│    │ 2. Serialize     │    │ 4. Return       │
│                 │    │    & Send        │    │    Results      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Execution Flow

1. **Compilation**: `torch.compile(model, backend="modal_backend")` registers the model
2. **Graph Capture**: PyTorch's FX system captures the computation graph  
3. **Serialization**: Tensors and graph are serialized for network transmission
4. **Remote Execution**: Modal GPU function receives and executes the computation
5. **Result Return**: Outputs are serialized back and returned to your code

### Backend Registration

The backend uses PyTorch's `@dynamo.register_backend` decorator:

```python
@dynamo.register_backend
def modal_backend(fx_graph: GraphModule, example_inputs: List[torch.Tensor]):
    # Create remote execution wrapper
    return ModalCompiledFunction(fx_graph, example_inputs)
```

## Supported Operations

The current implementation supports:

- **Basic Operations**: ReLU, Linear layers, Add, Conv2d, Pooling
- **Model Types**: MLPs, CNNs, Transformers  
- **Complex Graphs**: Multi-input/output models
- **Custom Operations**: Extensible operation mapping system

## Modal Server Functions

### Main Execution Function

```python
@app.function(gpu=modal.gpu.A100(size="40GB"))
def execute_pytorch_graph(serialized_graph, tensor_data, metadata):
    # Deserialize inputs -> Execute -> Serialize outputs
```

### Supported GPU Types

- A100 (40GB/80GB)
- A10G
- T4
- H100 (when available)

## Configuration

### Backend Options

```python
# Basic usage
model_compiled = torch.compile(model, backend="modal_backend")

# With custom options
model_compiled = compile_for_modal(model, 
                                  fullgraph=True,     # Compile entire model
                                  dynamic=True)       # Support dynamic shapes
```

### Modal Configuration

Customize GPU type and timeout in `modal_server.py`:

```python
@app.function(
    gpu=modal.gpu.A100(size="80GB"),  # Larger GPU
    timeout=1200,                     # 20 minute timeout
    retries=3                         # More retries
)
```

## Performance Considerations

### When to Use Remote Execution

✅ **Good for**:
- Large models that don't fit locally
- Batch inference workloads  
- Compute-intensive training steps
- Models requiring specific GPU memory

❌ **Not ideal for**:
- Small/fast models (network overhead)
- Real-time inference (latency sensitive)
- Models with frequent small operations

### Optimization Tips

1. **Batch Operations**: Combine multiple operations to reduce round trips
2. **Model Caching**: Cache compiled models on Modal to avoid recompilation
3. **Input Batching**: Process multiple inputs together
4. **Operation Fusion**: Let PyTorch fuse operations before remote execution

## Error Handling

The backend includes comprehensive error handling:

```python
def __call__(self, *args):
    try:
        # Try remote execution
        return self.execute_remote(*args)
    except Exception as e:
        logger.error(f"Remote execution failed: {e}")
        # Automatic fallback to local execution
        return self.fx_graph(*args)
```

## Testing

### Health Check

```bash
modal run modal_server.py::health_check
```

### Run Test Suite

```bash
python -m pytest test_modal_backend.py
```

### Manual Testing

```python
# Test simple operation
x = torch.randn(4, 128)
compiled_fn = torch.compile(torch.relu, backend="modal_backend")
result = compiled_fn(x)
```

## Troubleshooting

### Common Issues

1. **Modal Not Deployed**: Ensure `modal deploy modal_server.py` was run
2. **Network Errors**: Check internet connection and Modal status
3. **GPU Quota**: Verify Modal GPU quotas and usage
4. **Serialization Errors**: Some custom objects may not serialize properly

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Fallback Behavior

The backend automatically falls back to local execution on:
- Network connectivity issues
- Modal server errors  
- Serialization failures
- GPU unavailability

## Extension Points

### Adding Custom Operations

Extend `execute_simple_operations()` in `modal_server.py`:

```python
elif op_type == 'my_custom_op':
    result = my_custom_implementation(result, op['params'])
```

### Custom Model Types

Add model-specific optimizations:

```python
def execute_my_model_type(input_tensors, metadata, device):
    # Custom execution logic
    return outputs
```

### Backend Variants

Create specialized backends:

```python
@dynamo.register_backend  
def modal_fast_backend(fx_graph, example_inputs):
    # Optimized for speed
    pass

@dynamo.register_backend
def modal_memory_backend(fx_graph, example_inputs):  
    # Optimized for memory efficiency
    pass
```

## License

This project is provided as an example and educational resource. Please ensure compliance with Modal Labs terms of service and PyTorch licensing.

## Contributing

Contributions welcome! Areas for improvement:
- Better FX graph serialization/deserialization
- Support for more operation types
- Performance optimizations  
- Enhanced error handling
- Automatic GPU type selection

---

**Note**: This is an experimental implementation. For production use, consider additional optimizations around serialization, caching, and error recovery.