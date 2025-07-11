# torch-remote API Reference

## Table of Contents

- [Overview](#overview)
- [Core Device Functions](#core-device-functions)
  - [create_modal_device()](#create_modal_device)
  - [get_device_registry()](#get_device_registry)
- [Classes](#classes)
  - [BackendDevice](#backenddevice)
  - [DeviceRegistry](#deviceregistry)
- [Enums](#enums)
  - [GPUType](#gputype)
  - [BackendProvider](#backendprovider)
- [Tensor Operations](#tensor-operations)
  - [Factory Functions](#factory-functions)
  - [Tensor Methods](#tensor-methods)
- [Remote Device Module](#remote-device-module)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

The torch-remote library provides seamless remote GPU execution for PyTorch tensors through cloud providers. It extends PyTorch's device system to support remote GPU instances with automatic workload dispatch.

## Core Device Functions

### `create_modal_device()`

Creates a Modal backend device with the specified GPU type.

```python
def create_modal_device(gpu: Union[str, GPUType], **kwargs) -> BackendDevice
```

**Parameters:**
- `gpu` (Union[str, GPUType]): GPU type specification
  - String values: "T4", "L4", "A10G", "A100-40GB", "A100-80GB", "L40S", "H100", "H200", "B200"
  - GPUType enum values: `GPUType.T4`, `GPUType.L4`, etc.
- `**kwargs`: Additional Modal-specific configuration options

**Returns:**
- `BackendDevice`: Configured device instance for the specified GPU type

**Raises:**
- `ValueError`: If GPU type is invalid or not supported

**Example:**
```python
import torch_remote

# Create devices using string specification
t4_device = torch_remote.create_modal_device("T4")
a100_device = torch_remote.create_modal_device("A100-40GB")

# Create device using enum
from torch_remote import GPUType
h100_device = torch_remote.create_modal_device(GPUType.H100)

# Create tensor on remote device
tensor = torch.randn(3, 3, device=t4_device)
```

### `get_device_registry()`

Returns the global device registry instance.

```python
def get_device_registry() -> DeviceRegistry
```

**Returns:**
- `DeviceRegistry`: The global device registry managing all active devices

**Example:**
```python
import torch_remote

registry = torch_remote.get_device_registry()
print(f"Active devices: {len(registry._devices)}")
```

## Classes

### `BackendDevice`

Represents a remote GPU device with specific provider and GPU type.

```python
class BackendDevice:
    def __init__(self, provider: BackendProvider, gpu_type: GPUType, **kwargs)
```

**Constructor Parameters:**
- `provider` (BackendProvider): Cloud provider (currently only MODAL)
- `gpu_type` (GPUType): GPU type specification
- `**kwargs`: Provider-specific configuration

**Properties:**

#### `device_id`
```python
@property
def device_id(self) -> str
```
Unique identifier for the device instance.

#### `device_name`
```python
@property
def device_name(self) -> str
```
Human-readable device name (e.g., "Modal A100-40GB").

#### `modal_gpu_spec`
```python
@property
def modal_gpu_spec(self) -> str
```
Modal GPU specification string. Only available for Modal provider.

**Raises:**
- `ValueError`: If called on non-Modal device

#### `remote_index`
```python
@property
def remote_index(self) -> Optional[int]
```
Device's index in the device registry.

**Methods:**

#### `__str__()` and `__repr__()`
```python
def __str__(self) -> str
def __repr__(self) -> str
```
String representation of the device.

#### `__eq__()` and `__hash__()`
```python
def __eq__(self, other) -> bool
def __hash__(self) -> int
```
Equality comparison and hashing based on device_id.

**Example:**
```python
import torch_remote
from torch_remote import GPUType, BackendProvider

# Create device
device = torch_remote.create_modal_device("A100-40GB")

print(device.device_id)        # "modal-a10040gb-abc12345"
print(device.device_name)      # "Modal A100-40GB"
print(device.modal_gpu_spec)   # "A100-40GB"
print(device.remote_index)     # 0
```

### `DeviceRegistry`

Registry to manage active BackendDevice instances and ensure tensor compatibility.

```python
class DeviceRegistry:
    def __init__(self)
```

**Methods:**

#### `register_device()`
```python
def register_device(self, device: BackendDevice) -> int
```
Register a device and return its assigned index.

**Parameters:**
- `device` (BackendDevice): Device to register

**Returns:**
- `int`: Assigned device index

#### `get_device_by_index()`
```python
def get_device_by_index(self, index: int) -> Optional[BackendDevice]
```
Retrieve device by its registry index.

**Parameters:**
- `index` (int): Device index

**Returns:**
- `Optional[BackendDevice]`: Device instance or None if not found

#### `get_device_by_id()`
```python
def get_device_by_id(self, device_id: str) -> Optional[BackendDevice]
```
Retrieve device by its unique ID.

**Parameters:**
- `device_id` (str): Device identifier

**Returns:**
- `Optional[BackendDevice]`: Device instance or None if not found

#### `get_device_index()`
```python
def get_device_index(self, device: BackendDevice) -> Optional[int]
```
Get the registry index of a device.

**Parameters:**
- `device` (BackendDevice): Device instance

**Returns:**
- `Optional[int]`: Device index or None if not registered

#### `devices_compatible()`
```python
def devices_compatible(self, device1: BackendDevice, device2: BackendDevice) -> bool
```
Check if two devices are compatible for tensor operations.

**Parameters:**
- `device1` (BackendDevice): First device
- `device2` (BackendDevice): Second device

**Returns:**
- `bool`: True if devices are compatible (same device_id)

#### `clear()`
```python
def clear(self) -> None
```
Clear all registered devices from the registry.

**Example:**
```python
import torch_remote

registry = torch_remote.get_device_registry()
device = torch_remote.create_modal_device("T4")

# Device is automatically registered
index = registry.get_device_index(device)
retrieved = registry.get_device_by_index(index)
assert retrieved == device
```

## Enums

### `GPUType`

Enumeration of supported GPU types across cloud providers.

```python
class GPUType(Enum):
    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    A100_40GB = "A100-40GB"
    A100_80GB = "A100-80GB"
    L40S = "L40S"
    H100 = "H100"
    H200 = "H200"
    B200 = "B200"
```

**GPU Type Descriptions:**
- **T4**: Entry-level GPU, good for inference and light training
- **L4**: Mid-range GPU with better performance than T4
- **A10G**: Professional GPU for AI workloads
- **A100-40GB**: High-end GPU with 40GB memory
- **A100-80GB**: High-end GPU with 80GB memory  
- **L40S**: Latest generation professional GPU
- **H100**: Next-generation high-performance GPU
- **H200**: H100 variant with increased memory
- **B200**: Latest Blackwell architecture GPU

**Example:**
```python
from torch_remote import GPUType

# Iterate through all GPU types
for gpu_type in GPUType:
    print(f"GPU: {gpu_type.value}")
    device = torch_remote.create_modal_device(gpu_type)
```

### `BackendProvider`

Enumeration of supported cloud providers.

```python
class BackendProvider(Enum):
    MODAL = "modal"
    # Future providers:
    # RUNPOD = "runpod"
    # LAMBDA = "lambda"
```

Currently only Modal is supported, with additional providers planned for future releases.

## Tensor Operations

### Factory Functions

PyTorch factory functions are patched to support BackendDevice instances:

#### `torch.randn()`
```python
torch.randn(*size, device=None, **kwargs) -> Tensor
```

#### `torch.zeros()`
```python
torch.zeros(*size, device=None, **kwargs) -> Tensor
```

#### `torch.ones()`
```python
torch.ones(*size, device=None, **kwargs) -> Tensor
```

#### `torch.empty()`
```python
torch.empty(*size, device=None, **kwargs) -> Tensor
```

#### `torch.tensor()`
```python
torch.tensor(data, device=None, **kwargs) -> Tensor
```

**Parameters:**
- `device`: Can be a `BackendDevice` instance for remote execution
- Other parameters follow standard PyTorch conventions

**Example:**
```python
import torch
import torch_remote

device = torch_remote.create_modal_device("A100-40GB")

# Create tensors directly on remote device
x = torch.randn(100, 100, device=device)
y = torch.zeros(50, 50, device=device)
z = torch.ones(10, 10, device=device)
```

### Tensor Methods

#### `.to()`
The tensor `.to()` method is enhanced to support BackendDevice instances.

```python
tensor.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=None) -> Tensor
```

**Parameters:**
- `device`: Can be a `BackendDevice` instance
- Other parameters follow standard PyTorch conventions

**Example:**
```python
import torch
import torch_remote

# Create tensor on CPU
x = torch.randn(5, 5)

# Move to remote device
device = torch_remote.create_modal_device("T4")
x_remote = x.to(device)

# Move back to CPU
x_cpu = x_remote.cpu()
```

## Remote Device Module

The torch-remote library registers itself as a PyTorch device backend with the name "remote".

### Device Functions

#### `torch.remote.device_count()`
```python
torch.remote.device_count() -> int
```
Returns the number of available remote devices.

#### `torch.remote.current_device()`
```python
torch.remote.current_device() -> int
```
Returns the index of the current remote device.

#### `torch.remote.is_available()`
```python
torch.remote.is_available() -> bool
```
Returns True if remote execution is available.

#### `torch.remote.device()`
Context manager for device selection:

```python
with torch.remote.device(device_index):
    # Operations use the specified remote device
    pass
```

### Random Number Generation

#### `torch.remote.manual_seed()`
```python
torch.remote.manual_seed(seed: int) -> None
```
Set random seed for current remote device.

#### `torch.remote.manual_seed_all()`
```python
torch.remote.manual_seed_all(seed: int) -> None
```
Set random seed for all remote devices.

#### `torch.remote.get_rng_state()`
```python
torch.remote.get_rng_state(device="remote") -> Tensor
```
Get random number generator state.

#### `torch.remote.set_rng_state()`
```python
torch.remote.set_rng_state(new_state, device="remote") -> None
```
Set random number generator state.

## Error Handling

### Common Exceptions

The library uses standard PyTorch exceptions and adds specific validation:

#### `ValueError`
Raised for invalid GPU types or provider configurations:
```python
# Invalid GPU type
torch_remote.create_modal_device("InvalidGPU")  # Raises ValueError

# Using raw remote device strings (not allowed)
torch.randn(3, 3, device="remote")  # Raises ValueError
```

#### `RuntimeError`
Standard PyTorch runtime errors for tensor operations:
```python
device1 = torch_remote.create_modal_device("T4")
device2 = torch_remote.create_modal_device("A100-40GB")

x = torch.randn(3, 3, device=device1)
y = torch.randn(3, 3, device=device2)

# Cross-device operations may raise RuntimeError
z = x + y  # May raise RuntimeError
```

### Error Messages

The library provides clear error messages for common issues:

- **Invalid GPU type**: Lists all valid GPU types
- **Cross-device operations**: Explains device compatibility requirements
- **Provider limitations**: Indicates unsupported provider features

## Examples

### Basic Usage

```python
import torch
import torch_remote

# Create a device
device = torch_remote.create_modal_device("A100-40GB")

# Create tensors
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Operations execute remotely
z = x @ y  # Matrix multiplication on remote GPU

# Move result back to CPU
result = z.cpu()
```

### Multiple Devices

```python
import torch
import torch_remote

# Create different devices
t4_device = torch_remote.create_modal_device("T4")
a100_device = torch_remote.create_modal_device("A100-40GB")

# Create tensors on different devices
x_t4 = torch.randn(100, 100, device=t4_device)
x_a100 = torch.randn(200, 200, device=a100_device)

# Operations within same device work
y_t4 = x_t4 + x_t4     # Works - same device
y_a100 = x_a100 * 2    # Works - same device

# Cross-device operations fail
# z = x_t4 + x_a100    # Would raise RuntimeError
```

### Device Registry Management

```python
import torch_remote

# Create devices
device1 = torch_remote.create_modal_device("T4")
device2 = torch_remote.create_modal_device("L4")

# Check registry
registry = torch_remote.get_device_registry()
print(f"Registered devices: {len(registry._devices)}")

# Device lookup
index1 = registry.get_device_index(device1)
retrieved = registry.get_device_by_index(index1)
assert retrieved == device1

# Compatibility check
compatible = registry.devices_compatible(device1, device2)
print(f"Devices compatible: {compatible}")  # False
```

### Advanced Usage with Context Manager

```python
import torch
import torch_remote

device = torch_remote.create_modal_device("H100")
x = torch.randn(5, 5, device=device)

# Use remote device context
with torch.remote.device(device.remote_index):
    # Random operations use this device
    y = torch.randn(5, 5)  # Created on remote device
    z = x + y              # Both on same remote device
```

---

This API reference provides comprehensive documentation for all public interfaces in the torch-remote library. For additional examples and tutorials, see the main README.md file.