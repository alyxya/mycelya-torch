# Mycelya-Torch: PyTorch Remote Tensor Execution System

A production-ready PyTorch extension that enables transparent remote execution of tensor operations on cloud GPU infrastructure. The system uses a sequential tensor ID architecture with custom PyTorch integration for memory-efficient distributed computing.

## Architecture Overview

### Core Architectural Principles
- **Sequential Tensor ID Architecture**: Atomic counter generates unique storage IDs (1, 2, 3...) with metadata-based FNV-1a hash identification
- **Custom PyTorch Integration**: Complete custom TensorImpl, StorageImpl, and Allocator following pytorch-npu patterns with zero local memory overhead
- **Three-Layer Architecture**: C++ Backend, Python Coordination, Remote Execution with clean separation of concerns
- **Multi-GPU Cloud Support**: 10 GPU types supported (T4, L4, A10G, A100, L40S, H100, H200, B200) across cloud providers
- **Provider Abstraction**: Pluggable backend system with Modal (production), Mock (development), extensible for RunPod/Lambda Labs
- **RPC Batching**: Background thread processing reduces network overhead by ~10-100x with queue-based operation dispatch
- **HuggingFace Integration**: Direct remote model loading with parameter linking, bypassing data transfer entirely

### Production-Scale Features  
- **Thread-Safe Operations**: Background processing with proper synchronization and error handling
- **Comprehensive Test Coverage**: Complete test suite with 3-tier test strategy (critical/fast/comprehensive)
- **Enterprise Error Handling**: Clear error messages, graceful failure modes, and automatic resource cleanup
- **Memory Efficiency**: Zero local tensor data storage, raw bytes transfer, metadata caching with automatic invalidation
- **Performance Optimizations**: Meta tensor inference, view operation handling, dynamic output support

## Development Commands

To run tests:
```bash
# Minimal regression tests (every commit, <30 seconds)
pytest tests/test_regression.py::TestCriticalRegression -v

# Fast functional tests (PR reviews, ~2-5 minutes)  
pytest tests/test_regression.py -v

# Full comprehensive test suite
pytest tests/ -v
```

To run linting:
```bash
ruff check .
```

To run formatting:
```bash
ruff format .
```

To run type checking:
```bash
# No type checking tools currently configured
# Consider adding: mypy
```

## Build Configuration

### Modern Python Packaging
- **`pyproject.toml`**: Modern build system using setuptools with PyTorch C++ extensions
- **`setup.py`**: C++ extension compilation with platform-specific compiler flags
- **Dependencies**: torch>=2.0.0, modal>=1.1.0, numpy
- **License**: AGPL-3.0-or-later
- **Python Support**: 3.8+

### Code Quality Configuration
- **Ruff**: Line length 88, comprehensive rule selection (E, W, F, I, B, C4, UP)
- **Pytest**: Configuration in `pytest.ini` with custom markers
- **Build artifacts**: Compiled extensions in `build/` directory

## Codebase Structure

### Project Scale
- **Modular architecture** with Python and C++ components across comprehensive test coverage
- **Example applications** demonstrating HuggingFace integration and performance comparisons
- **Production-ready codebase** with enterprise-level code quality and documentation

### Core Python Modules (mycelya_torch/)
- `__init__.py` - Public API and PyTorch PrivateUse1 backend registration with tensor ID utilities
- `_orchestrator.py` - Central coordination with RPC batching, cache management, and background thread processing
- `_machine.py` - RemoteMachine abstraction with multi-provider support and context management  
- `_device.py` - DeviceManager for mapping local device indices to remote GPU configurations
- `_storage.py` - Sequential storage ID system with atomic counter and thread-safe generation (1, 2, 3...)
- `_backend_hooks.py` - PyTorch backend hooks and C++ interface bridge for transparent integration
- `_state_dict.py` - HuggingFace integration utilities for direct remote model loading
- `_utils.py` - Internal tensor utilities and metadata handling
- `_logging.py` - Hierarchical logging configuration with tensor hash IDs for debugging

### ATen Operation System (aten/)
Modular operation dispatch with clean separation of concerns:
- `__init__.py` - PyTorch library registrations for comprehensive ATen operation coverage
- `dispatch.py` - Main fallback kernel for unimplemented operations with meta tensor inference
- `copy.py` - Cross-device copy and transfer operations with raw bytes optimization
- `scalar.py` - Scalar operations with local execution optimization

### Provider Backend System (backends/)
- `base_client.py` - Abstract client interface with RPC batching, caching, and standardized provider API
- `modal/client.py` - Modal cloud provider implementation with multi-GPU support and connection management
- `mock/client.py` - Local execution provider using Modal's .local() for development and testing

### Remote Execution Infrastructure
- `_mycelya_torch_modal/modal_app.py` - Modal cloud GPU integration with dynamic app creation and lazy/realized storage

### C++ Backend Integration (csrc/)
Complete custom PyTorch integration following pytorch-npu patterns:
- `Mycelya.h` - Core header definitions, constants, and API declarations
- `MycelyaTensorImpl.cpp/.h` - Custom tensor implementation with FNV-1a metadata hash computation
- `MycelyaStorageImpl.cpp/.h` - Custom storage implementation with sequential ID tracking and zero local memory
- `MycelyaAllocator.cpp/.h` - Enhanced allocator with storage ID management and efficient allocation
- `MycelyaHooks.cpp` - PyTorch PrivateUse1 backend hooks with custom device management
- `MycelyaMem.cpp` - Memory management utilities and cross-platform compatibility
- `mycelya_extension.cpp` - Python bindings, C++ extensions, and API exposure

### Development Resources
- `examples/` - SmolLM2 inference, Modal integration testing, performance comparisons, HuggingFace loading
- `tests/` - Comprehensive test coverage with critical/fast/comprehensive markers
- Modern build system with `pyproject.toml`, `setup.py` for C++ extensions, and ruff configuration

## Current Architecture

### Key Design Principles
- **Sequential Tensor ID Architecture**: Atomic counter generates unique storage IDs (1, 2, 3...) with FNV-1a metadata hash computation for debugging
- **Custom PyTorch Integration**: Complete TensorImpl/StorageImpl/Allocator following pytorch-npu architecture patterns with zero local memory overhead
- **Clean Input/Output Separation**: Raw bytes transfer with numpy serialization, eliminating torch.save/load overhead (~2-5x faster)
- **Zero Local Memory**: No tensor data stored locally for remote tensors, only metadata maintained in custom implementations  
- **RPC Batching**: Background thread processing reduces network overhead by ~10-100x with queue-based operation dispatch
- **Multi-Provider Support**: Extensible backend system with Modal (production), Mock (development), ready for RunPod/Lambda Labs

### Memory Management Excellence
- **Zero local memory overhead**: Custom TensorImpl/StorageImpl store no tensor data locally, only metadata maintained
- **Sequential storage ID system**: Atomic counter generates unique IDs (1, 2, 3...) for efficient memory management across devices
- **Dual storage architecture**: Lazy allocation for meta operations, realized storage on remote GPUs for actual computation
- **Raw bytes transfer**: Direct numpy serialization bypasses torch.save/load overhead, achieving ~2-5x faster data transfer
- **Metadata-based caching**: Shape/stride/offset/dtype keys enable efficient caching without storing tensor data
- **FNV-1a hash computation**: On-demand metadata hash generation for debugging without memory allocations
- **Automatic cache invalidation**: Immediate invalidation at queue time ensures correctness with background batching

### Advanced Operation Dispatch Flow
1. **Meta Tensor Inference**: Shape computation using PyTorch's meta device eliminates data transfer for shape operations
2. **View Operation Optimization**: Local view creation with remote propagation maximizes memory efficiency  
3. **Dynamic Output Support**: Special handling for operations with data-dependent output shapes (e.g., nonzero, unique)
4. **RPC Batching Pipeline**: Operations queued in background thread, reducing network calls by ~10-100x
5. **Remote Execution**: All compute operations dispatched to cloud GPUs with proper error handling
6. **Thread-Safe Processing**: Background thread coordination with proper synchronization and error propagation
7. **Efficient Data Transfer**: Raw untyped storage bytes only when crossing device boundaries, no unnecessary serialization

## Usage Patterns

### Basic Usage
```python
import torch
import mycelya_torch

# Create remote machine with cloud GPU
machine = mycelya_torch.RemoteMachine("modal", "A100")

# Operations automatically execute on remote GPU with RPC batching
x = torch.randn(1000, 1000, device=machine.device())  # Storage ID: 1
y = torch.randn(1000, 1000, device=machine.device())  # Storage ID: 2
result = x @ y  # Matrix multiplication on remote A100, Storage ID: 3

# Each tensor has FNV-1a metadata hash for debugging
print(f"Result computed on {result.device}: {result.shape}")
```

### Production Neural Network Training
```python
import torch.nn as nn
import mycelya_torch

# Create remote machine with high-memory GPU
machine = mycelya_torch.RemoteMachine("modal", "A100")
device = machine.device()

# Model automatically uses sequential tensor IDs for all parameters
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(), 
    nn.Linear(512, 10)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Full training loop with remote gradient computation
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(dataloader):
        # Efficient data transfer with raw bytes optimization
        data, target = data.to(device), target.to(device)
        
        # All operations batched and executed remotely
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()  # Gradients computed on remote A100
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

### HuggingFace Model Integration
```python
import torch
import mycelya_torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create remote machine for inference
machine = mycelya_torch.RemoteMachine("modal", "H100")

# Load model architecture (no weights yet)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    torch_dtype=torch.float16,
    device_map=None  # Don't load weights yet
)

# Load weights directly on remote GPU - no data transfer
remote_state_dicts = mycelya_torch.load_huggingface_state_dicts(
    "microsoft/DialoGPT-medium", 
    machine.device()
)

# Load the remote weights into the model
model.load_state_dict(remote_state_dicts[""], strict=True)  # "" is root directory

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# All model parameters have sequential tensor IDs and reside remotely  
total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded with {total_params:,} parameters on {machine.device()}")

# Inference with automatic RPC batching
def generate_response(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(machine.device()) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Generation entirely on remote H100
        outputs = model.generate(
            **inputs, 
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Only final result transferred back
    response = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    return response

# Example usage
response = generate_response("Hello, how are you today?")
print(f"Generated response: {response}")
```

## Implementation Details

### Sequential Storage ID Architecture
- **Atomic Storage ID Generation**: Thread-safe counter producing unique incremental IDs (1, 2, 3, 4...)
- **FNV-1a Metadata Hash Computation**: Deterministic hash from shape/stride/dtype/offset/storage_id for debugging
- **Custom TensorImpl Integration**: MycelyaTensorImpl computes metadata hashes on-demand in C++
- **Zero Memory Overhead**: Hash generation without additional memory allocations or storage
- **Cross-Device Validation**: Storage IDs prevent operations between different remote machines
- **Internal API Access**: Metadata hash accessible via utility functions for debugging complex tensor flows
- **Process-Scoped Uniqueness**: Storage IDs unique within single Python process for memory efficiency

### Enterprise Error Handling
- **Cross-Device Operation Prevention**: Clear RuntimeError messages when mixing tensors from different machines
- **Type Safety Validation**: Descriptive errors for non-mycelya tensors accessing tensor ID APIs
- **Storage State Management**: Proper error handling for lazy vs realized storage access patterns
- **Connection Lifecycle Management**: Automatic reconnection with graceful degradation and retry logic
- **Background Thread Error Propagation**: Proper error handling in RPC batching with future-based error reporting
- **Provider-Specific Errors**: Modal connection failures, authentication errors, and resource exhaustion handling
- **Memory Management Errors**: Clear errors for storage allocation failures and cleanup issues

### Multi-Provider Architecture
- **Standardized Client Interface**: Abstract base client with consistent API across providers
- **Modal Production Backend**: Complete cloud provider with multi-GPU support, dynamic app creation, connection pooling
- **Mock Development Backend**: Local execution using Modal's .local() for testing without cloud dependencies  
- **Extensible Provider System**: Clean interfaces ready for RunPod, Lambda Labs, AWS/GCP integration
- **Provider-Agnostic Features**: RPC batching, caching, error handling work across all providers
- **Connection Management**: Proper initialization, cleanup, and resource management per provider
- **GPU Type Abstraction**: Unified interface for 10 different GPU types across cloud providers

## Documentation Maintenance

**IMPORTANT**: This file should be updated whenever making significant changes to the codebase.

### Update This File When:
- Adding new core modules or changing module responsibilities
- Modifying the sequential tensor ID architecture or metadata hash system  
- Adding/removing provider backends (RunPod, Lambda Labs, etc.) or GPU types
- Changing development commands (test, lint, typecheck) or build configuration
- Making breaking changes to the public API or internal architecture
- Major performance optimizations or C++ implementation changes
- Updates to RPC batching system or background thread processing
- Reorganizing ATen operation handling or modular dispatch system
- Changes to HuggingFace integration or direct model loading capabilities
- Updates to build system, development workflows, or testing strategies

## Development Guidelines

### Production Code Quality Standards

#### License Compliance (AGPL-3.0-or-later)
- **All source files must maintain AGPL license headers**
- New files require: `Copyright (C) 2025 alyxya, SPDX-License-Identifier: AGPL-3.0-or-later`
- Contributions must preserve open source licensing for derivative works

#### Sequential Tensor ID Architecture Rules
- **Never store tensor data locally** for remote tensors - only metadata in custom implementations
- **Sequential storage IDs** (1, 2, 3...) generated by atomic counter for memory efficiency
- **FNV-1a metadata hash computation** from tensor properties in MycelyaTensorImpl for debugging
- **Zero memory overhead** design - hash computed on-demand without additional allocations
- **Custom PyTorch integration** following pytorch-npu patterns with complete TensorImpl/StorageImpl/Allocator
- **Process-scoped uniqueness** for storage IDs enabling efficient cross-device validation

#### Multi-Provider Backend Implementation
- Follow Modal implementation pattern in `_mycelya_torch_modal/` for new providers
- Implement standardized client interface (base_client.py) with RPC batching support
- Support multi-GPU configuration with lazy/realized storage architecture
- Handle connection lifecycle, authentication, and background thread processing
- Integrate with hierarchical logging system using tensor hash IDs for debugging
- Provide Mock provider equivalent for local development and testing

#### Enterprise-Level Code Quality
- **Ruff linting/formatting**: Line length 88, comprehensive rule selection (E,W,F,I,B,C4,UP)
- **Google C++ Style**: All C++ files with consistent formatting and comprehensive documentation
- **3-Tier Testing Strategy**: Critical (<30s), Fast (~2-5min), Comprehensive (~10-30min)
- **Test Markers**: Use `@pytest.mark.critical`, `@pytest.mark.fast`, `@pytest.mark.slow` for categorization
- **Comprehensive Error Handling**: Clear RuntimeError messages, graceful failure modes, proper error propagation
- **Thread-Safe Operations**: Background processing with proper synchronization and future-based error handling
- **Modular Organization**: Clean separation between ATen operation handlers, provider backends, and core coordination

#### Performance and Memory Optimization
- **RPC Batching**: Background thread reduces network calls by ~10-100x with queue-based dispatch
- **Raw Bytes Transfer**: Direct numpy serialization eliminating torch.save/load overhead (~2-5x faster)
- **Meta Tensor Integration**: Shape computation without data transfer using PyTorch's meta device
- **View Operation Optimization**: Local view creation with remote propagation for memory efficiency
- **Automatic Cache Invalidation**: Immediate invalidation at queue time ensuring correctness