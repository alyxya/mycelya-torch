# torch-remote Troubleshooting Guide

This guide covers common issues you might encounter when using torch-remote and provides step-by-step solutions.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Device and Connection Issues](#device-and-connection-issues)
3. [Usage Errors](#usage-errors)
4. [Performance Issues](#performance-issues)
5. [Debugging Techniques](#debugging-techniques)
6. [Environment Validation](#environment-validation)

---

## Installation Issues

### C++ Extension Compilation Problems

#### **Problem**: Compilation fails with compiler errors
```
error: Microsoft Visual C++ 14.0 is required
error: unable to execute 'gcc': No such file or directory
```

**Solution**:
1. **Windows**:
   ```bash
   # Install Visual Studio Build Tools
   # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```

2. **Linux/macOS**:
   ```bash
   # Install build essentials
   sudo apt-get install build-essential  # Ubuntu/Debian
   xcode-select --install                 # macOS
   ```

3. **Verify PyTorch installation**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

4. **Clean and reinstall**:
   ```bash
   pip uninstall torch-remote
   python setup.py clean
   pip install .
   ```

#### **Problem**: PyTorch version compatibility errors
```
RuntimeError: The detected CUDA version mismatches the version that was used to compile PyTorch
```

**Solution**:
1. Check PyTorch CUDA compatibility:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
   ```

2. Install matching PyTorch version:
   ```bash
   # For CUDA 11.8
   pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
   
   # For CPU only
   pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
   ```

### Modal Dependency Issues

#### **Problem**: Modal import fails
```
ModuleNotFoundError: No module named 'modal'
ImportError: Failed to import modal
```

**Solution**:
1. Install Modal:
   ```bash
   pip install modal>=0.60.0
   ```

2. Verify Modal installation:
   ```bash
   python -c "import modal; print(modal.__version__)"
   ```

3. If still failing, check for conflicts:
   ```bash
   pip list | grep modal
   pip uninstall modal
   pip install modal>=0.60.0
   ```

---

## Device and Connection Issues

### Modal Authentication Problems

#### **Problem**: Authentication failures
```
modal.exception.AuthError: Authentication failed
modal.exception.NotAuthenticatedError: Not authenticated
```

**Solution**:
1. **Set up Modal authentication**:
   ```bash
   modal setup
   ```
   Follow the prompts to authenticate with your Modal account.

2. **Verify authentication**:
   ```bash
   modal token current
   ```

3. **Re-authenticate if needed**:
   ```bash
   modal token logout
   modal setup
   ```

4. **Check token environment**:
   ```bash
   echo $MODAL_TOKEN_ID
   echo $MODAL_TOKEN_SECRET
   ```

**Prevention**: Always run `modal setup` before first use.

### Device Creation Failures

#### **Problem**: Invalid GPU type specified
```
ValueError: Invalid GPU type 'RTX4090'. Valid types: ['T4', 'L4', 'A10G', 'A100-40GB', 'A100-80GB', 'L40S', 'H100', 'H200', 'B200']
```

**Solution**:
1. **Use supported GPU types**:
   ```python
   import torch_remote
   
   # Check available GPU types
   for gpu_type in torch_remote.GPUType:
       print(gpu_type.value)
   
   # Create device with valid GPU type
   device = torch_remote.create_modal_machine("A100-40GB")
   ```

2. **Case-sensitive GPU types**: Ensure exact spelling and case.

#### **Problem**: Device initialization timeout
```
TimeoutError: Device initialization timed out after 300 seconds
```

**Solution**:
1. **Check Modal service status**:
   ```bash
   modal status
   ```

2. **Try smaller GPU type** (T4/L4 start faster than A100/H100):
   ```python
   device = torch_remote.create_modal_machine("T4")  # Faster startup
   ```

3. **Increase timeout** by modifying GPU config if needed.

**Prevention**: Start with smaller GPU types for testing.

### Remote Execution Timeouts

#### **Problem**: Operations timeout on remote GPU
```
TimeoutError: Remote operation timed out after 450 seconds
```

**Solution**:
1. **Check operation complexity**:
   - Break large operations into smaller chunks
   - Consider if the operation is appropriate for remote execution

2. **Monitor Modal app logs**:
   ```bash
   modal logs <app-name>
   ```

3. **Verify GPU availability**:
   ```python
   # Check if operation is being sent to remote GPU
   import logging
   logging.basicConfig(level=logging.INFO)
   
   # Look for "🚀 Creating remote job" messages
   result = tensor1 + tensor2
   ```

4. **Use simpler operations** to test connectivity:
   ```python
   import torch
   device = torch_remote.create_modal_machine("T4")
   x = torch.randn(10, 10, device=device)
   y = x + 1  # Simple operation
   ```

### Network Connectivity Issues

#### **Problem**: Connection refused or network errors
```
ConnectionError: Failed to connect to Modal service
requests.exceptions.ConnectionError: HTTPSConnectionPool
```

**Solution**:
1. **Check internet connection**:
   ```bash
   ping modal.com
   curl -I https://api.modal.com
   ```

2. **Check firewall/proxy settings**:
   - Ensure ports 443 (HTTPS) is open
   - Configure proxy if needed:
     ```bash
     export HTTPS_PROXY=your-proxy-url
     export HTTP_PROXY=your-proxy-url
     ```

3. **Verify DNS resolution**:
   ```bash
   nslookup api.modal.com
   ```

4. **Try different network** (mobile hotspot, etc.) to isolate network issues.

---

## Usage Errors

### Mixed Device Operations

#### **Problem**: Operations between different remote devices
```
RuntimeError: Cannot perform operations between tensors on different remote devices: 'modal-t4-abc123' and 'modal-a100-def456'
```

**Solution**:
This is **by design** - each remote device represents a separate GPU instance.

1. **Use same device for all tensors**:
   ```python
   device = torch_remote.create_modal_machine("A100-40GB")
   x = torch.randn(10, 10, device=device)
   y = torch.randn(10, 10, device=device)
   z = x + y  # ✅ Works - same device
   ```

2. **Transfer tensors if needed**:
   ```python
   device1 = torch_remote.create_modal_machine("T4")
   device2 = torch_remote.create_modal_machine("L4")
   
   x = torch.randn(10, 10, device=device1)
   y = torch.randn(10, 10, device=device2)
   
   # Transfer to same device
   y_on_device1 = y.to(device1)
   z = x + y_on_device1  # ✅ Works
   ```

**Prevention**: Plan your device usage - keep related tensors on the same device.

### Tensor Conversion Issues

#### **Problem**: String-based device specification
```
ValueError: Remote devices must be RemoteMachine objects. Use create_modal_machine() or similar to create a RemoteMachine.
```

**Solution**:
Don't use string device names for remote devices:

```python
# ❌ Wrong
x = torch.randn(10, 10, device="remote")
x = torch.randn(10, 10, device="remote:0")

# ✅ Correct
device = torch_remote.create_modal_machine("A100-40GB")
x = torch.randn(10, 10, device=device)
```

#### **Problem**: Device ID not preserved
```
RuntimeError: Device ID must be explicitly specified for remote tensor creation
```

**Solution**:
1. **Check tensor has device ID**:
   ```python
   tensor = torch.randn(10, 10, device=device)
   print(f"Device ID: {getattr(tensor, '_device_id', 'NOT SET')}")
   ```

2. **Recreate tensor if device ID missing**:
   ```python
   if not hasattr(tensor, '_device_id'):
       tensor = tensor.to(device)  # This should set _device_id
   ```

### Operation Dispatch Problems

#### **Problem**: Operation not using remote execution
```
# Operation runs locally instead of on remote GPU
```

**Solution**:
1. **Check if operation is in skip list**:
   Operations like `.cpu()`, `.to()`, `.view()`, `.size()` run locally by design.

2. **Use compute-intensive operations** to trigger remote execution:
   ```python
   # These will use remote execution:
   z = torch.mm(x, y)        # Matrix multiplication
   z = torch.conv2d(x, w)    # Convolution
   z = torch.relu(x)         # Activation functions
   z = x + y                 # Element-wise operations on large tensors
   ```

3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   
   # Look for these messages:
   # "🚀 Creating remote job for aten.add.Tensor"
   # "✅ Remote operation completed"
   ```

### Memory-Related Errors

#### **Problem**: Out of memory on remote GPU
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Solution**:
1. **Use smaller tensors**:
   ```python
   # Reduce tensor size
   x = torch.randn(1000, 1000, device=device)  # Instead of (10000, 10000)
   ```

2. **Choose appropriate GPU type**:
   ```python
   # For large models, use high-memory GPUs
   device = torch_remote.create_modal_machine("A100-80GB")  # 80GB memory
   device = torch_remote.create_modal_machine("H100")       # 80GB memory
   ```

3. **Process in batches**:
   ```python
   # Process large tensors in chunks
   batch_size = 1000
   for i in range(0, len(data), batch_size):
       batch = data[i:i+batch_size].to(device)
       result = model(batch)
   ```

---

## Performance Issues

### Slow Remote Execution

#### **Problem**: Operations are unexpectedly slow

**Diagnosis**:
1. **Check if operation is running remotely**:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   
   # Should see: "🚀 Creating remote job for aten.xxx"
   result = tensor1 + tensor2
   ```

2. **Measure operation time**:
   ```python
   import time
   start = time.time()
   result = torch.mm(x, y)
   end = time.time()
   print(f"Operation took {end - start:.2f} seconds")
   ```

**Solutions**:
1. **Use larger tensors** - small operations have high overhead:
   ```python
   # Small tensors may run locally (faster)
   x = torch.randn(10, 10, device=device)
   
   # Large tensors will run remotely
   x = torch.randn(1000, 1000, device=device)
   ```

2. **Batch operations**:
   ```python
   # Instead of many small operations
   for i in range(100):
       result = x + 1
   
   # Do one larger operation
   result = x + 100
   ```

3. **Choose appropriate GPU**:
   ```python
   # For compute-heavy workloads
   device = torch_remote.create_modal_machine("H100")  # Fastest
   device = torch_remote.create_modal_machine("A100-40GB")  # Good performance
   
   # For testing/light workloads
   device = torch_remote.create_modal_machine("T4")  # Cheaper, adequate
   ```

### Container Startup Delays

#### **Problem**: First operation takes a long time

This is **normal behavior** - Modal containers have cold start time.

**Solutions**:
1. **Use smaller GPU types** for faster startup:
   ```python
   device = torch_remote.create_modal_machine("T4")  # ~30-60 seconds
   device = torch_remote.create_modal_machine("L4")  # ~30-60 seconds
   # vs
   device = torch_remote.create_modal_machine("H100")  # ~2-5 minutes
   ```

2. **Keep containers warm** by doing periodic operations:
   ```python
   # Do a small operation every few minutes to keep container alive
   keepalive = torch.randn(2, 2, device=device)
   result = keepalive + 1
   ```

3. **Plan for cold starts** in your workflow timing.

### GPU Selection Optimization

#### **Problem**: Choosing the right GPU type

**Guidelines**:

| GPU Type | Memory | Best For | Startup Time |
|----------|--------|----------|--------------|
| T4 | 16GB | Testing, small models | Fast (~30s) |
| L4 | 24GB | Medium models, inference | Fast (~30s) |
| A10G | 24GB | Training, medium workloads | Medium (~1-2min) |
| A100-40GB | 40GB | Large models, training | Medium (~1-2min) |
| A100-80GB | 80GB | Very large models | Medium (~1-2min) |
| H100 | 80GB | Cutting-edge performance | Slow (~3-5min) |

**Selection logic**:
```python
def choose_gpu(model_size_gb, workload_type):
    if model_size_gb < 10:
        return "T4" if workload_type == "inference" else "L4"
    elif model_size_gb < 30:
        return "A100-40GB"
    else:
        return "A100-80GB"

# Example usage
gpu_type = choose_gpu(model_size_gb=15, workload_type="training")
device = torch_remote.create_modal_machine(gpu_type)
```

---

## Debugging Techniques

### Enable Verbose Logging

**Setup comprehensive logging**:
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable torch_remote specific logging
torch_remote_logger = logging.getLogger('torch_remote')
torch_remote_logger.setLevel(logging.DEBUG)

# Enable Modal logging
modal_logger = logging.getLogger('modal')
modal_logger.setLevel(logging.INFO)
```

**Look for these key messages**:
- `🚀 Creating remote job for aten.xxx` - Remote execution starting
- `✅ Remote operation completed` - Remote execution finished
- `❌ Remote operation failed` - Remote execution error
- `Modal T4 executing: aten.xxx` - Operation running on GPU

### Diagnostic Commands

**1. Check device registration**:
```python
import torch_remote

registry = torch_remote.get_device_registry()
print(f"Registered devices: {len(registry._devices)}")
for device_id, device in registry._devices.items():
    print(f"  {device_id}: {device}")
```

**2. Verify tensor device information**:
```python
tensor = torch.randn(10, 10, device=device)
print(f"Device: {tensor.device}")
print(f"Device type: {tensor.device.type}")
print(f"Device index: {tensor.device.index}")
print(f"Device ID: {getattr(tensor, '_device_id', 'NOT SET')}")
print(f"Tensor class: {tensor.__class__}")
```

**3. Test basic operations**:
```python
def test_device(device):
    print(f"Testing device: {device}")
    try:
        # Create tensors
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        
        # Test basic operation
        z = x + y
        print(f"  ✅ Addition: {x.shape} + {y.shape} = {z.shape}")
        
        # Test matrix multiplication
        mm = torch.mm(x, y)
        print(f"  ✅ MatMul: {x.shape} @ {y.shape} = {mm.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

# Test all your devices
devices = [
    torch_remote.create_modal_machine("T4"),
    torch_remote.create_modal_machine("L4"),
]

for device in devices:
    test_device(device)
```

### Modal-Specific Debugging

**1. Check Modal logs**:
```bash
# List Modal apps
modal app list

# View logs for specific app
modal logs torch-remote-<device-id>

# Follow logs in real-time
modal logs torch-remote-<device-id> --follow
```

**2. Check Modal status**:
```bash
# Check Modal service status
modal status

# Check authentication
modal token current

# List running apps
modal app list --running
```

---

## Environment Validation

### Validate Complete Setup

**Run this comprehensive check**:
```python
def validate_torch_remote_setup():
    """Comprehensive validation of torch-remote setup."""
    
    print("🔍 Validating torch-remote setup...")
    
    # 1. Check basic imports
    try:
        import torch
        import torch_remote
        import modal
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # 2. Check versions
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Modal: {modal.__version__}")
    
    # 3. Check Modal authentication
    try:
        # This will raise if not authenticated
        modal.App("test-auth")
        print("✅ Modal authentication working")
    except Exception as e:
        print(f"❌ Modal authentication failed: {e}")
        return False
    
    # 4. Check device creation
    try:
        device = torch_remote.create_modal_machine("T4")
        print(f"✅ Device creation: {device}")
    except Exception as e:
        print(f"❌ Device creation failed: {e}")
        return False
    
    # 5. Check tensor creation
    try:
        tensor = torch.randn(5, 5, device=device)
        print(f"✅ Tensor creation: {tensor.shape} on {tensor.device}")
    except Exception as e:
        print(f"❌ Tensor creation failed: {e}")
        return False
    
    # 6. Check basic operation
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        z = x + y  # Should see remote execution logs
        print(f"✅ Basic operation: {z.shape}")
    except Exception as e:
        print(f"❌ Basic operation failed: {e}")
        return False
    
    print("🎉 All checks passed! torch-remote is working correctly.")
    return True

# Run validation
validate_torch_remote_setup()
```

### Common Configuration Checks

**1. Python environment**:
```bash
python --version  # Should be 3.8+
pip list | grep torch
pip list | grep modal
```

**2. Modal setup**:
```bash
modal --version
modal token current
modal status
```

**3. System resources**:
```bash
# Check available memory
free -h

# Check Python path
which python
echo $PYTHONPATH
```

---

## Getting Help

If you're still experiencing issues after trying these solutions:

1. **Check for known issues** in the GitHub repository
2. **Enable debug logging** and collect logs:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG, filename='torch_remote_debug.log')
   ```

3. **Prepare minimal reproduction case**:
   ```python
   import torch
   import torch_remote
   
   device = torch_remote.create_modal_machine("T4")
   x = torch.randn(10, 10, device=device)
   y = x + 1  # Your failing operation here
   ```

4. **Include environment information**:
   - Operating system and version
   - Python version
   - PyTorch version
   - Modal version
   - torch-remote version
   - Complete error traceback

5. **Check Modal status**: https://status.modal.com for service disruptions

Remember: torch-remote is designed for compute-intensive operations on large tensors. Small operations may run locally for performance reasons, which is expected behavior.