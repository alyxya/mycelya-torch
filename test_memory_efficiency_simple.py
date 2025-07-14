#!/usr/bin/env python3
"""
Test memory efficiency improvements with meta tensors.
Simple version that focuses on tensor properties rather than system memory.
"""

import torch
import torch_remote

def test_meta_tensor_properties(modal_t4_device):
    """Test that remote tensors are using meta tensors underneath."""
    print("🧪 Testing Meta Tensor Properties")
    print("=" * 50)
    
    device = modal_t4_device
    
    # Test 1: Direct tensor creation
    print("1. Testing direct tensor creation...")
    large_tensor = torch.randn(1000, 1000, device=device.device())
    
    print(f"   Tensor type: {type(large_tensor).__name__}")
    print(f"   Reported device: {large_tensor.device}")
    print(f"   Underlying device: {large_tensor.data.device}")
    print(f"   Is persistent: {large_tensor.is_persistent}")
    print(f"   Tensor ID: {large_tensor.tensor_id[:8]}..." if large_tensor.tensor_id else "N/A")
    
    # Check if using meta tensors
    is_meta = large_tensor.data.device.type == 'meta'
    print(f"   ✅ Using meta tensors: {is_meta}")
    
    if is_meta:
        print(f"   ✅ EXCELLENT: Zero local memory overhead")
    else:
        print(f"   ⚠️  WARNING: Using {large_tensor.data.device.type} tensors locally")
    
    # Test 2: CPU-to-remote transfer
    print("\n2. Testing CPU-to-remote transfer...")
    cpu_tensor = torch.randn(500, 500)
    print(f"   Original CPU tensor size: {cpu_tensor.numel() * cpu_tensor.element_size()} bytes")
    
    remote_tensor = cpu_tensor.to(device.device())
    print(f"   Remote tensor type: {type(remote_tensor).__name__}")
    print(f"   Remote underlying device: {remote_tensor.data.device}")
    print(f"   Remote is persistent: {remote_tensor.is_persistent}")
    
    is_meta_transfer = remote_tensor.data.device.type == 'meta'
    print(f"   ✅ Transfer uses meta tensors: {is_meta_transfer}")
    
    # Test 3: Operations
    print("\n3. Testing operations...")
    x = torch.randn(100, 100, device=device.device())
    y = torch.randn(100, 100, device=device.device())
    z = x + y
    
    print(f"   Operation result underlying device: {z.data.device}")
    is_meta_ops = z.data.device.type == 'meta'
    print(f"   ✅ Operations use meta tensors: {is_meta_ops}")
    
    assert is_meta and is_meta_transfer and is_meta_ops

def test_data_integrity(modal_t4_device):
    """Test that data integrity is maintained despite using meta tensors."""
    print("\n🧪 Testing Data Integrity with Meta Tensors")
    print("=" * 50)
    
    device = modal_t4_device
    
    # Test 1: Round-trip data integrity
    print("1. Testing round-trip data integrity...")
    original_data = torch.randn(50, 50)
    print(f"   Original sum: {original_data.sum().item():.6f}")
    
    # Send to remote
    remote_tensor = original_data.to(device.device())
    print(f"   Remote tensor created (meta device: {remote_tensor.data.device.type == 'meta'})")
    
    # Retrieve from remote
    retrieved_data = remote_tensor.cpu()
    print(f"   Retrieved sum: {retrieved_data.sum().item():.6f}")
    
    # Check integrity
    matches = torch.allclose(original_data, retrieved_data, rtol=1e-4, atol=1e-6)
    print(f"   ✅ Data integrity preserved: {matches}")
    
    # Test 2: Operations integrity
    print("\n2. Testing operations integrity...")
    x_cpu = torch.randn(20, 20)
    y_cpu = torch.randn(20, 20)
    expected_result = x_cpu.mm(y_cpu)
    
    # Remote operations
    x_remote = x_cpu.to(device.device())
    y_remote = y_cpu.to(device.device())
    z_remote = x_remote.mm(y_remote)
    actual_result = z_remote.cpu()
    
    operations_match = torch.allclose(expected_result, actual_result, rtol=1e-4, atol=1e-6)
    print(f"   ✅ Operation results match: {operations_match}")
    
    assert matches and operations_match

def test_different_tensor_types(modal_t4_device):
    """Test meta tensors work with different dtypes and shapes."""
    print("\n🧪 Testing Different Tensor Types with Meta Tensors")
    print("=" * 50)
    
    device = modal_t4_device
    
    test_cases = [
        ("float32", torch.float32, lambda: torch.randn(10, 10)),
        ("float64", torch.float64, lambda: torch.randn(10, 10, dtype=torch.float64)),
        ("int32", torch.int32, lambda: torch.ones(10, 10, dtype=torch.int32)),
        ("int64", torch.int64, lambda: torch.ones(10, 10, dtype=torch.int64)),
        ("1D tensor", torch.float32, lambda: torch.randn(100)),
        ("3D tensor", torch.float32, lambda: torch.randn(5, 5, 5)),
        ("scalar", torch.float32, lambda: torch.tensor(5.0)),
    ]
    
    all_passed = True
    
    for name, dtype, tensor_factory in test_cases:
        try:
            print(f"   Testing {name}...")
            
            # Create tensor
            if name == "scalar":
                cpu_tensor = tensor_factory()
                remote_tensor = cpu_tensor.to(device.device())
            else:
                if name in ["int32", "int64"]:
                    # Use ones for integer types
                    remote_tensor = torch.ones(10, 10, dtype=dtype, device=device.device())
                else:
                    # Use randn for float types, handle different shapes
                    if "1D" in name:
                        remote_tensor = torch.randn(100, device=device.device())
                    elif "3D" in name:
                        remote_tensor = torch.randn(5, 5, 5, device=device.device())
                    else:
                        remote_tensor = torch.randn(10, 10, dtype=dtype, device=device.device())
            
            # Check properties
            is_meta = remote_tensor.data.device.type == 'meta'
            correct_dtype = remote_tensor.dtype == dtype
            
            print(f"     ✅ Meta tensor: {is_meta}, Correct dtype: {correct_dtype}")
            
            if not (is_meta and correct_dtype):
                all_passed = False
                
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            all_passed = False
    
    assert all_passed

def test_backward_compatibility(modal_t4_device):
    """Test that existing functionality still works."""
    print("\n🧪 Testing Backward Compatibility")
    print("=" * 50)
    
    try:
        device = modal_t4_device
        
        # Test from existing test suite
        print("1. Running subset of existing tests...")
        
        # Basic operations
        x = torch.randn(2, 2, device=device.device())
        y = torch.randn(2, 2, device=device.device()) 
        z = x.mm(y)
        result = z.cpu()
        
        print("   ✅ Matrix multiplication works")
        
        # Broadcasting
        broadcast_result = x + torch.randn(1, 2, device=device.device())
        print("   ✅ Broadcasting works")
        
        # Different dtypes
        int_tensor = torch.ones(2, 2, dtype=torch.int32, device=device.device())
        int_result = int_tensor.cpu()
        print("   ✅ Integer dtypes work")
        
        assert True
        
    except Exception as e:
        print(f"   ❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Compatibility test failed: {e}"

def run_memory_efficiency_tests():
    """Run all memory efficiency tests."""
    print("🚀 MEMORY EFFICIENCY TEST SUITE WITH META TENSORS")
    print("=" * 60)
    
    tests = [
        ("Meta Tensor Properties", test_meta_tensor_properties),
        ("Data Integrity", test_data_integrity),
        ("Different Tensor Types", test_different_tensor_types),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status} {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ FAILED {test_name}: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 60)
    print("📊 MEMORY EFFICIENCY TEST RESULTS")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print("=" * 60)
    print(f"🎯 SCORE: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All memory efficiency tests passed!")
        print("💡 Meta tensors successfully implemented - zero local memory overhead!")
    else:
        print("⚠️  Some tests failed. Check implementation.")
    
    return passed == total

if __name__ == "__main__":
    run_memory_efficiency_tests()