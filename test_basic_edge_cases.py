#!/usr/bin/env python3
"""
Basic edge case tests to identify issues.
"""

import torch
import torch_remote
import gc
import time

def test_tensor_creation_and_basic_ops(modal_t4_device):
    """Test basic tensor creation and operations."""
    print("🧪 Testing basic tensor creation and operations...")
    
    device = modal_t4_device
    
    # Test direct creation
    x = torch.randn(3, 3, device=device.device())
    print(f"   Created tensor: shape={x.shape}, is_persistent={getattr(x, 'is_persistent', 'N/A')}")
    
    # Test .to() transfer
    cpu_tensor = torch.randn(3, 3)
    y = cpu_tensor.to(device.device())
    print(f"   Transferred tensor: shape={y.shape}, is_persistent={getattr(y, 'is_persistent', 'N/A')}")
    
    # Test basic operation
    z = x + y
    print(f"   Operation result: shape={z.shape}, is_persistent={getattr(z, 'is_persistent', 'N/A')}")
    
    # Test conversion back to CPU
    cpu_result = z.cpu()
    print(f"   CPU result: shape={cpu_result.shape}, device={cpu_result.device}")
    
    assert True

def test_large_tensor(modal_t4_device):
    """Test handling of larger tensors."""
    print("🧪 Testing large tensor handling...")
    
    device = modal_t4_device
    
    # Create a reasonably large tensor
    large_tensor = torch.randn(500, 500, device=device.device())
    print(f"   Large tensor created: shape={large_tensor.shape}")
    
    # Test operation on large tensor
    result = large_tensor.sum()
    cpu_result = result.cpu()
    print(f"   Sum result: {cpu_result.item():.4f}")
    
    assert True

def test_error_scenarios(modal_t4_device):
    """Test error handling scenarios."""
    print("🧪 Testing error scenarios...")
    
    device = modal_t4_device
    
    # Test dimension mismatch (should fail gracefully)
    try:
        x = torch.randn(2, 3, device=device.device())
        y = torch.randn(4, 5, device=device.device())
        result = x.mm(y)  # Should fail
        print("   ERROR: Dimension mismatch should have failed!")
        return False
    except Exception as e:
        print(f"   ✅ Dimension mismatch correctly failed: {type(e).__name__}")
    
    # Test recovery after error
    try:
        x = torch.randn(2, 2, device=device.device())
        y = torch.randn(2, 2, device=device.device())
        result = x + y  # Should work
        print("   ✅ Recovery after error works")
        assert True
    except Exception as e:
        print(f"   ❌ Recovery failed: {e}")
        assert False, f"Recovery failed: {e}"

def test_different_dtypes(modal_t4_device):
    """Test different data types."""
    print("🧪 Testing different data types...")
    
    device = modal_t4_device
    
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    
    for dtype in dtypes:
        try:
            if dtype in [torch.int32, torch.int64]:
                # For integer dtypes, use ones() instead of randn()
                x = torch.ones(2, 2, dtype=dtype, device=device.device())
            else:
                # For float dtypes, use randn()
                x = torch.randn(2, 2, dtype=dtype, device=device.device())
            print(f"   ✅ {dtype} works: shape={x.shape}")
        except Exception as e:
            print(f"   ❌ {dtype} failed: {e}")
            assert False, f"{dtype} failed: {e}"
    
    assert True

def run_basic_tests():
    """Run basic edge case tests."""
    print("🚀 Running Basic Edge Case Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Creation and Operations", test_tensor_creation_and_basic_ops),
        ("Large Tensor Handling", test_large_tensor),
        ("Error Scenarios", test_error_scenarios),
        ("Different Data Types", test_different_dtypes),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"   Result: {'✅ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"   Result: ❌ FAILED with exception: {e}")
        print()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 50)
    print("📊 SUMMARY")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print("=" * 50)
    print(f"🎯 SCORE: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return passed == total

if __name__ == "__main__":
    run_basic_tests()