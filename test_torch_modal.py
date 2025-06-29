#!/usr/bin/env python3
"""
Comprehensive test suite for torch-modal package.

This script consolidates all testing functionality including:
- Import testing
- Device availability testing  
- Modal tensor creation and operations
- C extension functionality
- Error handling and edge cases
"""

import torch
import traceback
import sys


class TestResult:
    """Simple test result tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test(self, name, test_func):
        """Run a test and track results."""
        print(f"Testing {name}...", end=" ")
        try:
            result = test_func()
            if result:
                print("✓ PASS")
                self.passed += 1
            else:
                print("✗ FAIL")
                self.failed += 1
                self.errors.append(f"{name}: Test function returned False")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            self.failed += 1
            self.errors.append(f"{name}: {e}")
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        return self.failed == 0


def test_basic_imports():
    """Test basic torch and torch_modal imports."""
    try:
        import torch_modal
        return True
    except ImportError:
        return False


def test_c_extension_import():
    """Test C extension import."""
    try:
        import torch_modal._C
        return hasattr(torch_modal._C, 'modal')
    except ImportError:
        return False


def test_device_functions():
    """Test modal device functions."""
    try:
        import torch_modal
        return (torch_modal.modal.is_available() and 
                torch_modal.modal.device_count() >= 1 and
                torch_modal.modal.get_device_name() == "Modal Device")
    except Exception:
        return False


def test_tensor_modal_method():
    """Test that tensors have modal() method."""
    try:
        import torch_modal
        x = torch.randn(2, 2)
        return hasattr(x, 'modal') and callable(x.modal)
    except Exception:
        return False


def test_modal_tensor_creation():
    """Test modal tensor creation via .modal() method."""
    try:
        import torch_modal
        x = torch.randn(2, 2)
        y = x.modal()
        return y is not None and y.shape == x.shape
    except Exception:
        return False


def test_modal_tensor_operations():
    """Test operations on modal tensors."""
    try:
        import torch_modal
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        
        x_modal = x.modal()
        y_modal = y.modal()
        
        # Test addition
        z = x_modal + y_modal
        
        # Test matrix multiplication  
        w = x_modal.mm(y_modal)
        
        return z is not None and w is not None and w.shape == (2, 2)
    except Exception:
        return False


def test_c_extension_direct():
    """Test C extension modal function directly."""
    try:
        import torch_modal._C
        x = torch.randn(2, 2)
        y = torch_modal._C.modal(x, None, None, False, False)
        return y is not None and y.shape == x.shape
    except Exception:
        return False


def test_dtype_conversion():
    """Test modal conversion with dtype parameter."""
    try:
        import torch_modal
        x = torch.randn(2, 2, dtype=torch.float32)
        y = x.modal(dtype=torch.float64)
        return y.dtype == torch.float64
    except Exception:
        return False


def test_copy_parameter():
    """Test modal conversion with copy parameter."""
    try:
        import torch_modal
        x = torch.randn(2, 2)
        y = x.modal(copy=True)
        z = x.modal(copy=False)
        return y is not None and z is not None
    except Exception:
        return False


def test_error_handling():
    """Test that errors are handled gracefully."""
    try:
        import torch_modal
        # These operations might fail, but shouldn't crash
        try:
            torch.randn(3, 3, device='modal')  # Should fail gracefully
        except Exception:
            pass  # Expected to fail
        
        try:
            x = torch.randn(2, 2).modal()
            y = torch.randn(2, 2)  # CPU tensor
            z = x.mm(y)  # Mixed device - may or may not work
        except Exception:
            pass  # May fail, that's OK
            
        return True  # If we get here without segfault, it's good
    except Exception:
        return False


def main():
    """Run all tests."""
    print("PyTorch Modal Device Test Suite")
    print("=" * 50)
    
    results = TestResult()
    
    # Core functionality tests
    results.test("Basic imports", test_basic_imports)
    results.test("C extension import", test_c_extension_import)
    results.test("Device functions", test_device_functions)
    results.test("Tensor modal method", test_tensor_modal_method)
    results.test("Modal tensor creation", test_modal_tensor_creation)
    results.test("Modal tensor operations", test_modal_tensor_operations)
    results.test("C extension direct call", test_c_extension_direct)
    
    # Advanced tests
    results.test("Dtype conversion", test_dtype_conversion)
    results.test("Copy parameter", test_copy_parameter)
    results.test("Error handling", test_error_handling)
    
    # Print summary
    success = results.summary()
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {results.failed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())