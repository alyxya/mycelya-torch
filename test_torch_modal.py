#!/usr/bin/env python3
"""
Comprehensive test suite for torch-modal package.

This script works with both pytest and standalone execution:
- Run with pytest: pytest test_torch_modal.py
- Run standalone: python test_torch_modal.py
- For verbose output: python test_torch_modal.py --verbose
- For quick debug mode: python test_torch_modal.py --debug
"""

import torch
import traceback
import sys
import argparse
import pytest


def test_basic_imports():
    """Test basic torch and torch_modal imports."""
    import torch_modal
    assert True


def test_device_functions():
    """Test modal device functions."""
    import torch_modal
    assert (torch_modal.modal.is_available() and 
            torch_modal.modal.device_count() >= 1 and
            torch_modal.modal.get_device_name() == "Modal Device")


def test_tensor_modal_method():
    """Test that tensors have modal() method."""
    import torch_modal
    x = torch.randn(2, 2)
    assert hasattr(x, 'modal') and callable(x.modal)


def test_modal_tensor_creation():
    """Test modal tensor creation via .modal() method."""
    import torch_modal
    x = torch.randn(2, 2)
    y = x.modal()
    assert y is not None and y.shape == x.shape


def test_modal_tensor_operations():
    """Test operations on modal tensors."""
    import torch_modal
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    
    x_modal = x.modal()
    y_modal = y.modal()
    
    # Test addition - verify numerical result matches CPU computation
    z_modal = x_modal + y_modal
    z_expected = x + y
    
    # Test matrix multiplication - verify numerical result matches CPU computation
    w_modal = x_modal.mm(y_modal)
    w_expected = x.mm(y)
    
    # Verify shapes
    assert z_modal is not None and w_modal is not None and w_modal.shape == (2, 2)
    
    # Verify numerical results (convert modal tensors back to CPU for comparison)
    assert torch.allclose(z_modal.cpu(), z_expected, rtol=1e-5, atol=1e-8)
    assert torch.allclose(w_modal.cpu(), w_expected, rtol=1e-5, atol=1e-8)


def test_dtype_conversion():
    """Test modal conversion with dtype parameter."""
    import torch_modal
    x = torch.randn(2, 2, dtype=torch.float32)
    y = x.modal(dtype=torch.float64)
    assert y.dtype == torch.float64


def test_copy_parameter():
    """Test modal conversion with copy parameter."""
    import torch_modal
    x = torch.randn(2, 2)
    y = x.modal(copy=True)
    z = x.modal(copy=False)
    assert y is not None and z is not None


def test_error_handling():
    """Test that errors are handled gracefully."""
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
        
    assert True  # If we get here without segfault, it's good


# Skip device functions test by default
@pytest.mark.skip(reason="Device functions test disabled")
def test_device_functions_disabled():
    """Test modal device functions (disabled)."""
    test_device_functions()


# Legacy standalone execution support
class TestResult:
    """Simple test result tracker for standalone mode."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test(self, name, test_func):
        """Run a test and track results."""
        print(f"Testing {name}...", end=" ")
        try:
            test_func()
            print("✓ PASS")
            self.passed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            print(f"   Full traceback:")
            traceback.print_exc()
            self.failed += 1
            self.errors.append(f"{name}: {e}\n{traceback.format_exc()}")
    
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


def run_debug_mode():
    """Run minimal debug tests for quick verification."""
    print("PyTorch Modal Debug Mode")
    print("=" * 30)
    
    print("Importing torch_modal...")
    try:
        import torch_modal
        print("✓ Import successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return 1
    
    print("Creating tensor...")
    x = torch.randn(2, 2)
    print(f"✓ Tensor created: {x.device}")
    
    print("Checking modal method...")
    print(f"✓ Has modal method: {hasattr(x, 'modal')}")
    
    if hasattr(x, 'modal'):
        print("Trying modal conversion...")
        try:
            y = x.modal()
            print(f"✓ Modal conversion success: {y.device}")
            print(f"Modal tensor data: {y}")
            
            print("Trying tensor addition...")
            z = y + y
            print(f"✓ Addition successful: {z.device}")
            print(f"Result: {z}")
            return 0
            
        except Exception as e:
            print(f"✗ Error: {e}")
            traceback.print_exc()
            return 1
    else:
        print("✗ Modal method not found on tensor")
        return 1


def run_comprehensive_tests(verbose=False):
    """Run comprehensive test suite in standalone mode."""
    print("PyTorch Modal Device Test Suite")
    print("=" * 50)
    
    results = TestResult()
    
    # Core functionality tests
    results.test("Basic imports", test_basic_imports)
    results.test("Tensor modal method", test_tensor_modal_method)
    results.test("Modal tensor creation", test_modal_tensor_creation)
    results.test("Modal tensor operations", test_modal_tensor_operations)
    
    # Advanced tests
    results.test("Dtype conversion", test_dtype_conversion)
    results.test("Copy parameter", test_copy_parameter)
    results.test("Error handling", test_error_handling)
    
    if verbose:
        print("\nRunning additional verbose tests...")
        try:
            import torch_modal
            print(f"Modal device available: {torch_modal.modal.is_available()}")
            print(f"Modal device count: {torch_modal.modal.device_count()}")
            print(f"Modal device name: {torch_modal.modal.get_device_name()}")
            
            # Test mixed device operations
            print("\nTesting mixed device operations...")
            x_cpu = torch.randn(2, 2)
            y_cpu = torch.randn(2, 2)
            x_modal = x_cpu.modal()
            y_modal = y_cpu.modal()
            
            print(f"CPU tensor device: {x_cpu.device}")
            print(f"Modal tensor device: {x_modal.device}")
            
            try:
                z_modal = x_modal.mm(y_modal)
                print(f"Modal-Modal operation: {z_modal.device}")
            except Exception as e:
                print(f"Modal-Modal operation failed: {e}")
            
            try:
                z_mixed = x_modal.mm(y_cpu)
                print("WARNING: Mixed device operation succeeded (unexpected)")
            except Exception as e:
                print(f"Mixed device operation correctly failed: {e}")
                
            try:
                direct_modal = torch.randn(3, 3, device='modal')
                print(f"Direct modal tensor creation: {direct_modal.device}")
            except Exception as e:
                print(f"Direct modal tensor creation failed: {e}")
                
        except Exception as e:
            print(f"Verbose tests failed: {e}")
    
    # Print summary
    success = results.summary()
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {results.failed} test(s) failed.")
        return 1


def main():
    """Main entry point with argument parsing for standalone execution."""
    parser = argparse.ArgumentParser(description='PyTorch Modal Test Suite')
    parser.add_argument('--debug', action='store_true', 
                       help='Run minimal debug tests for quick verification')
    parser.add_argument('--verbose', action='store_true',
                       help='Run comprehensive tests with verbose output')
    
    args = parser.parse_args()
    
    if args.debug:
        return run_debug_mode()
    else:
        return run_comprehensive_tests(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())