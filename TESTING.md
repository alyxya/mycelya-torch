# PyTorch Modal Testing Guide

This document describes the testing structure for the torch-modal package.

## Test Files

### `test_torch_modal.py` - **Main Test Suite**
**Use this for CI/automated testing and comprehensive validation.**

The primary test suite with structured test cases covering:
- ✅ Import testing
- ✅ C extension functionality  
- ✅ Device availability and properties
- ✅ Modal tensor creation and operations
- ✅ Parameter handling (dtype, copy, etc.)
- ✅ Error handling and edge cases

**Usage:**
```bash
python test_torch_modal.py
```

**Output:** Structured test results with pass/fail counts and detailed error reporting.

### `simple_test.py` - **Quick Verification**
**Use this for quick development checks.**

Minimal test to verify the basic modal() method works:
- Import torch_modal
- Check if modal() method exists on tensors
- Create a modal tensor

**Usage:**
```bash
python simple_test.py
```

### `debug_test.py` - **Development Debugging**
**Use this when debugging implementation issues.**

Basic debugging script that tests:
- Package import
- Modal tensor creation
- Simple operations (addition)
- Error output with stack traces

**Usage:**
```bash
python debug_test.py
```

### `test_modal.py` - **Advanced Features**
**Use this to test more complex functionality.**

Comprehensive feature testing including:
- Device properties and availability
- Matrix operations between modal tensors
- Mixed device operations (CPU + modal)
- Direct modal tensor creation attempts
- Error handling for unsupported operations

**Usage:**
```bash
python test_modal.py
```

## Test Hierarchy

```
Complexity:     simple_test.py → debug_test.py → test_modal.py → test_torch_modal.py
Use Case:       Quick check   → Debug issues → Feature test → Full validation
```

## Running All Tests

To run all tests in sequence:

```bash
echo "=== Simple Test ===" && python simple_test.py && \
echo -e "\n=== Debug Test ===" && python debug_test.py && \
echo -e "\n=== Modal Test ===" && python test_modal.py && \
echo -e "\n=== Full Test Suite ===" && python test_torch_modal.py
```

## What Was Removed

The following redundant test files were consolidated into the above structure:
- `c_extension_debug.py` → functionality moved to `test_torch_modal.py`
- `minimal_test.py` → functionality moved to `simple_test.py`
- `segfault_debug.py` → no longer needed (segfaults fixed)
- `test_modal_method.py` → functionality moved to `simple_test.py`

## Adding New Tests

When adding new functionality to torch-modal:

1. **Quick check**: Add to `simple_test.py` 
2. **Feature test**: Add to `test_modal.py`
3. **Comprehensive test**: Add new test function to `test_torch_modal.py`

All test files should be kept runnable and self-contained.