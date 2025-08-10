// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Mycelya.h"
#include "MycelyaTensorImpl.h"

#include <ATen/Context.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/autograd/python_variable.h>

static PyObject *_initExtension(PyObject *self, PyObject *noargs) {
  HANDLE_TH_ERRORS

  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject *_getDefaultGenerator(PyObject *self, PyObject *arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg),
              "_get_default_generator expects an int, but got ",
              THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  return THPGenerator_initDefaultGenerator(at::globalContext().defaultGenerator(
      c10::Device(c10::DeviceType::PrivateUse1, idx)));

  END_HANDLE_TH_ERRORS
}

// Test function to check if a tensor is using custom TensorImpl
static PyObject *_is_using_custom_tensorimpl(PyObject *self, PyObject *arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPVariable_Check(arg),
              "_is_using_custom_tensorimpl expects a tensor, but got ",
              THPUtils_typename(arg));
  
  auto tensor = THPVariable_Unpack(arg);
  
  // Check if tensor is using our custom TensorImpl
  auto* impl_ptr = dynamic_cast<mycelya::MycelyaTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl_ptr) {
    return PyBool_FromLong(1);
  } else {
    return PyBool_FromLong(0);
  }
  
  END_HANDLE_TH_ERRORS
}

// Test function to check if a tensor was accessed via custom TensorImpl
static PyObject *_was_tensor_accessed_via_custom_impl(PyObject *self, PyObject *arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPVariable_Check(arg),
              "_was_tensor_accessed_via_custom_impl expects a tensor, but got ",
              THPUtils_typename(arg));
  
  auto tensor = THPVariable_Unpack(arg);
  
  // Check if tensor is using our custom TensorImpl and was accessed
  auto* impl_ptr = dynamic_cast<mycelya::MycelyaTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl_ptr) {
    return PyBool_FromLong(impl_ptr->was_accessed_via_custom_impl() ? 1 : 0);
  } else {
    return PyBool_FromLong(0);
  }
  
  END_HANDLE_TH_ERRORS
}

static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {"_is_using_custom_tensorimpl", _is_using_custom_tensorimpl, METH_O, nullptr},
    {"_was_tensor_accessed_via_custom_impl", _was_tensor_accessed_via_custom_impl, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef mycelya_C_module = {
    PyModuleDef_HEAD_INIT, "mycelya_torch._C", nullptr, -1, methods};

PyMODINIT_FUNC PyInit__C(void) {
  PyObject *mod = PyModule_Create(&mycelya_C_module);

  py::object mycelya_mod = py::module_::import("mycelya_torch");
  // Only borrowed from the python side!
  mycelya::set_impl_factory(mycelya_mod.attr("impl_factory").ptr());

  return mod;
}