// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Mycelya.h"
#include "MycelyaTensorImpl.h"

#include <ATen/Context.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

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

// Get the metadata hash for a mycelya tensor
static PyObject *_get_metadata_hash(PyObject *self, PyObject *arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPVariable_Check(arg),
              "_get_metadata_hash expects a tensor, but got ",
              THPUtils_typename(arg));

  auto tensor = THPVariable_Unpack(arg);

  // Check if tensor is using our custom TensorImpl
  auto *impl_ptr =
      dynamic_cast<mycelya::MycelyaTensorImpl *>(tensor.unsafeGetTensorImpl());
  if (impl_ptr) {
    auto metadata_hash = impl_ptr->get_metadata_hash();
    return PyLong_FromUnsignedLongLong(metadata_hash);
  } else {
    TORCH_CHECK(false, "Tensor is not a mycelya tensor with custom TensorImpl");
  }

  END_HANDLE_TH_ERRORS
}

static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {"_get_metadata_hash", _get_metadata_hash, METH_O, nullptr},
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