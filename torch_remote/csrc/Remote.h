#pragma once

#include <torch/csrc/utils/pybind.h>
#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <random>
#include <string>

namespace remote {

using remote_ptr_t = uint64_t;
using tensor_id_t = std::string;

void set_impl_factory(PyObject* factory);
py::function get_method(const char* name);

static constexpr char kFreeMethod[] = "free";
static constexpr char kHostFreeMethod[] = "hostFree";
static constexpr char kCreateTensorMethod[] = "create_tensor_with_id";
static constexpr char kFreeTensorMethod[] = "free_tensor_with_id";

// C++ tensor creation functions
at::Tensor empty_remote(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format);

at::Tensor empty_strided_remote(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);

// Utility functions for tensor ID management
tensor_id_t generate_tensor_id();
bool validate_device_index(c10::DeviceIndex device_index);

template <const char* name>
static void ReportAndDelete(void* ptr) {
  if (!ptr || !Py_IsInitialized()) {
    return;
  }

  py::gil_scoped_acquire acquire;

  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  // Always stash, this will be a no-op if there is no error
  PyErr_Fetch(&type, &value, &traceback);

  TORCH_CHECK(
      get_method(name)(reinterpret_cast<remote_ptr_t>(ptr)).cast<bool>(),
      "Failed to free memory pointer at ",
      ptr);

  // If that user code raised an error, just print it without raising it
  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  // Restore the original error
  PyErr_Restore(type, value, traceback);
}

} // namespace remote