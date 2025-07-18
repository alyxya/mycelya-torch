// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <torch/csrc/utils/pybind.h>
#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <random>
#include <string>

namespace remote {

using remote_ptr_t = uint64_t;
using storage_id_t = uint64_t;  // Changed from string to integer for efficient storage as data pointer

void set_driver_exec(PyObject* driver_exec_fn);
py::function get_method(const char* name);

static constexpr char kFreeMethod[] = "free";
static constexpr char kCreateStorageMethod[] = "create_storage_with_id";
static constexpr char kFreeStorageMethod[] = "free_storage_with_id";
static constexpr char kGenerateStorageIdMethod[] = "generate_storage_id";
static constexpr char kMallocMethod[] = "malloc";  // Unused legacy method name

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

// Utility functions for storage ID management
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

  // For storage ID-based deletion, convert pointer back to storage ID
  if (name == kFreeStorageMethod) {
    storage_id_t storage_id = reinterpret_cast<storage_id_t>(ptr);
    TORCH_CHECK(
        get_method(name)(storage_id).cast<bool>(),
        "Failed to free storage with ID ",
        storage_id);
  } else {
    // Legacy pointer-based deletion
    TORCH_CHECK(
        get_method(name)(reinterpret_cast<remote_ptr_t>(ptr)).cast<bool>(),
        "Failed to free memory pointer at ",
        ptr);
  }

  // If that user code raised an error, just print it without raising it
  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  // Restore the original error
  PyErr_Restore(type, value, traceback);
}

} // namespace remote