// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "MycelyaAllocator.h"

#include <torch/csrc/utils/pybind.h>

#include "Mycelya.h" // For get_method

namespace mycelya {

at::DataPtr MycelyaAllocator::allocate(size_t nbytes) {
  pybind11::gil_scoped_acquire acquire;
  auto curr_device_idx = get_method("get_device")().cast<c10::DeviceIndex>();
  auto curr_device = c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
  void *data = nullptr;

  // Create storage and get the generated storage ID
  // Returns the storage ID on success, or 0 on failure
  storage_id_t storage_id =
      get_method("create_storage")(nbytes, curr_device_idx)
          .cast<storage_id_t>();

  TORCH_CHECK(storage_id != 0, "Failed to allocate storage (", nbytes,
              " bytes) on mycelya device ", curr_device_idx);

  // Store the storage ID as the data pointer (always non-zero)
  data = reinterpret_cast<void *>(storage_id);

  return {data, data, &ReportAndDelete, curr_device};
}

void MycelyaAllocator::ReportAndDelete(void *ptr) {
  if (!ptr || !Py_IsInitialized()) {
    return;
  }

  pybind11::gil_scoped_acquire acquire;

  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  // Always stash, this will be a no-op if there is no error
  PyErr_Fetch(&type, &value, &traceback);

  // Convert pointer back to storage ID for deletion
  storage_id_t storage_id = reinterpret_cast<storage_id_t>(ptr);
  get_method("free_storage")(storage_id);

  // If that user code raised an error, just print it without raising it
  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  // Restore the original error
  PyErr_Restore(type, value, traceback);
}

at::DeleterFnPtr MycelyaAllocator::raw_deleter() const {
  return &ReportAndDelete;
}

void MycelyaAllocator::copy_data(void *dest, const void *src,
                                 std::size_t count) const {
  // No-op: Mycelya tensors handle data copying through PyTorch operations
  // rather than raw memory copying
}

// Global allocator instance
static MycelyaAllocator global_mycelya_alloc;

MycelyaAllocator &get_mycelya_allocator() { return global_mycelya_alloc; }

} // namespace mycelya

// Register the allocator with PyTorch
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1,
                   &mycelya::global_mycelya_alloc);