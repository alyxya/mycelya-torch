// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Remote.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/Allocator.h>
#include <iomanip>
#include <sstream>
#include <torch/library.h>

namespace remote {
namespace {

// ID-based allocator that stores storage IDs as data pointers
struct RemoteAllocator final : at::Allocator {
  RemoteAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    auto curr_device_idx = get_method("get_device")().cast<c10::DeviceIndex>();
    auto curr_device =
        c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
    void *data = nullptr;

    // Always generate a unique storage ID for all tensors
    // This ensures every tensor gets a unique ID
    storage_id_t storage_id =
        get_method("generate_storage_id")().cast<storage_id_t>();

    // Call Python method to create storage with ID and register it
    // This should create the storage remotely and return success/failure
    bool success =
        get_method("create_storage_with_id")(storage_id, nbytes, curr_device_idx)
            .cast<bool>();

    TORCH_CHECK(success, "Failed to allocate storage with ID ", storage_id,
                " (", nbytes, " bytes) on remote device ", curr_device_idx);

    // Store the storage ID as the data pointer (always non-zero)
    data = reinterpret_cast<void *>(storage_id);

    return {data, data, &ReportAndDelete, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void *dest, const void *src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    // Convert data pointers back to storage IDs for the copy operation
    storage_id_t dest_id = reinterpret_cast<storage_id_t>(dest);
    storage_id_t src_id = reinterpret_cast<storage_id_t>(src);
    get_method("copy_data_by_id")(dest_id, src_id, count);
  }
};

static RemoteAllocator global_remote_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_remote_alloc);

} // namespace

// Validate device index
bool validate_device_index(c10::DeviceIndex device_index) {
  py::gil_scoped_acquire acquire;
  try {
    auto device_count = get_method("device_count")().cast<c10::DeviceIndex>();
    return device_index >= 0 && device_index < device_count;
  } catch (...) {
    return false;
  }
}

// Storage cleanup function for allocator
void ReportAndDelete(void *ptr) {
  if (!ptr || !Py_IsInitialized()) {
    return;
  }

  py::gil_scoped_acquire acquire;

  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  // Always stash, this will be a no-op if there is no error
  PyErr_Fetch(&type, &value, &traceback);

  // Convert pointer back to storage ID for deletion
  storage_id_t storage_id = reinterpret_cast<storage_id_t>(ptr);
  get_method("free_storage_with_id")(storage_id);

  // If that user code raised an error, just print it without raising it
  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  // Restore the original error
  PyErr_Restore(type, value, traceback);
}

// C++ implementation of empty_remote using direct allocator integration
at::Tensor empty_remote(at::IntArrayRef size,
                        c10::optional<at::ScalarType> dtype,
                        c10::optional<at::Layout> layout,
                        c10::optional<at::Device> device,
                        c10::optional<bool> pin_memory,
                        c10::optional<at::MemoryFormat> memory_format) {

  // Handle device resolution
  c10::Device target_device =
      device.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));
  if (target_device.type() != c10::DeviceType::PrivateUse1) {
    target_device =
        c10::Device(c10::DeviceType::PrivateUse1, target_device.index());
  }

  // Validate device index
  TORCH_CHECK(validate_device_index(target_device.index()),
              "Invalid device index: ", target_device.index());

  // Handle other parameters
  auto resolved_dtype = dtype.value_or(at::get_default_dtype_as_scalartype());
  auto resolved_layout = layout.value_or(at::Layout::Strided);
  auto resolved_memory_format =
      memory_format.value_or(at::MemoryFormat::Contiguous);

  TORCH_CHECK(resolved_layout == at::Layout::Strided,
              "Only strided layout is supported");
  TORCH_CHECK(!pin_memory.value_or(false),
              "Pin memory is not supported on remote devices");

  // Set device guard to ensure allocation happens on correct device
  const c10::DeviceGuard device_guard(target_device);

  // Use the enhanced allocator to create tensor
  constexpr c10::DispatchKeySet remote_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_remote_alloc, remote_dks,
                                   resolved_dtype, resolved_memory_format);
}

// C++ implementation of empty_strided_remote
at::Tensor empty_strided_remote(at::IntArrayRef size, at::IntArrayRef stride,
                                c10::optional<at::ScalarType> dtype,
                                c10::optional<at::Layout> layout,
                                c10::optional<at::Device> device,
                                c10::optional<bool> pin_memory) {

  // Handle device resolution
  c10::Device target_device =
      device.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));
  if (target_device.type() != c10::DeviceType::PrivateUse1) {
    target_device =
        c10::Device(c10::DeviceType::PrivateUse1, target_device.index());
  }

  // Validate device index
  TORCH_CHECK(validate_device_index(target_device.index()),
              "Invalid device index: ", target_device.index());

  // Handle other parameters
  auto resolved_dtype = dtype.value_or(at::get_default_dtype_as_scalartype());
  auto resolved_layout = layout.value_or(at::Layout::Strided);

  TORCH_CHECK(resolved_layout == at::Layout::Strided,
              "Only strided layout is supported");
  TORCH_CHECK(!pin_memory.value_or(false),
              "Pin memory is not supported on remote devices");

  // Set device guard to ensure allocation happens on correct device
  const c10::DeviceGuard device_guard(target_device);

  // Use the enhanced allocator to create strided tensor
  constexpr c10::DispatchKeySet remote_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(size, stride, &global_remote_alloc,
                                           remote_dks, resolved_dtype);
}

// C++ implementation of as_strided for view operations
at::Tensor as_strided_remote(const at::Tensor &self, at::IntArrayRef size,
                             at::IntArrayRef stride,
                             c10::optional<int64_t> storage_offset) {

  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "as_strided_remote expects a remote tensor");

  // Use the CPU implementation directly to avoid infinite recursion
  // This creates a view tensor that shares storage without triggering remote
  // calls
  return at::cpu::as_strided(self, size, stride, storage_offset);
}

// C++ implementation of set_ for tensor metadata operations
at::Tensor &set_remote(at::Tensor &result, at::Storage storage,
                       int64_t storage_offset, at::IntArrayRef size,
                       at::IntArrayRef stride) {
  // Use the CPU implementation directly to avoid infinite recursion
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

// C++ implementation of resize_ that explicitly calls storage resize hooks
const at::Tensor &
resize_remote_(const at::Tensor &self, at::IntArrayRef size,
               c10::optional<at::MemoryFormat> memory_format) {

  // Calculate required storage size for new shape
  int64_t new_numel = 1;
  for (auto s : size) {
    new_numel *= s;
  }

  size_t element_size = self.dtype().itemsize();
  size_t required_bytes = new_numel * element_size;
  size_t current_bytes = self.storage().nbytes();

  // Get storage reference for potential resize
  auto storage = self.storage();

  // Only resize storage if we need MORE space (growth)
  // For shrinking, keep existing storage and just change tensor view
  if (required_bytes > current_bytes) {
    // Directly call the resize hook through PrivateUse1HooksInterface
    at::detail::getPrivateUse1Hooks().resizePrivateUse1Bytes(storage,
                                                             required_bytes);
  }

  // Calculate new strides for contiguous layout (assuming contiguous memory
  // format)
  std::vector<int64_t> new_stride(size.size());
  if (size.size() > 0) {
    new_stride[size.size() - 1] = 1;
    for (int64_t i = size.size() - 2; i >= 0; i--) {
      new_stride[i] = new_stride[i + 1] * size[i + 1];
    }
  }

  // Update tensor metadata using set_ operation
  // This updates shape, stride, and storage_offset without allocating new
  // storage
  const_cast<at::Tensor &>(self).set_(storage, 0, size, new_stride);

  return self;
}

// Register the C++ implementations directly with PyTorch's dispatch system
// This follows the OpenReg pattern where empty operations are implemented in
// C++
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // Register our C++ implementations for empty tensor creation
  // These will override the Python fallback for these specific operations
  m.impl("empty.memory_format", empty_remote);
  m.impl("empty_strided", empty_strided_remote);

  // Register as_strided and set_ like pytorch-openreg-2
  // All other operations (transpose, squeeze, unsqueeze, view) go through
  // Python fallback
  m.impl("as_strided", as_strided_remote);
  m.impl("set_.source_Storage_storage_offset", set_remote);

  // Register resize_ following OpenReg pattern - uses default implementation
  // with custom hook
  m.impl("resize_", resize_remote_);
}

} // namespace remote