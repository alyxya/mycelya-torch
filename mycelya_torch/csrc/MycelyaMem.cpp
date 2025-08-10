// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Mycelya.h"
#include "MycelyaTensorImpl.h"
#include "MycelyaStorageImpl.h"
#include "MycelyaAllocator.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/Allocator.h>
#include <iomanip>
#include <sstream>
#include <torch/library.h>

namespace mycelya {
namespace {

// Always use custom TensorImpl - this is now the default and only option

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

// C++ implementation of empty_mycelya using direct allocator integration
at::Tensor empty_mycelya(at::IntArrayRef size,
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
  auto resolved_memory_format [[maybe_unused]] =
      memory_format.value_or(at::MemoryFormat::Contiguous);

  TORCH_CHECK(resolved_layout == at::Layout::Strided,
              "Only strided layout is supported");
  TORCH_CHECK(!pin_memory.value_or(false),
              "Pin memory is not supported on remote devices");

  // Set device guard to ensure allocation happens on correct device
  const c10::DeviceGuard device_guard(target_device);

  // Calculate storage size requirements
  int64_t numel = 1;
  for (auto s : size) {
    numel *= s;
  }
  size_t element_size = c10::elementSize(resolved_dtype);
  size_t size_bytes = numel * element_size;

  // Create custom storage using registered factory (this will automatically
  // use our custom MycelyaStorageImpl through c10::SetStorageImplCreate)
  c10::intrusive_ptr<c10::StorageImpl> storage_impl = make_mycelya_storage_impl(
    c10::StorageImpl::use_byte_size_t(),
    c10::SymInt(size_bytes),
    c10::DataPtr(),  // Empty DataPtr - let the factory call our allocator
    &get_mycelya_allocator(),
    true);

  // Create tensor using custom MycelyaTensorImpl (following pytorch-npu pattern)
  auto tensor = at::detail::make_tensor<MycelyaTensorImpl>(
    c10::Storage(storage_impl), 
    caffe2::TypeMeta::fromScalarType(resolved_dtype));
  
  // Set the proper sizes and strides for the requested shape
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, c10::contiguous_strides(size));
  return tensor;
}

// C++ implementation of empty_strided_mycelya
at::Tensor empty_strided_mycelya(at::IntArrayRef size, at::IntArrayRef stride,
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

  // Calculate storage size requirements based on strides
  int64_t storage_size = 1;
  for (size_t i = 0; i < size.size(); ++i) {
    if (size[i] == 0) {
      storage_size = 0;
      break;
    }
    storage_size = std::max(storage_size, (size[i] - 1) * stride[i] + 1);
  }
  size_t element_size = c10::elementSize(resolved_dtype);
  size_t size_bytes = storage_size * element_size;

  // Create custom storage using registered factory
  c10::intrusive_ptr<c10::StorageImpl> storage_impl = make_mycelya_storage_impl(
    c10::StorageImpl::use_byte_size_t(),
    c10::SymInt(size_bytes),
    c10::DataPtr(),  // Empty DataPtr - let the factory call our allocator
    &get_mycelya_allocator(),
    true);

  // Create tensor using custom MycelyaTensorImpl (following pytorch-npu pattern)
  auto tensor = at::detail::make_tensor<MycelyaTensorImpl>(
    c10::Storage(storage_impl), 
    caffe2::TypeMeta::fromScalarType(resolved_dtype));
  
  // Set the proper sizes and strides for the requested shape
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

// C++ implementation of as_strided for view operations
at::Tensor as_strided_mycelya(const at::Tensor &self, at::IntArrayRef size,
                             at::IntArrayRef stride,
                             c10::optional<int64_t> storage_offset) {

  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "as_strided_mycelya expects a mycelya tensor");

  // Use the CPU implementation directly to avoid infinite recursion
  // This creates a view tensor that shares storage without triggering mycelya
  // calls
  return at::cpu::as_strided(self, size, stride, storage_offset);
}

// C++ implementation of set_ for tensor metadata operations
at::Tensor &set_mycelya(at::Tensor &result, at::Storage storage,
                       int64_t storage_offset, at::IntArrayRef size,
                       at::IntArrayRef stride) {
  // Use the CPU implementation directly to avoid infinite recursion
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

// C++ implementation of resize_ that explicitly calls storage resize hooks
const at::Tensor &
resize_mycelya_(const at::Tensor &self, at::IntArrayRef size,
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
  m.impl("empty.memory_format", empty_mycelya);
  m.impl("empty_strided", empty_strided_mycelya);

  // Register as_strided and set_ like pytorch-openreg-2
  // All other operations (transpose, squeeze, unsqueeze, view) go through
  // Python fallback
  m.impl("as_strided", as_strided_mycelya);
  m.impl("set_.source_Storage_storage_offset", set_mycelya);

  // Register resize_ following OpenReg pattern - uses default implementation
  // with custom hook
  m.impl("resize_", resize_mycelya_);
}

} // namespace mycelya
