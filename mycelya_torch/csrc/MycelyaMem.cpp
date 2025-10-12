// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <ATen/InferSize.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/Allocator.h>
#include <torch/library.h>

#include <iomanip>
#include <sstream>

#include "Mycelya.h"
#include "MycelyaAllocator.h"
#include "MycelyaStorageImpl.h"
#include "MycelyaTensorImpl.h"

namespace mycelya {

// C++ implementation of empty_mycelya using custom TensorImpl (simplified from
// NPU pattern)
at::Tensor empty_mycelya(at::IntArrayRef size,
                         c10::optional<at::ScalarType> dtype,
                         c10::optional<at::Layout> layout,
                         c10::optional<at::Device> device,
                         c10::optional<bool> pin_memory,
                         c10::optional<at::MemoryFormat> memory_format) {
  // Require explicit device - no defaults to avoid masking bugs
  TORCH_CHECK(device.has_value(),
              "empty_mycelya requires explicit device specification");
  c10::Device target_device = *device;
  TORCH_CHECK(
      target_device.type() == c10::DeviceType::PrivateUse1,
      "empty_mycelya expects PrivateUse1 device, got: ", target_device.type());

  const auto resolved_dtype = c10::dtype_or_default(dtype);
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided,
              "Only strided layout is supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory),
              "Pin memory is not supported on remote devices");

  const c10::DeviceGuard device_guard(target_device);

  int64_t nelements = c10::multiply_integers(size);
  auto dtype_meta = c10::scalarTypeToTypeMeta(resolved_dtype);
  int64_t size_bytes = nelements * dtype_meta.itemsize();

  // Create custom storage (required for our custom StorageImpl)
  c10::intrusive_ptr<c10::StorageImpl> storage_impl = make_mycelya_storage_impl(
      c10::StorageImpl::use_byte_size_t(), c10::SymInt(size_bytes),
      c10::DataPtr(), // Empty DataPtr - let the factory call our allocator
      &get_mycelya_allocator(), true);

  // Create tensor using custom MycelyaTensorImpl (required for metadata hash)
  auto tensor = at::detail::make_tensor<MycelyaTensorImpl>(
      c10::Storage(storage_impl), dtype_meta);

  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        size, c10::contiguous_strides(size));
  }

  return tensor;
}

// C++ implementation of empty_strided_mycelya following NPU pattern
at::Tensor empty_strided_mycelya(at::IntArrayRef size, at::IntArrayRef stride,
                                 c10::optional<at::ScalarType> dtype,
                                 c10::optional<at::Layout> layout,
                                 c10::optional<at::Device> device,
                                 c10::optional<bool> pin_memory) {
  // Require explicit device same as empty_mycelya
  TORCH_CHECK(device.has_value(),
              "empty_strided_mycelya requires explicit device specification");
  c10::Device target_device = *device;
  TORCH_CHECK(target_device.type() == c10::DeviceType::PrivateUse1,
              "empty_strided_mycelya expects PrivateUse1 device, got: ",
              target_device.type());

  TORCH_CHECK(size.size() == stride.size(),
              "empty_strided: size and stride must have the same length");

  const auto resolved_dtype = c10::dtype_or_default(dtype);
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided,
              "Only strided layout is supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory),
              "Pin memory is not supported on remote devices");

  const c10::DeviceGuard device_guard(target_device);

  // Calculate storage size needed for the strided layout
  // storage_size = 1 + sum((size[i] - 1) * stride[i]) for all dimensions
  int64_t storage_size = 1;
  for (size_t i = 0; i < size.size(); i++) {
    if (size[i] == 0) {
      storage_size = 0;
      break;
    }
    storage_size += (size[i] - 1) * stride[i];
  }

  auto dtype_meta = c10::scalarTypeToTypeMeta(resolved_dtype);
  int64_t size_bytes = storage_size * dtype_meta.itemsize();

  // Create custom storage with the correct size for strided layout
  c10::intrusive_ptr<c10::StorageImpl> storage_impl = make_mycelya_storage_impl(
      c10::StorageImpl::use_byte_size_t(), c10::SymInt(size_bytes),
      c10::DataPtr(), // Empty DataPtr - let the factory call our allocator
      &get_mycelya_allocator(), true);

  // Create tensor using custom MycelyaTensorImpl
  auto tensor = at::detail::make_tensor<MycelyaTensorImpl>(
      c10::Storage(storage_impl), dtype_meta);

  // Set the requested size and stride
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

// C++ implementation of as_strided for view operations
at::Tensor as_strided_mycelya(const at::Tensor &self, at::IntArrayRef size,
                              at::IntArrayRef stride,
                              c10::optional<int64_t> storage_offset) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "as_strided_mycelya expects a mycelya tensor");

  int64_t offset = storage_offset.value_or(self.storage_offset());

  // Create a new mycelya tensor with the same storage but different view
  // parameters This preserves the custom MycelyaTensorImpl unlike the CPU
  // fallback
  auto result =
      at::detail::make_tensor<MycelyaTensorImpl>(self.storage(), self.dtype());

  // Set the new sizes and strides for the view
  auto *impl = result.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(size, stride, offset);
  return result;
}

// C++ implementation of view for efficient view operations
at::Tensor view_mycelya(const at::Tensor &self, at::IntArrayRef size) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "view_mycelya expects a mycelya tensor");

  at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(stride.has_value(),
              "view size is not compatible with input tensor's size and stride "
              "(at least one dimension spans across two contiguous subspaces). "
              "Use .reshape(...) instead.");

  return as_strided_mycelya(self, inferred_size, *stride,
                            self.storage_offset());
}

// C++ implementation of _unsafe_view to preserve MycelyaTensorImpl
at::Tensor _unsafe_view_mycelya(const at::Tensor &self, at::IntArrayRef size) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "_unsafe_view_mycelya expects a mycelya tensor");

  at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "_unsafe_view size is not compatible with input tensor's size and stride "
      "(at least one dimension spans across two contiguous subspaces).");

  return as_strided_mycelya(self, inferred_size, *stride,
                            self.storage_offset());
}

// C++ implementation of set_.source_Storage_storage_offset for tensor metadata
// operations
at::Tensor &set_mycelya(at::Tensor &result, at::Storage storage,
                        int64_t storage_offset, at::IntArrayRef size,
                        at::IntArrayRef stride) {
  TORCH_CHECK(result.device().type() == c10::DeviceType::PrivateUse1,
              "set_mycelya expects a mycelya tensor");

  // Update the tensor's storage and metadata while preserving custom TensorImpl
  auto *impl = result.unsafeGetTensorImpl();
  impl->set_storage_and_dtype(storage, result.dtype());
  impl->set_sizes_and_strides(size, stride, storage_offset);
  return result;
}

// C++ implementation of set_.source_Tensor for tensor aliasing operations
at::Tensor &set_source_tensor_mycelya(at::Tensor &self,
                                      const at::Tensor &source) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "set_source_tensor_mycelya expects a mycelya tensor");
  TORCH_CHECK(source.device().type() == c10::DeviceType::PrivateUse1,
              "set_source_tensor_mycelya expects a mycelya source tensor");

  // Delegate to the general set_ function with source tensor's metadata
  return set_mycelya(self, source.storage(), source.storage_offset(),
                     source.sizes(), source.strides());
}

// C++ implementation of set_.source_Storage for storage aliasing operations
at::Tensor &set_source_storage_mycelya(at::Tensor &self, at::Storage source) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "set_source_storage_mycelya expects a mycelya tensor");

  // Calculate size based on storage bytes and element size
  size_t element_size = self.dtype().itemsize();
  TORCH_CHECK(source.nbytes() % element_size == 0, "Storage size (",
              source.nbytes(), ") not divisible by element size (",
              element_size, ")");
  int64_t numel = source.nbytes() / element_size;

  // Delegate to the general set_ function with 1D shape and contiguous stride
  return set_mycelya(self, source, 0, {numel}, {1});
}

// C++ implementation of resize_ that explicitly calls storage resize hooks
const at::Tensor &
resize_mycelya_(const at::Tensor &self, at::IntArrayRef size,
                c10::optional<at::MemoryFormat> memory_format) {
  int64_t new_numel = c10::multiply_integers(size);

  size_t element_size = self.dtype().itemsize();
  size_t required_bytes = new_numel * element_size;
  size_t current_bytes = self.storage().nbytes();

  // Get storage reference for potential resize
  auto storage = self.storage();

  if (required_bytes > current_bytes) {
    at::detail::getPrivateUse1Hooks().resizePrivateUse1Bytes(storage,
                                                             required_bytes);
  }

  std::vector<int64_t> new_stride(size.size());
  if (size.size() > 0) {
    new_stride[size.size() - 1] = 1;
    for (int64_t i = size.size() - 2; i >= 0; i--) {
      new_stride[i] = new_stride[i + 1] * size[i + 1];
    }
  }

  const_cast<at::Tensor &>(self).set_(storage, self.storage_offset(), size, new_stride);

  return self;
}

// C++ implementation of alias that preserves MycelyaTensorImpl
at::Tensor alias_mycelya(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "alias_mycelya expects a mycelya tensor");

  return as_strided_mycelya(self, self.sizes(), self.strides(),
                            self.storage_offset());
}

// C++ implementation of _reshape_alias that calls as_strided_mycelya
at::Tensor _reshape_alias_mycelya(const at::Tensor &self, at::IntArrayRef size,
                                  at::IntArrayRef stride) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "_reshape_alias_mycelya expects a mycelya tensor");

  // Use as_strided_mycelya to create the alias with provided size and strides
  return as_strided_mycelya(self, size, stride, self.storage_offset());
}

// C++ implementation of _lazy_clone that preserves MycelyaTensorImpl
at::Tensor _lazy_clone_mycelya(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "_lazy_clone_mycelya expects a mycelya tensor");

  auto scalar_type = c10::typeMetaToScalarType(self.dtype());
  auto result = empty_mycelya(self.sizes(), scalar_type, c10::Layout::Strided,
                              self.device(), c10::nullopt, c10::nullopt);

  result.copy_(self);

  return result;
}

// Register the C++ implementations directly with PyTorch's dispatch system
// This follows the OpenReg pattern where empty operations are implemented in
// C++
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // Register our C++ implementations for empty tensor creation
  // These will override the Python fallback for these specific operations
  m.impl("empty.memory_format", empty_mycelya);
  m.impl("empty_strided", empty_strided_mycelya);

  // Register view operations in C++ for better performance
  m.impl("view", view_mycelya);
  m.impl("as_strided", as_strided_mycelya);
  m.impl("_unsafe_view", _unsafe_view_mycelya);
  m.impl("_reshape_alias", _reshape_alias_mycelya);

  // Register set_ operations in C++ following OpenReg pattern
  m.impl("set_.source_Tensor", set_source_tensor_mycelya);
  m.impl("set_.source_Storage", set_source_storage_mycelya);
  m.impl("set_.source_Storage_storage_offset", set_mycelya);

  // Register resize_ following OpenReg pattern - uses default implementation
  // with custom hook
  m.impl("resize_", resize_mycelya_);

  // Register alias and _lazy_clone operations in C++
  m.impl("alias", alias_mycelya);
  m.impl("_lazy_clone", _lazy_clone_mycelya);
}

} // namespace mycelya
