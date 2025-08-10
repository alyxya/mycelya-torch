// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "MycelyaTensorImpl.h"
#include <ATen/TensorUtils.h>
#include <c10/core/TensorImpl.h>
#include <iostream>

namespace mycelya {

// MycelyaTensorImpl implementation
MycelyaTensorImpl::MycelyaTensorImpl(
  const c10::Storage& storage, 
  const caffe2::TypeMeta& data_type)
  : c10::TensorImpl(
      c10::Storage(storage),  // Copy construct for move
      c10::DispatchKeySet{c10::DispatchKey::PrivateUse1, c10::DispatchKey::AutogradPrivateUse1},
      data_type) {
  
  // Following pytorch-npu pattern
  is_non_overlapping_and_dense_ = false;
  
  // Simple verification: log creation for debugging
  if (std::getenv("MYCELYA_DEBUG_TENSORIMPL")) {
    std::cout << "[MycelyaTensorImpl] Created tensor with storage_id=" 
              << get_storage_id() << " dtype=" << data_type.name() << std::endl;
  }
}

void MycelyaTensorImpl::shallow_copy_from(const c10::intrusive_ptr<c10::TensorImpl>& impl) {
  mark_accessed();
  
  // Copy metadata from source tensor implementation
  // This is similar to how pytorch-npu handles shallow copy
  set_storage_and_dtype(impl->storage(), impl->dtype());
  set_sizes_and_strides(impl->sizes(), impl->strides(), impl->storage_offset());
  
  refresh_numel();
  refresh_contiguous();
}

c10::intrusive_ptr<c10::TensorImpl> MycelyaTensorImpl::shallow_copy_and_detach(
  const c10::VariableVersion& version_counter,
  bool allow_tensor_metadata_change) const {
  
  mark_accessed();
  
  // Create new MycelyaTensorImpl with same storage
  auto impl = c10::make_intrusive<MycelyaTensorImpl>(
    storage(),
    dtype());
  
  // Copy metadata from this tensor to the new tensor
  impl->set_storage_and_dtype(storage(), dtype());
  impl->set_sizes_and_strides(sizes(), strides(), storage_offset());
  impl->set_version_counter(version_counter);
  
  impl->refresh_numel();
  impl->refresh_contiguous();
  
  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> MycelyaTensorImpl::shallow_copy_and_detach(
  c10::VariableVersion&& version_counter,
  bool allow_tensor_metadata_change) const {
  
  mark_accessed();
  
  // Create new MycelyaTensorImpl with same storage
  auto impl = c10::make_intrusive<MycelyaTensorImpl>(
    storage(),
    dtype());
  
  // Copy metadata from this tensor to the new tensor
  impl->set_storage_and_dtype(storage(), dtype());
  impl->set_sizes_and_strides(sizes(), strides(), storage_offset());
  impl->set_version_counter(std::move(version_counter));
  
  impl->refresh_numel();
  impl->refresh_contiguous();
  
  return impl;
}

storage_id_t MycelyaTensorImpl::get_storage_id() const {
  mark_accessed();
  return reinterpret_cast<storage_id_t>(storage().data_ptr().get());
}

void MycelyaTensorImpl::mark_accessed() const {
  accessed_via_custom_impl_ = true;
  
  // Simple verification: log access for debugging
  if (std::getenv("MYCELYA_DEBUG_TENSORIMPL")) {
    std::cout << "[MycelyaTensorImpl] Accessed tensor with storage_id=" 
              << reinterpret_cast<storage_id_t>(storage().data_ptr().get()) << std::endl;
  }
}

bool MycelyaTensorImpl::was_accessed_via_custom_impl() const {
  return accessed_via_custom_impl_;
}

// Factory functions
at::Tensor make_mycelya_tensor_with_custom_impl(
  const c10::Storage& storage,
  const caffe2::TypeMeta& data_type) {
  
  // Create tensor using the custom TensorImpl, following pytorch-npu pattern
  return at::detail::make_tensor<MycelyaTensorImpl>(storage, data_type);
}


} // namespace mycelya