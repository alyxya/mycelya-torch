// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "MycelyaTensorImpl.h"

#include <ATen/TensorUtils.h>
#include <c10/core/TensorImpl.h>

#include <atomic>
#include <functional>

namespace mycelya {

// MycelyaTensorImpl implementation
MycelyaTensorImpl::MycelyaTensorImpl(const c10::Storage &storage,
                                     const caffe2::TypeMeta &data_type)
    : c10::TensorImpl(
          c10::Storage(storage), // Copy construct for move
          c10::DispatchKeySet{c10::DispatchKey::PrivateUse1,
                              c10::DispatchKey::AutogradPrivateUse1},
          data_type) {
  // Following pytorch-npu pattern
  is_non_overlapping_and_dense_ = false;
}

void MycelyaTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<c10::TensorImpl> &impl) {
  // Copy metadata from source tensor implementation
  // This is similar to how pytorch-npu handles shallow copy
  set_storage_and_dtype(impl->storage(), impl->dtype());
  set_sizes_and_strides(impl->sizes(), impl->strides(), impl->storage_offset());

  refresh_numel();
  refresh_contiguous();
}

c10::intrusive_ptr<c10::TensorImpl> MycelyaTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion &version_counter,
    bool allow_tensor_metadata_change) const {
  // Create new MycelyaTensorImpl with same storage
  auto impl = c10::make_intrusive<MycelyaTensorImpl>(storage(), dtype());

  // Copy metadata from this tensor to the new tensor
  impl->set_storage_and_dtype(storage(), dtype());
  impl->set_sizes_and_strides(sizes(), strides(), storage_offset());

  if (!impl->is_inference()) {
    impl->set_version_counter(version_counter);
  }
  impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();

  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> MycelyaTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion &&version_counter,
    bool allow_tensor_metadata_change) const {
  // Create new MycelyaTensorImpl with same storage
  auto impl = c10::make_intrusive<MycelyaTensorImpl>(storage(), dtype());

  // Copy metadata from this tensor to the new tensor
  impl->set_storage_and_dtype(storage(), dtype());
  impl->set_sizes_and_strides(sizes(), strides(), storage_offset());

  if (!impl->is_inference()) {
    impl->set_version_counter(std::move(version_counter));
  }
  impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();

  return impl;
}

storage_id_t MycelyaTensorImpl::get_storage_id() const {
  return reinterpret_cast<storage_id_t>(storage().data_ptr().get());
}

uint64_t MycelyaTensorImpl::get_metadata_hash() const {
  // Simple but effective hash combining shape, strides, dtype, offset, and
  // storage ID Using FNV-1a style hash for fast computation
  uint64_t hash = 14695981039346656037ULL; // FNV offset basis
  const uint64_t prime = 1099511628211ULL; // FNV prime

  // Hash shape dimensions
  for (auto size : sizes()) {
    hash ^= static_cast<uint64_t>(size);
    hash *= prime;
  }

  // Hash stride values
  for (auto stride : strides()) {
    hash ^= static_cast<uint64_t>(stride);
    hash *= prime;
  }

  // Hash dtype (use name hash since TypeMeta doesn't have simple conversion)
  auto dtype_name = dtype().name();
  for (char c : dtype_name) {
    hash ^= static_cast<uint64_t>(c);
    hash *= prime;
  }

  // Hash storage offset
  hash ^= static_cast<uint64_t>(storage_offset());
  hash *= prime;

  // Hash storage ID to distinguish tensors with different storage
  hash ^= get_storage_id();
  hash *= prime;

  return hash;
}

} // namespace mycelya
