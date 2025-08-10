// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include "Mycelya.h"
#include <c10/core/TensorImpl.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/intrusive_ptr.h>

namespace mycelya {

// Custom StorageImpl that holds storage IDs and metadata for mycelya tensors
struct MycelyaStorageImpl : public c10::StorageImpl {
  explicit MycelyaStorageImpl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable);

  // Get the storage ID directly from the stored data pointer
  storage_id_t get_storage_id() const;

  // Simple verification flag to test custom behavior
  mutable bool accessed_via_custom_impl = false;
};

// Custom TensorImpl that stores metadata locally for efficient access
class MycelyaTensorImpl : public c10::TensorImpl {
public:
  // Constructor that follows pytorch-npu pattern
  explicit MycelyaTensorImpl(
    const c10::Storage& storage, 
    const caffe2::TypeMeta& data_type);

  // Override shallow copy methods for proper view semantics
  void shallow_copy_from(const c10::intrusive_ptr<c10::TensorImpl>& impl) final;
  
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const final;
  
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const final;

  // Get storage ID efficiently without casting data pointer
  storage_id_t get_storage_id() const;
  
  // Simple verification method to test custom behavior
  void mark_accessed() const;
  bool was_accessed_via_custom_impl() const;

private:
  // Simple verification flag to test custom behavior
  mutable bool accessed_via_custom_impl_ = false;
};

// Factory function to create tensors with custom MycelyaTensorImpl
at::Tensor make_mycelya_tensor_with_custom_impl(
  const c10::Storage& storage,
  const caffe2::TypeMeta& data_type);

// Factory function to create custom storage impl
c10::intrusive_ptr<c10::StorageImpl> make_mycelya_storage_impl(
  c10::StorageImpl::use_byte_size_t use_byte_size,
  c10::SymInt size_bytes,
  c10::DataPtr data_ptr,
  c10::Allocator* allocator,
  bool resizable);

} // namespace mycelya