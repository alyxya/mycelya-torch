// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include "MycelyaStorageImpl.h"
#include <c10/core/TensorImpl.h>
#include <c10/util/intrusive_ptr.h>

namespace mycelya {

using tensor_id_t = uint64_t;

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
  
  // Get unique tensor ID for this tensor instance
  tensor_id_t get_tensor_id() const;
  
  // Simple verification method to test custom behavior
  void mark_accessed() const;
  bool was_accessed_via_custom_impl() const;

private:
  // Simple verification flag to test custom behavior
  mutable bool accessed_via_custom_impl_ = false;
  
  // Unique tensor ID for this tensor instance
  const tensor_id_t tensor_id_;
  
  // Static counter for generating incremental tensor IDs
  static tensor_id_t next_tensor_id();
};

// Factory function to create tensors with custom MycelyaTensorImpl
at::Tensor make_mycelya_tensor_with_custom_impl(
  const c10::Storage& storage,
  const caffe2::TypeMeta& data_type);


} // namespace mycelya