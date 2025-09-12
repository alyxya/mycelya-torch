// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <c10/core/StorageImpl.h>
#include <c10/util/intrusive_ptr.h>

namespace mycelya {

using storage_id_t = uint64_t;

// Custom StorageImpl that holds storage IDs and metadata for mycelya tensors
struct MycelyaStorageImpl : public c10::StorageImpl {
  explicit MycelyaStorageImpl(c10::StorageImpl::use_byte_size_t,
                              c10::SymInt size_bytes, c10::DataPtr data_ptr,
                              c10::Allocator *allocator, bool resizable);

  // Get the storage ID directly from the stored data pointer
  storage_id_t get_storage_id() const;
};

// Factory function to create custom storage impl
c10::intrusive_ptr<c10::StorageImpl>
make_mycelya_storage_impl(c10::StorageImpl::use_byte_size_t use_byte_size,
                          c10::SymInt size_bytes, c10::DataPtr data_ptr,
                          c10::Allocator *allocator, bool resizable);

} // namespace mycelya