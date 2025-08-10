// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "MycelyaStorageImpl.h"
#include <iostream>

namespace mycelya {

// MycelyaStorageImpl implementation
MycelyaStorageImpl::MycelyaStorageImpl(
  c10::StorageImpl::use_byte_size_t use_byte_size,
  c10::SymInt size_bytes,
  c10::DataPtr data_ptr,
  c10::Allocator* allocator,
  bool resizable)
  : c10::StorageImpl(
      use_byte_size,
      size_bytes,
      std::move(data_ptr),
      allocator,
      resizable) {
  // Simple verification: log creation for debugging
  if (std::getenv("MYCELYA_DEBUG_TENSORIMPL")) {
    std::cout << "[MycelyaStorageImpl] Created with storage_id=" 
              << get_storage_id() << std::endl;
  }
}

storage_id_t MycelyaStorageImpl::get_storage_id() const {
  accessed_via_custom_impl = true;
  return reinterpret_cast<storage_id_t>(data_ptr().get());
}

c10::intrusive_ptr<c10::StorageImpl> make_mycelya_storage_impl(
  c10::StorageImpl::use_byte_size_t use_byte_size,
  c10::SymInt size_bytes,
  c10::DataPtr data_ptr,
  c10::Allocator* allocator,
  bool resizable) {
  
  // Critical: Following pytorch-npu pattern
  // If no data_ptr provided, call the allocator to get memory
  if (data_ptr.get() == nullptr) {
    data_ptr = allocator->allocate(size_bytes.as_int_unchecked());
  }
  
  return c10::make_intrusive<MycelyaStorageImpl>(
    use_byte_size,
    size_bytes, 
    std::move(data_ptr),
    allocator,
    resizable);
}

} // namespace mycelya