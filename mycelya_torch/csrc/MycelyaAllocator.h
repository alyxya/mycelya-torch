// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <c10/core/Allocator.h>

namespace mycelya {

using storage_id_t = uint64_t;

// ID-based allocator that stores storage IDs as data pointers
struct MycelyaAllocator final : at::Allocator {
  MycelyaAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override;

  static void ReportAndDelete(void *ptr);

  at::DeleterFnPtr raw_deleter() const override;

  void copy_data(void *dest, const void *src, std::size_t count) const final;
};

// Get the global mycelya allocator instance
MycelyaAllocator &get_mycelya_allocator();

} // namespace mycelya