// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/utils/pybind.h>

#include <random>
#include <string>

namespace mycelya {

using mycelya_ptr_t = uint64_t;
using storage_id_t = uint64_t; // Changed from string to integer for efficient
                               // storage as data pointer

void set_impl_factory(PyObject *factory);
py::function get_method(const char *name);

// C++ tensor creation functions
at::Tensor empty_mycelya(at::IntArrayRef size,
                         c10::optional<at::ScalarType> dtype,
                         c10::optional<at::Layout> layout,
                         c10::optional<at::Device> device,
                         c10::optional<bool> pin_memory,
                         c10::optional<at::MemoryFormat> memory_format);

at::Tensor empty_strided_mycelya(at::IntArrayRef size, at::IntArrayRef stride,
                                 c10::optional<at::ScalarType> dtype,
                                 c10::optional<at::Layout> layout,
                                 c10::optional<at::Device> device,
                                 c10::optional<bool> pin_memory);

} // namespace mycelya