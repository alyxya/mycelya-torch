// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Remote.h"

#include <c10/core/Allocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <torch/library.h>
#include <random>
#include <sstream>
#include <iomanip>

namespace remote {
namespace {

// ID-based allocator that stores tensor IDs as data pointers
struct RemoteAllocator final : at::Allocator {
  RemoteAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    auto curr_device_idx = get_method("getDevice")().cast<c10::DeviceIndex>();
    auto curr_device = c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
    void* data = nullptr;
    
    if (nbytes > 0) {
      // Generate a unique tensor ID
      tensor_id_t tensor_id = generate_tensor_id();
      
      // Call Python method to create tensor with ID and register it
      // This should create the tensor remotely and return success/failure
      bool success = get_method(kCreateTensorMethod)(tensor_id, nbytes, curr_device_idx).cast<bool>();
      
      TORCH_CHECK(success, "Failed to allocate tensor with ID ", tensor_id, 
                  " (", nbytes, " bytes) on remote device ", curr_device_idx);
      
      // Store the tensor ID as the data pointer
      data = reinterpret_cast<void*>(tensor_id);
    }
    
    return {data, data, &ReportAndDelete<kFreeTensorMethod>, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete<kFreeTensorMethod>;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    // Convert data pointers back to tensor IDs for the copy operation
    tensor_id_t dest_id = reinterpret_cast<tensor_id_t>(dest);
    tensor_id_t src_id = reinterpret_cast<tensor_id_t>(src);
    get_method("copy_data_by_id")(dest_id, src_id, count);
  }
};

static RemoteAllocator global_remote_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_remote_alloc);

} // namespace

// Utility function to generate unique tensor IDs
tensor_id_t generate_tensor_id() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<uint64_t> dis;
  
  // Generate a unique 64-bit integer ID
  // Use non-zero values to avoid confusion with null pointers
  tensor_id_t id;
  do {
    id = dis(gen);
  } while (id == 0);
  
  return id;
}

// Validate device index
bool validate_device_index(c10::DeviceIndex device_index) {
  py::gil_scoped_acquire acquire;
  try {
    auto device_count = get_method("deviceCount")().cast<c10::DeviceIndex>();
    return device_index >= 0 && device_index < device_count;
  } catch (...) {
    return false;
  }
}

// C++ implementation of empty_remote using direct allocator integration
at::Tensor empty_remote(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {
  
  // Handle device resolution
  c10::Device target_device = device.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));
  if (target_device.type() != c10::DeviceType::PrivateUse1) {
    target_device = c10::Device(c10::DeviceType::PrivateUse1, target_device.index());
  }
  
  // Validate device index
  TORCH_CHECK(validate_device_index(target_device.index()), 
              "Invalid device index: ", target_device.index());
  
  // Handle other parameters
  auto resolved_dtype = dtype.value_or(at::get_default_dtype_as_scalartype());
  auto resolved_layout = layout.value_or(at::Layout::Strided);
  auto resolved_memory_format = memory_format.value_or(at::MemoryFormat::Contiguous);
  
  TORCH_CHECK(resolved_layout == at::Layout::Strided, "Only strided layout is supported");
  TORCH_CHECK(!pin_memory.value_or(false), "Pin memory is not supported on remote devices");
  
  // Set device guard to ensure allocation happens on correct device
  const c10::DeviceGuard device_guard(target_device);
  
  // Use the enhanced allocator to create tensor
  constexpr c10::DispatchKeySet remote_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(
      size, &global_remote_alloc, remote_dks, resolved_dtype, resolved_memory_format);
}

// C++ implementation of empty_strided_remote
at::Tensor empty_strided_remote(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  
  // Handle device resolution
  c10::Device target_device = device.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));
  if (target_device.type() != c10::DeviceType::PrivateUse1) {
    target_device = c10::Device(c10::DeviceType::PrivateUse1, target_device.index());
  }
  
  // Validate device index
  TORCH_CHECK(validate_device_index(target_device.index()), 
              "Invalid device index: ", target_device.index());
  
  // Handle other parameters
  auto resolved_dtype = dtype.value_or(at::get_default_dtype_as_scalartype());
  auto resolved_layout = layout.value_or(at::Layout::Strided);
  
  TORCH_CHECK(resolved_layout == at::Layout::Strided, "Only strided layout is supported");
  TORCH_CHECK(!pin_memory.value_or(false), "Pin memory is not supported on remote devices");
  
  // Set device guard to ensure allocation happens on correct device
  const c10::DeviceGuard device_guard(target_device);
  
  // Use the enhanced allocator to create strided tensor
  constexpr c10::DispatchKeySet remote_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, &global_remote_alloc, remote_dks, resolved_dtype);
}


// Register the C++ implementations directly with PyTorch's dispatch system
// This follows the OpenReg pattern where empty operations are implemented in C++
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // Register our C++ implementations for empty tensor creation
  // These will override the Python fallback for these specific operations
  m.impl("empty.memory_format", empty_remote);
  m.impl("empty_strided", empty_strided_remote);
}

} // namespace remote