#include "Remote.h"

#include <c10/core/Allocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <torch/library.h>
#include <random>
#include <sstream>
#include <iomanip>

namespace remote {
namespace {

// Enhanced allocator that supports both legacy pointer-based and new ID-based allocation
struct RemoteAllocator final : at::Allocator {
  RemoteAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    auto curr_device_idx = get_method("getDevice")().cast<c10::DeviceIndex>();
    auto curr_device = c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
    void* data = nullptr;
    
    if (nbytes > 0) {
      // Try ID-based allocation first
      try {
        // Generate a unique tensor ID
        tensor_id_t tensor_id = generate_tensor_id();
        
        // Call Python method to create tensor mapping with ID, now returns pointer
        auto ptr_result = get_method(kCreateTensorMethod)(tensor_id, nbytes, curr_device_idx).cast<remote_ptr_t>();
        
        if (ptr_result != 0) {
          // Use the actual pointer returned from Python
          data = reinterpret_cast<void*>(ptr_result);
        } else {
          throw std::runtime_error("ID-based allocation returned null pointer");
        }
      } catch (...) {
        // Fallback to legacy pointer-based allocation
        data = reinterpret_cast<void*>(
            get_method("malloc")(nbytes).cast<remote_ptr_t>());
      }
      
      TORCH_CHECK(data, "Failed to allocate ", nbytes, " bytes on remote device.");
    }
    
    return {data, data, &ReportAndDelete<kFreeMethod>, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete<kFreeMethod>;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    get_method("copy_data")(
        reinterpret_cast<remote_ptr_t>(dest),
        reinterpret_cast<remote_ptr_t>(src),
        count);
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
  
  std::stringstream ss;
  ss << "tensor_" << std::hex << dis(gen) << "_" << std::hex << dis(gen);
  return ss.str();
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
at::Tensor empty_remote_cpp(
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
at::Tensor empty_strided_remote_cpp(
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

// C++ implementation for tensor device conversion (handles .cpu() calls)
at::Tensor to_device_cpp(
    const at::Tensor& self,
    at::Device device, 
    at::ScalarType dtype,
    bool non_blocking, 
    bool copy, 
    c10::optional<at::MemoryFormat> memory_format) {
  
  // If converting from remote to CPU, use the Python copy_from_device function
  if (self.device().type() == c10::DeviceType::PrivateUse1 && device.type() == c10::DeviceType::CPU) {
    py::gil_scoped_acquire acquire;
    try {
      // Call the Python copy_from_device function
      py::module_ aten_impl = py::module_::import("torch_remote._aten_impl");
      py::function copy_from_device = aten_impl.attr("copy_from_device");
      py::object result = copy_from_device(self);
      at::Tensor cpu_tensor = result.cast<at::Tensor>();
      
      // Handle dtype conversion if requested
      if (cpu_tensor.dtype() != dtype) {
        cpu_tensor = cpu_tensor.to(dtype);
      }
      
      return cpu_tensor;
    } catch (const std::exception& e) {
      TORCH_CHECK(false, "Failed to copy remote tensor to CPU: ", e.what());
    }
  }
  
  // For other device transfers, fall back to default implementation
  return self.to(device, dtype, non_blocking, copy, memory_format);
}

// C++ implementation for dtype_layout variant
at::Tensor to_dtype_layout_cpp(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype, 
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, 
    c10::optional<bool> pin_memory,
    bool non_blocking, 
    bool copy, 
    c10::optional<at::MemoryFormat> memory_format) {
  
  // If converting from remote to CPU, use our custom conversion
  if (self.device().type() == c10::DeviceType::PrivateUse1 && 
      device.has_value() && device.value().type() == c10::DeviceType::CPU) {
    
    py::gil_scoped_acquire acquire;
    try {
      // Call the Python copy_from_device function
      py::module_ aten_impl = py::module_::import("torch_remote._aten_impl");
      py::function copy_from_device = aten_impl.attr("copy_from_device");
      py::object result = copy_from_device(self);
      at::Tensor cpu_tensor = result.cast<at::Tensor>();
      
      // Handle dtype conversion if requested
      if (dtype.has_value() && cpu_tensor.dtype() != dtype.value()) {
        cpu_tensor = cpu_tensor.to(dtype.value());
      }
      
      return cpu_tensor;
    } catch (const std::exception& e) {
      TORCH_CHECK(false, "Failed to copy remote tensor to CPU: ", e.what());
    }
  }
  
  // For other device transfers, fall back to default implementation
  return self.to(dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
}

// Register the C++ implementations directly with PyTorch's dispatch system
// This follows the OpenReg pattern where empty operations are implemented in C++
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // Register our C++ implementations for empty tensor creation
  // These will override the Python fallback for these specific operations
  m.impl("empty.memory_format", empty_remote_cpp);
  m.impl("empty_strided", empty_strided_remote_cpp);
  
  // Register custom to implementations for proper .cpu() support
  m.impl("to.device", to_device_cpp);
  m.impl("to.dtype_layout", to_dtype_layout_cpp);
}

} // namespace remote