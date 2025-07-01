#include "Modal.h"

#include <ATen/EmptyTensor.h>
#include <c10/core/Allocator.h>
#include <torch/library.h>

namespace modal {
namespace {

struct ModalAllocator final : at::Allocator {
  ModalAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    auto curr_device_idx = get_method("getDevice")().cast<c10::DeviceIndex>();
    auto curr_device =
        c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
    void* data = nullptr;
    if (nbytes > 0) {
      data = reinterpret_cast<void*>(
          get_method("malloc")(nbytes).cast<modal_ptr_t>());
      TORCH_CHECK(
          data, "Failed to allocator ", nbytes, " bytes on modal device.");
    }
    return {data, data, &ReportAndDelete<kFreeMethod>, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete<kFreeMethod>;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    get_method("copy_data")(
        reinterpret_cast<modal_ptr_t>(dest),
        reinterpret_cast<modal_ptr_t>(src),
        count);
  }
};

static ModalAllocator global_modal_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_modal_alloc);

// Empty op needs C++ code and cannot be handled by python side fallback
at::Tensor empty_modal(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(
      size, &global_modal_alloc, pu1_dks, dtype, memory_format_opt);
}

at::Tensor empty_strided_modal(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, &global_modal_alloc, pu1_dks, dtype);
}

} // namespace

// Register ATEN implementations
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_modal);
  m.impl("empty_strided", empty_strided_modal);
}

} // namespace modal