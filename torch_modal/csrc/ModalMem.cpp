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


} // namespace

// Register ATEN implementations
// TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
//   m.impl("empty.memory_format", empty_modal);
//   m.impl("empty_strided", empty_strided_modal);
// }

} // namespace modal