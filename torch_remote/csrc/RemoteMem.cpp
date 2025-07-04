#include "Remote.h"

#include <c10/core/Allocator.h>

namespace remote {
namespace {

struct RemoteAllocator final : at::Allocator {
  RemoteAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    auto curr_device_idx = get_method("getDevice")().cast<c10::DeviceIndex>();
    auto curr_device =
        c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
    void* data = nullptr;
    if (nbytes > 0) {
      data = reinterpret_cast<void*>(
          get_method("malloc")(nbytes).cast<remote_ptr_t>());
      TORCH_CHECK(
          data, "Failed to allocator ", nbytes, " bytes on remote device.");
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


} // namespace remote