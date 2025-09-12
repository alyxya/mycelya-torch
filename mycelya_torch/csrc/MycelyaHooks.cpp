// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include "Mycelya.h"
#include "MycelyaStorageImpl.h"

namespace mycelya {
namespace {

// Python factory function for method implementations
PyObject *py_factory;

static c10::DeviceIndex device_count() {
  py::gil_scoped_acquire acquire;
  return get_method("device_count")().cast<c10::DeviceIndex>();
}

static c10::DeviceIndex current_device_idx() {
  py::gil_scoped_acquire acquire;
  return get_method("get_device")().cast<c10::DeviceIndex>();
}

class MycelyaGeneratorImpl : public at::CPUGeneratorImpl {
public:
  MycelyaGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~MycelyaGeneratorImpl() override = default;
};

static at::Generator make_mycelya_generator(c10::DeviceIndex device_index) {
  return at::make_generator<MycelyaGeneratorImpl>(device_index);
}

// Default, global generators, one per device.
static std::vector<at::Generator> default_generators;

struct MycelyaHooksInterface : public at::PrivateUse1HooksInterface {
  MycelyaHooksInterface() {};
  ~MycelyaHooksInterface() override = default;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    py::gil_scoped_acquire acquire;
    return get_method("has_primary_context")(device_index).cast<bool>();
  }

  at::Allocator *getPinnedMemoryAllocator() const override {
    TORCH_CHECK(false, "Pinned memory is not supported for remote tensors");
  }

  bool isPinnedPtr(const void *data) const override { return false; }

  const at::Generator &
  getDefaultGenerator(c10::DeviceIndex device_index) const override {
    static bool flag [[maybe_unused]] = []() {
      auto device_nums = device_count();
      default_generators.resize(device_nums);
      for (auto i = 0; i < device_nums; i++) {
        default_generators[i] = make_mycelya_generator(i);
        default_generators[i].seed();
      }
      return true;
    }();

    c10::DeviceIndex idx = device_index;
    if (idx == -1) {
      idx = current_device_idx();
    } else {
      TORCH_CHECK(idx >= 0 && idx < device_count());
    }
    return default_generators[idx];
  }

  at::Generator getNewGenerator(c10::DeviceIndex device_index) const override {
    return make_mycelya_generator(device_index);
  }

  void resizePrivateUse1Bytes(const c10::Storage &storage,
                              size_t new_bytes) const override {
    py::gil_scoped_acquire acquire;

    size_t old_bytes = storage.nbytes();

    // If expanding storage, update local storage size first, then notify remote
    if (new_bytes > old_bytes) {
      // Get storage ID from the data pointer
      storage_id_t storage_id =
          reinterpret_cast<storage_id_t>(storage.data_ptr().get());

      // Update the local storage's internal size tracking first
      const_cast<c10::Storage &>(storage).unsafeGetStorageImpl()->set_nbytes(
          new_bytes);

      // Call Python function to resize remote storage with new byte count
      get_method("resize_storage")(storage_id, static_cast<int64_t>(new_bytes));
    }
  }
};

static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new MycelyaHooksInterface());

  // Register custom storage factory function (following pytorch-npu pattern)
  // This enables PyTorch to create custom MycelyaStorageImpl instances
  // when creating storages for PrivateUse1 device
  c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1,
                            &make_mycelya_storage_impl);

  return true;
}();

// Device guard registration
struct RemoteGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  RemoteGuardImpl() = default;
  explicit RemoteGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == static_type);
  }

  c10::DeviceType type() const override { return static_type; }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    auto old_device_index =
        get_method("exchange_device")(d.index()).cast<c10::DeviceIndex>();
    return c10::Device(static_type, old_device_index);
  }

  c10::Device getDevice() const override {
    return c10::Device(static_type, current_device_idx());
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    get_method("set_device")(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    get_method("unchecked_set_device")(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto stream_id = get_method("get_stream")(d.index()).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    // Default stream is always stream ID 0
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
  }

  c10::Stream
  getStreamFromGlobalPool(c10::Device d,
                          bool isHighPriority = false) const override {
    // Default to stream ID 0 like getDefaultStream
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
  }

  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    py::gil_scoped_acquire acquire;
    auto stream_id =
        get_method("get_new_stream")(d.index(), priority).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto previous_stream_id =
        get_method("exchange_stream")(s.id(), s.device().index())
            .cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, s.device(), previous_stream_id);
  }

  void createEvent(void **event, const c10::DeviceIndex device_index,
                   const c10::EventFlag flag) const {
    py::gil_scoped_acquire acquire;
    auto event_id =
        get_method("create_event")(device_index, (int64_t)flag).cast<int64_t>();
    *event = reinterpret_cast<void *>(event_id);
  }

  void
  destroyEvent(void *event,
               const c10::DeviceIndex device_index) const noexcept override {
    py::gil_scoped_acquire acquire;
    get_method("destroy_event")((int64_t)event, device_index);
  }

  void record(void **event, const c10::Stream &stream,
              const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override {
    py::gil_scoped_acquire acquire;
    get_method("record")((int64_t)event, stream, device_index, (int64_t)flag);
  }

  void block(void *event, const c10::Stream &stream) const override {
    py::gil_scoped_acquire acquire;
    get_method("block")((int64_t)event, stream);
  }

  bool queryEvent(void *event) const override {
    py::gil_scoped_acquire acquire;
    return get_method("query_event")((int64_t)event).cast<bool>();
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  bool queryStream(const c10::Stream &stream) const override {
    py::gil_scoped_acquire acquire;
    return get_method("query_stream")(stream).cast<bool>();
  }

  virtual void synchronizeStream(const c10::Stream &stream) const override {
    py::gil_scoped_acquire acquire;
    get_method("synchronize_stream")(stream);
  }

  void synchronizeEvent(void *event) const override {
    py::gil_scoped_acquire acquire;
    get_method("synchronize_event")((int64_t)event);
  }

  void recordDataPtrOnStream(const c10::DataPtr &data_ptr,
                             const c10::Stream &stream) const override {
    py::gil_scoped_acquire acquire;
    // Convert DataPtr to int64_t to avoid pybind11 registration issues
    get_method("record_data_ptr_on_stream")(
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(data_ptr.get())),
        stream);
  }

  double elapsedTime(void *event1, void *event2,
                     const c10::DeviceIndex device_index) const override {
    py::gil_scoped_acquire acquire;
    return get_method("elapsed_time")((int64_t)event1, (int64_t)event2,
                                      device_index)
        .cast<double>();
  }
};

// Register our device guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, RemoteGuardImpl);

} // namespace

// Setter for the python factory function
void set_impl_factory(PyObject *factory) { py_factory = factory; }

py::function get_method(const char *name) {
  auto factory = py::cast<py::function>(py_factory);
  return factory(name);
}

} // namespace mycelya
