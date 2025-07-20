// Copyright (C) 2025 alyxya
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Remote.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace remote {
namespace {

// Python driver_exec function for direct operation execution
PyObject* py_driver_exec;


static c10::DeviceIndex device_count() {
  py::gil_scoped_acquire acquire;
  return get_method("deviceCount")().cast<c10::DeviceIndex>();
}

static c10::DeviceIndex current_device_idx() {
  py::gil_scoped_acquire acquire;
  return get_method("getDevice")().cast<c10::DeviceIndex>();
}

class RemoteGeneratorImpl : public at::CPUGeneratorImpl {
 public:
  RemoteGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~RemoteGeneratorImpl() override = default;
};

static at::Generator make_remote_generator(c10::DeviceIndex device_index) {
  return at::make_generator<RemoteGeneratorImpl>(device_index);
}

// Default, global generators, one per device.
static std::vector<at::Generator> default_generators;

struct RemoteHooksInterface : public at::PrivateUse1HooksInterface {
  RemoteHooksInterface() {};
  ~RemoteHooksInterface() override = default;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    py::gil_scoped_acquire acquire;
    return get_method("hasPrimaryContext")(device_index).cast<bool>();
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    TORCH_CHECK(false, "Pinned memory is not supported for remote tensors");
  }

  bool isPinnedPtr(const void* data) const override {
    return false;
  }

  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    static bool flag [[maybe_unused]] = []() {
      auto device_nums = device_count();
      default_generators.resize(device_nums);
      for (auto i = 0; i < device_nums; i++) {
        default_generators[i] = make_remote_generator(i);
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
    return make_remote_generator(device_index);
  }

  void resizePrivateUse1Bytes(
      const c10::Storage& storage,
      size_t new_bytes) const override {
    size_t old_bytes = storage.nbytes();
    
    // If expanding storage, we need to resize remotely first
    if (new_bytes > old_bytes) {
      // Get storage ID from the data pointer
      storage_id_t storage_id = reinterpret_cast<storage_id_t>(storage.data_ptr().get());
      
      try {
        py::gil_scoped_acquire acquire;
        
        // Call Python function to resize remote storage
        // For simplicity, assume float32 (4 bytes per element) - this could be enhanced
        size_t new_elements = new_bytes / 4;  // Assume float32
        py::list new_shape = py::cast(std::vector<int64_t>{static_cast<int64_t>(new_elements)});
        
        // Call resize_storage on the appropriate backend
        auto resize_result = get_method("resize_storage_by_id")(storage_id, new_shape, "float32");
        bool success = resize_result.cast<bool>();
        
        if (!success) {
          // Continue with local update anyway to avoid crash
          // TODO: Consider throwing an exception here in production
        }
      } catch (const std::exception& e) {
        // Continue with local update anyway to avoid crash
        // TODO: Consider throwing an exception here in production
      }
    }
    
    // Update the local storage's internal size tracking
    const_cast<c10::Storage&>(storage).unsafeGetStorageImpl()->set_nbytes(new_bytes);
  }
};

// Hooks will be registered explicitly from remote_extension.cpp

// Device guard registration
struct RemoteGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  RemoteGuardImpl() = default;
  explicit RemoteGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == static_type);
  }

  c10::DeviceType type() const override {
    return static_type;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    auto old_device_index =
        get_method("exchangeDevice")(d.index()).cast<c10::DeviceIndex>();
    return c10::Device(static_type, old_device_index);
  }

  c10::Device getDevice() const override {
    return c10::Device(static_type, current_device_idx());
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    auto device = get_method("setDevice")(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto device = get_method("uncheckedSetDevice")(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto stream_id = get_method("getStream")(d.index()).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    py::gil_scoped_acquire acquire;
    return get_method("getDefaultStream")(d.index()).cast<c10::Stream>();
  }

  c10::Stream getStreamFromGlobalPool(
      c10::Device d,
      bool isHighPriority = false) const override {
    py::gil_scoped_acquire acquire;
    return get_method("getStreamFromGlobalPool")(d.index(), isHighPriority)
        .cast<c10::Stream>();
  }

  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    py::gil_scoped_acquire acquire;
    auto stream_id =
        get_method("getNewStream")(d.index(), priority).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto stream_id = get_method("exchangeStream")(s).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, s.device(), stream_id);
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    py::gil_scoped_acquire acquire;
    get_method("destroyEvent")((int64_t)event, device_index);
  }

  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
    py::gil_scoped_acquire acquire;
    get_method("record")((int64_t)event, stream, device_index, (int64_t)flag);
  }

  void block(void* event, const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    get_method("block")((int64_t)event, stream);
  }

  bool queryEvent(void* event) const override {
    py::gil_scoped_acquire acquire;
    return get_method("queryEvent")((int64_t)event).cast<bool>();
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  bool queryStream(const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    return get_method("queryStream")(stream).cast<bool>();
  }

  virtual void synchronizeStream(const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    get_method("synchronizeStream")(stream);
  }

  void synchronizeEvent(void* event) const override {
    py::gil_scoped_acquire acquire;
    get_method("synchronizeEvent")((int64_t)event);
  }

  void recordDataPtrOnStream(
      const c10::DataPtr& data_ptr,
      const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    get_method("recordDataPtrOnStream")(data_ptr, stream);
  }

  double elapsedTime(
      void* event1,
      void* event2,
      const c10::DeviceIndex device_index) const override {
    py::gil_scoped_acquire acquire;
    return get_method("elapsedTime")(
               (int64_t)event1, (int64_t)event2, device_index)
        .cast<double>();
  }
};

// Register our device guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, RemoteGuardImpl);

} // namespace

// Function to explicitly register hooks - called from Python extension init
void register_remote_hooks() {
  static bool hooks_registered = false;
  if (!hooks_registered) {
    at::RegisterPrivateUse1HooksInterface(new RemoteHooksInterface());
    hooks_registered = true;
  }
}

// Setter for the python driver_exec function
void set_driver_exec(PyObject* driver_exec_fn) {
  py_driver_exec = driver_exec_fn;
}

py::function get_method(const char* name) {
  auto driver_exec = py::cast<py::function>(py_driver_exec);
  // Create a function that calls driver_exec with the operation name
  return py::cpp_function([driver_exec, name](py::args args) {
    // Call driver_exec(name, *args)
    py::tuple args_with_name = py::make_tuple(py::str(name)) + args;
    return driver_exec(*args_with_name);
  });
}

} // namespace remote