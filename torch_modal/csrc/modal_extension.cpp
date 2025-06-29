#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

// Modal tensor creation function
at::Tensor modal(const at::Tensor& self, c10::optional<c10::Device> device, 
                 c10::optional<c10::ScalarType> dtype, bool non_blocking, bool copy) {
    // Convert to modal device (PrivateUse1)
    c10::Device modal_device(c10::DeviceType::PrivateUse1, 0);
    
    // Create a CPU copy with proper dtype
    at::Tensor cpu_tensor = self.cpu();
    if (dtype.has_value()) {
        cpu_tensor = cpu_tensor.to(dtype.value());
    }
    
    // Create a tensor using from_blob to avoid triggering device allocation
    at::TensorOptions options = at::TensorOptions()
        .dtype(cpu_tensor.dtype())
        .device(modal_device);
    
    // Clone the CPU data first
    at::Tensor cpu_copy = cpu_tensor.clone();
    
    // Create modal tensor from the CPU storage but with modal device
    at::Tensor result = at::from_blob(
        cpu_copy.data_ptr(),
        cpu_copy.sizes(),
        cpu_copy.strides(),
        options
    );
    
    return result;
}

// Simple modal hooks interface
struct ModalHooksInterface : public at::PrivateUse1HooksInterface {
    ~ModalHooksInterface() override = default;
    
    void init() const override {
        // No initialization needed for modal device
    }
    
    at::Generator getNewGenerator(c10::DeviceIndex device_index = -1) const override {
        // Return CPU generator for modal device
        return at::detail::createCPUGenerator(device_index);
    }
    
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
        return false; // Modal device doesn't have primary context
    }
    
    bool isAvailable() const override {
        return true; // Modal device is always available
    }
    
    c10::Device getDeviceFromPtr(void* data) const override {
        // Modal tensors use CPU storage, so return modal device for any pointer
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }
};

// Get modal hooks instance
at::PrivateUse1HooksInterface* get_modal_hooks() {
    static ModalHooksInterface modal_hooks;
    return &modal_hooks;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("modal", &modal, "Move tensor to modal device");
    
    // Register the modal hooks interface
    at::RegisterPrivateUse1HooksInterface(get_modal_hooks());
}

// Modal CPU fallback function
void modal_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    at::native::cpu_fallback(op, stack);
}

// Register CPU fallback for main modal backend
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&modal_cpu_fallback>());
}

// Use fallthrough for autograd (like torch_npu does)
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}