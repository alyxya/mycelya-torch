#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/Allocator.h>

// Modal device allocator that delegates to CPU
class ModalAllocator : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t nbytes) override {
        auto cpu_allocator = c10::GetCPUAllocator();
        return cpu_allocator->allocate(nbytes);
    }
    
    c10::DeleterFnPtr raw_deleter() const override {
        auto cpu_allocator = c10::GetCPUAllocator();
        return cpu_allocator->raw_deleter();
    }
    
    void copy_data(void* dest, const void* src, std::size_t count) const override {
        std::memcpy(dest, src, count);
    }
};

// Global modal allocator instance
static ModalAllocator g_modal_allocator;

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
    
    // Clone the CPU tensor to avoid modifying original
    at::Tensor result = copy ? cpu_tensor.clone() : cpu_tensor;
    
    // Create modal tensor using from_blob with manual device override
    // This creates a tensor that reports as modal device but uses CPU memory
    at::Tensor modal_tensor = at::from_blob(
        result.data_ptr(),
        result.sizes(),
        result.strides(),
        at::TensorOptions().dtype(result.dtype()).device(modal_device)
    );
    
    // If copy was requested, clone the tensor to ensure we own the data
    if (copy) {
        // We need to be careful with clone() as it might reset device
        auto cloned = result.clone();
        modal_tensor = at::from_blob(
            cloned.data_ptr(),
            cloned.sizes(),
            cloned.strides(),
            at::TensorOptions().dtype(cloned.dtype()).device(modal_device)
        );
    }
    
    return modal_tensor;
}

// Simple modal hooks interface
struct ModalHooksInterface : public at::PrivateUse1HooksInterface {
    ~ModalHooksInterface() override = default;
    
    void init() const override {
        // Register modal device allocator
        c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_modal_allocator);
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
    
    // Critical method: resize storage for modal device
    void resizePrivateUse1Bytes(const c10::Storage& storage, size_t new_bytes) const override {
        // For modal device, we delegate to CPU storage operations
        // Since modal tensors use CPU storage underneath, we can safely resize
        auto cpu_allocator = c10::GetCPUAllocator();
        storage.unsafeGetStorageImpl()->set_nbytes(new_bytes);
        if (new_bytes > 0) {
            // Only resize if we actually need more space
            auto current_size = storage.unsafeGetStorageImpl()->nbytes();
            if (new_bytes > current_size) {
                auto new_data = cpu_allocator->allocate(new_bytes);
                // Copy existing data if any
                if (current_size > 0 && storage.data()) {
                    std::memcpy(new_data.get(), storage.data(), std::min(current_size, new_bytes));
                }
                storage.unsafeGetStorageImpl()->set_data_ptr_noswap(std::move(new_data));
            }
        }
    }
    
    // Get default generator for modal device
    const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index = -1) const override {
        static at::Generator modal_gen = at::detail::createCPUGenerator(device_index);
        return modal_gen;
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
    
    // Force immediate initialization of the modal device
    auto hooks = get_modal_hooks();
    hooks->init();
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