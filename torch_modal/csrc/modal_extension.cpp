#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>

// Device compatibility checking
void check_modal_device_compatibility(const at::Tensor& tensor) {
    if (tensor.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error("Modal tensor operations require all tensors to be on modal device");
    }
}

void check_modal_device_compatibility(const at::TensorList& tensors) {
    for (const auto& tensor : tensors) {
        check_modal_device_compatibility(tensor);
    }
}

// Modal tensor creation function
at::Tensor modal(const at::Tensor& self, c10::optional<c10::Device> device, 
                 c10::optional<c10::ScalarType> dtype, bool non_blocking, bool copy) {
    // Convert to modal device (PrivateUse1)
    c10::Device modal_device(c10::DeviceType::PrivateUse1, 0);
    
    // For now, just copy to CPU and mark as modal device
    at::Tensor result = self.to(modal_device, dtype, non_blocking, copy);
    return result;
}

// Modal device fallback function
void modal_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // Check if all tensors are on modal device
    auto& arguments = torch::jit::last(*stack, op.schema().arguments().size());
    
    for (const auto& arg : arguments) {
        if (arg.isTensor()) {
            check_modal_device_compatibility(arg.toTensor());
        } else if (arg.isTensorList()) {
            check_modal_device_compatibility(arg.toTensorList());
        }
    }
    
    // Fall back to CPU implementation
    at::native::cpu_fallback(op, stack);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("modal", &modal, "Move tensor to modal device");
}

// Register the modal device fallback
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&modal_fallback>());
}

// Register autograd fallback
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&modal_fallback>());
}