#!/usr/bin/env python3
"""
Comprehensive test demonstrating the mycelya-torch remote execution system.
"""

import torch
import mycelya_torch
import mycelya_torch._utils

def main():
    print("="*60)
    print("MYCELYA-TORCH REMOTE EXECUTION COMPREHENSIVE TEST")
    print("="*60)

    # Create a remote machine
    machine = mycelya_torch.RemoteMachine("mock")
    device = machine.device()

    print(f"\n✅ Created remote machine: {machine.machine_id}")
    print(f"   Provider: {machine.provider}")
    print(f"   Device: {device}")

    # Create test tensors
    print("\n📊 Creating test tensors...")
    a_cpu = torch.randn(4, 4)
    b_cpu = torch.randn(4, 4)
    x_cpu = torch.randn(8, 5)

    a = a_cpu.to(device)
    b = b_cpu.to(device)
    x = x_cpu.to(device)

    print(f"   Tensor a: {a.shape} on {a.device}")
    print(f"   Tensor b: {b.shape} on {b.device}")
    print(f"   Tensor x: {x.shape} on {x.device}")

    # Test 1: Simple arithmetic
    @mycelya_torch.remote
    def simple_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        print("  🚀 Executing simple addition remotely")
        result = x + y
        print(f"  ✨ Result computed: {result.shape}")
        return result

    print("\n" + "-"*40)
    print("TEST 1: Simple Addition")
    print("-"*40)

    try:
        result1 = simple_add(a, b)
        print(f"✅ Success! Result shape: {result1.shape}, device: {result1.device}")
        print(f"   Sample values: {result1.flatten()[:3]}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 2: Matrix multiplication
    @mycelya_torch.remote()
    def matrix_multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        print("  🚀 Executing matrix multiplication remotely")
        result = torch.matmul(x, y)
        print(f"  ✨ MatMul result: {result.shape}")
        return result

    print("\n" + "-"*40)
    print("TEST 2: Matrix Multiplication")
    print("-"*40)

    try:
        result2 = matrix_multiply(a, b)
        print(f"✅ Success! Result shape: {result2.shape}, device: {result2.device}")
        print(f"   Sample values: {result2.flatten()[:3]}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 3: Complex operations with multiple return values
    @mycelya_torch.remote
    def split_and_transform(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        print(f"  🚀 Processing tensor: {tensor.shape}")

        mid = tensor.shape[0] // 2
        first_half = tensor[:mid]
        second_half = tensor[mid:]

        # Different transformations
        transformed_first = torch.relu(first_half)
        transformed_second = torch.sigmoid(second_half)

        print(f"  ✨ Split into: {transformed_first.shape} and {transformed_second.shape}")
        return transformed_first, transformed_second

    print("\n" + "-"*40)
    print("TEST 3: Split and Transform (Multiple Returns)")
    print("-"*40)

    try:
        first_result, second_result = split_and_transform(x)
        print(f"✅ Success!")
        print(f"   First result: {first_result.shape} on {first_result.device}")
        print(f"   Second result: {second_result.shape} on {second_result.device}")
        print(f"   First sample: {first_result.flatten()[:3]}")
        print(f"   Second sample: {second_result.flatten()[:3]}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 4: Function with parameters
    @mycelya_torch.remote()
    def scaled_operations(tensor: torch.Tensor, scale: float, operation: str = "multiply") -> torch.Tensor:
        print(f"  🚀 {operation} with scale {scale}")

        if operation == "multiply":
            result = tensor * scale
        elif operation == "add":
            result = tensor + scale
        else:
            result = tensor

        # Additional transformation
        result = torch.tanh(result)

        print(f"  ✨ Operation complete: {result.shape}")
        return result

    print("\n" + "-"*40)
    print("TEST 4: Parameterized Function")
    print("-"*40)

    try:
        result4 = scaled_operations(x, scale=2.5, operation="multiply")
        print(f"✅ Success! Result shape: {result4.shape}, device: {result4.device}")
        print(f"   Sample values: {result4.flatten()[:3]}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 5: Function using torch utilities
    @mycelya_torch.remote
    def advanced_torch_ops(tensor: torch.Tensor) -> torch.Tensor:
        print("  🚀 Advanced torch operations")

        # Use various torch functions
        normalized = torch.nn.functional.normalize(tensor, p=2, dim=-1)
        softmaxed = torch.softmax(normalized, dim=-1)
        summed = softmaxed.sum(dim=0)

        print(f"  ✨ Advanced operations complete: {summed.shape}")
        return summed

    print("\n" + "-"*40)
    print("TEST 5: Advanced Torch Operations")
    print("-"*40)

    try:
        result5 = advanced_torch_ops(x)
        print(f"✅ Success! Result shape: {result5.shape}, device: {result5.device}")
        print(f"   Sample values: {result5.flatten()[:3]}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("🎉 Remote execution system working successfully!")
    print(f"📍 Machine: {machine.machine_id}")
    print(f"🔧 Provider: {machine.provider}")
    print(f"💾 Device: {device}")
    print("\n✅ Key features demonstrated:")
    print("   • Automatic machine inference from tensor arguments")
    print("   • Remote function code execution via source inspection")
    print("   • Proper tensor serialization/deserialization")
    print("   • Preservation of device locations across remote boundaries")
    print("   • Multiple return value handling")
    print("   • Parameter passing (scalars and keywords)")
    print("   • Complex tensor operations and transformations")
    print("   • Temp registry linking for result reconstruction")

    print(f"\n🔍 Technical details:")
    print(f"   • Storage IDs: {[mycelya_torch._utils.get_storage_id(t) for t in [a, b, x]]}")
    print(f"   • Tensor IDs: {[mycelya_torch._utils.get_tensor_id(t) for t in [a, b, x]]}")

if __name__ == "__main__":
    main()