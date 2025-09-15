#!/usr/bin/env python3
"""
Example demonstrating the mycelya-torch remote execution system with custom pickle.

This example shows how to use the @remote() decorator to execute functions
remotely on mycelya tensors, with automatic machine inference and tensor handling.
"""

import torch

import mycelya_torch
import mycelya_torch._utils  # For get_storage_id function

# Create a remote machine
print("Creating remote machine...")
machine = mycelya_torch.RemoteMachine("mock")  # Using mock for local testing
device = machine.device("cpu")

print(f"Remote device: {device}")
print(f"Machine ID: {machine.machine_id}")


# Example 1: Simple matrix operations
@mycelya_torch.remote()
def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two matrices remotely."""
    print(f"Executing on remote machine - a.shape: {a.shape}, b.shape: {b.shape}")
    result = torch.matmul(a, b)
    print(f"Result computed, shape: {result.shape}")
    return result


# Example 2: More complex function with multiple operations
@mycelya_torch.remote()
def complex_computation(x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    """Perform a complex computation involving multiple tensor operations."""
    print(f"Starting complex computation, scale: {scale}")

    # Multiple operations that will all execute remotely
    y = x * scale
    z = torch.relu(y)
    w = torch.softmax(z, dim=-1)

    # Sum reduction
    result = w.sum(dim=0)

    print(f"Complex computation complete, result shape: {result.shape}")
    return result


# Example 3: Function returning multiple tensors
@mycelya_torch.remote()
def split_and_process(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split tensor and process each half differently."""
    print(f"Processing tensor with shape {tensor.shape}")

    mid = tensor.shape[0] // 2
    first_half = tensor[:mid]
    second_half = tensor[mid:]

    # Process each half differently
    processed_first = torch.tanh(first_half)
    processed_second = torch.sigmoid(second_half)

    return processed_first, processed_second


# Example 4: Function with device manipulation
@mycelya_torch.remote()
def device_aware_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Function that's aware of the device it's running on."""
    print(f"Is CUDA available in remote context: {torch.cuda.is_available()}")

    # Create a new tensor on the same device
    c = torch.ones_like(a)

    # Perform computation
    result = (a + b) * c

    return result


def main():
    """Run the examples."""
    print("\n" + "=" * 60)
    print("MYCELYA-TORCH REMOTE EXECUTION EXAMPLES")
    print("=" * 60)

    # Create test tensors on the remote device
    print("\n1. Creating test tensors...")

    # Create tensors locally first, then move to remote device
    # This ensures the remote storage is properly allocated
    a_cpu = torch.randn(4, 4)
    b_cpu = torch.randn(4, 4)
    x_cpu = torch.randn(8, 5)

    print("Moving tensors to remote device...")
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    x = x_cpu.to(device)

    print(f"Created tensor a: shape={a.shape}, device={a.device}")
    print(f"Created tensor b: shape={b.shape}, device={b.device}")
    print(f"Created tensor x: shape={x.shape}, device={x.device}")

    # Let's also verify the tensors are properly stored remotely
    print(f"Tensor a storage ID: {mycelya_torch._utils.get_storage_id(a)}")
    print(f"Tensor b storage ID: {mycelya_torch._utils.get_storage_id(b)}")
    print(f"Tensor x storage ID: {mycelya_torch._utils.get_storage_id(x)}")

    # Example 1: Matrix multiplication
    print("\n" + "-" * 40)
    print("EXAMPLE 1: Matrix Multiplication")
    print("-" * 40)

    try:
        result1 = matrix_multiply(a, b)
        print("✅ Matrix multiplication successful!")
        print(f"   Result shape: {result1.shape}")
        print(f"   Result device: {result1.device}")
        print(f"   Result sample values: {result1.flatten()[:3]}")
    except Exception as e:
        print(f"❌ Matrix multiplication failed: {e}")

    # Example 2: Complex computation
    print("\n" + "-" * 40)
    print("EXAMPLE 2: Complex Computation")
    print("-" * 40)

    try:
        result2 = complex_computation(x, scale=3.0)
        print("✅ Complex computation successful!")
        print(f"   Result shape: {result2.shape}")
        print(f"   Result device: {result2.device}")
        print(f"   Result sample values: {result2.flatten()[:3]}")
    except Exception as e:
        print(f"❌ Complex computation failed: {e}")

    # Example 3: Multiple return values
    print("\n" + "-" * 40)
    print("EXAMPLE 3: Multiple Return Values")
    print("-" * 40)

    try:
        first, second = split_and_process(x)
        print("✅ Split and process successful!")
        print(f"   First result shape: {first.shape}, device: {first.device}")
        print(f"   Second result shape: {second.shape}, device: {second.device}")
        print(f"   First sample: {first.flatten()[:3]}")
        print(f"   Second sample: {second.flatten()[:3]}")
    except Exception as e:
        print(f"❌ Split and process failed: {e}")

    # Example 4: Device-aware function
    print("\n" + "-" * 40)
    print("EXAMPLE 4: Device-Aware Function")
    print("-" * 40)

    try:
        result4 = device_aware_function(a, b)
        print("✅ Device-aware function successful!")
        print(f"   Result shape: {result4.shape}")
        print(f"   Result device: {result4.device}")
        print(f"   Result sample values: {result4.flatten()[:3]}")
    except Exception as e:
        print(f"❌ Device-aware function failed: {e}")

    # Demonstrate machine inference
    print("\n" + "-" * 40)
    print("MACHINE INFERENCE DEMONSTRATION")
    print("-" * 40)

    # Show that the decorator automatically inferred the machine
    print(f"All operations executed on machine: {machine.machine_id}")
    print(f"Machine provider: {machine.provider}")
    print(f"Machine GPU type: {machine.gpu_type}")

    # Test error handling - mixed machines (this would fail in real scenario)
    print("\n" + "-" * 40)
    print("ERROR HANDLING EXAMPLE")
    print("-" * 40)

    print("Note: The following would fail if tensors were from different machines:")
    print("@remote()")
    print("def mixed_machine_function(tensor1, tensor2):")
    print("    return tensor1 + tensor2")
    print("")
    print("# If tensor1 and tensor2 were from different RemoteMachines,")
    print("# this would raise a RuntimeError about mixed machines")


if __name__ == "__main__":
    main()

    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✅ Automatic machine inference from tensor arguments")
    print("✅ Remote function execution with source code transfer")
    print("✅ Proper tensor serialization/deserialization")
    print("✅ Device preservation across remote boundaries")
    print("✅ Multiple return value handling")
    print("✅ Complex tensor operations")
    print("✅ Error handling and validation")
