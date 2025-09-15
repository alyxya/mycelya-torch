# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Integration tests and debugging utilities for mycelya-torch package.

This file contains integration tests that span multiple functional areas
and debugging utilities for troubleshooting remote tensor operations.

Most basic functionality tests have been refactored into modular test files:
- test_device_management.py - Device creation, validation, registry
- test_basic_operations.py - Arithmetic operations, tensor creation
- test_view_operations.py - View, reshape, transpose operations
- test_autograd_basic.py - Basic gradient computation
- test_autograd_complex.py - Complex gradient scenarios
- test_tensor_transfers.py - CPU<->remote transfers
- test_loss_functions.py - Loss function integration
- test_error_handling.py - Error handling and validation

For comprehensive testing, run all test files together:
pytest tests/ -v
"""

import torch

# =============================================================================
# DEBUGGING AND DIAGNOSTIC TESTS
# =============================================================================


def test_basic_tensor_creation_debug(shared_machines, provider):
    """Debug function: Basic tensor creation with detailed output."""
    print("\n=== Debug: Basic Tensor Creation ===")

    # Test basic tensor creation
    x = torch.randn(2, 2)
    print(f"Created CPU tensor: {x.shape}, {x.dtype}, device: {x.device}")

    # Transfer to remote
    device_type = "cpu" if provider == "mock" else "cuda"
    x_remote = x.to(shared_machines["T4"].device(device_type))
    print(
        f"Remote tensor: {x_remote.shape}, {x_remote.dtype}, device: {x_remote.device}"
    )

    # Basic operation
    y_remote = x_remote + x_remote
    print(f"Operation result: {y_remote.shape}, device: {y_remote.device}")

    # Transfer back
    y_cpu = y_remote.cpu()
    print(f"Back to CPU: {y_cpu.shape}, device: {y_cpu.device}")

    # Verify numerical consistency
    expected = x + x
    assert torch.allclose(y_cpu, expected, rtol=1e-4, atol=1e-6)
    print("✓ Numerical consistency verified")


def test_long_dtype_debug(shared_machines, provider):
    """Debug function: Long dtype handling."""
    print("\n=== Debug: Long Dtype ===")

    try:
        # Create long tensor
        targets = torch.randint(0, 3, (5,), dtype=torch.long)
        print(f"CPU long tensor: {targets.shape}, {targets.dtype}")

        # Transfer to remote
        device_type = "cpu" if provider == "mock" else "cuda"
        targets_remote = targets.to(shared_machines["T4"].device(device_type))
        print(f"Remote long tensor: {targets_remote.shape}, {targets_remote.dtype}")

        # Transfer back
        targets_back = targets_remote.cpu()
        print(f"Back to CPU: {targets_back.shape}, {targets_back.dtype}")

        # Verify consistency
        assert torch.equal(targets, targets_back)
        print("✓ Long dtype handling works correctly")

    except Exception as e:
        print(f"Long dtype not supported: {e}")


def test_cross_entropy_dtype_debug(shared_machines, provider):
    """Debug function: Cross entropy with dtype analysis."""
    print("\n=== Debug: Cross Entropy Dtypes ===")

    batch_size = 3
    num_classes = 4

    # Create input and targets
    inputs = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    print(f"Inputs: {inputs.shape}, {inputs.dtype}")
    print(f"Targets: {targets.shape}, {targets.dtype}")

    # Transfer to remote
    device_type = "cpu" if provider == "mock" else "cuda"
    inputs_remote = inputs.to(shared_machines["T4"].device(device_type))
    targets_remote = targets.to(shared_machines["T4"].device(device_type))

    print(f"Remote inputs: {inputs_remote.shape}, {inputs_remote.dtype}")
    print(f"Remote targets: {targets_remote.shape}, {targets_remote.dtype}")

    try:
        # Compute cross entropy
        loss = torch.nn.functional.cross_entropy(inputs_remote, targets_remote)
        print(f"Loss: {loss.item()}, dtype: {loss.dtype}")

        # Backward pass
        loss.backward()
        print(f"Gradients shape: {inputs.grad.shape}")
        print("✓ Cross entropy with dtype handling works")

    except Exception as e:
        print(f"Cross entropy failed: {e}")


def test_mse_loss_shape_debug(shared_machines, provider):
    """Debug function: MSE loss shape analysis."""
    print("\n=== Debug: MSE Loss Shapes ===")

    # Test various shapes
    shapes = [(2, 2), (3, 4), (1, 5), (2, 3, 4)]

    for shape in shapes:
        print(f"\nTesting shape: {shape}")

        inputs = torch.randn(shape, requires_grad=True)
        targets = torch.randn(shape)

        print(f"Inputs: {inputs.shape}, Targets: {targets.shape}")

        # Transfer to remote
        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = inputs.to(shared_machines["T4"].device(device_type))
        targets_remote = targets.to(shared_machines["T4"].device(device_type))

        try:
            # Compute MSE loss
            loss = torch.nn.functional.mse_loss(inputs_remote, targets_remote)
            print(f"Loss: {loss.item()}")

            # Backward pass
            loss.backward()
            print(f"Gradient shape: {inputs.grad.shape}")
            print("✓ MSE loss works for this shape")

        except Exception as e:
            print(f"MSE loss failed for shape {shape}: {e}")


def test_direct_tensor_creation(shared_machines, provider):
    """Debug function: Direct tensor creation methods."""
    print("\n=== Debug: Direct Tensor Creation ===")

    device_type = "cpu" if provider == "mock" else "cuda"
    device = shared_machines["T4"].device(device_type)
    print(f"Target device: {device}")

    # Test various creation methods
    creation_methods = [
        ("randn", lambda: torch.randn(2, 2).to(device)),
        ("zeros", lambda: torch.zeros(2, 2).to(device)),
        ("ones", lambda: torch.ones(2, 2).to(device)),
        ("empty", lambda: torch.empty(2, 2).to(device)),
    ]

    for name, create_fn in creation_methods:
        try:
            tensor = create_fn()
            print(f"{name}: {tensor.shape}, {tensor.dtype}, {tensor.device}")
            print(
                f"  Values range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]"
            )
        except Exception as e:
            print(f"{name}: Failed - {e}")


def test_various_tensor_creation_functions(shared_machines, provider):
    """Debug function: Test various tensor creation functions."""
    print("\n=== Debug: Tensor Creation Functions ===")

    device_type = "cpu" if provider == "mock" else "cuda"
    device = shared_machines["T4"].device(device_type)

    # Test tensor creation functions
    functions = [
        ("torch.randn", lambda: torch.randn(3, 3).to(device)),
        ("torch.rand", lambda: torch.rand(3, 3).to(device)),
        ("torch.zeros", lambda: torch.zeros(3, 3).to(device)),
        ("torch.ones", lambda: torch.ones(3, 3).to(device)),
        ("torch.eye", lambda: torch.eye(3).to(device)),
        ("torch.arange", lambda: torch.arange(9).reshape(3, 3).to(device)),
    ]

    for name, func in functions:
        try:
            tensor = func()
            print(f"✓ {name}: {tensor.shape}, {tensor.dtype}")
        except Exception as e:
            print(f"✗ {name}: {e}")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_gradient_propagation_cpu_to_remote(shared_machines, provider):
    """Integration test: Gradient propagation from CPU through remote operations."""
    print("\n=== Integration: CPU->Remote Gradient Flow ===")

    # Start with CPU tensor requiring gradients
    x_cpu = torch.randn(3, 3, requires_grad=True)
    print(f"Starting tensor: {x_cpu.shape}, requires_grad: {x_cpu.requires_grad}")

    # Transfer to remote
    device_type = "cpu" if provider == "mock" else "cuda"
    x_remote = x_cpu.to(shared_machines["T4"].device(device_type))
    print(f"Remote tensor requires_grad: {x_remote.requires_grad}")

    # Perform operations on remote
    y_remote = x_remote * 2 + 1
    z_remote = y_remote.sum()

    print(f"Final tensor: {z_remote.shape}, device: {z_remote.device}")

    # Backward pass
    z_remote.backward()

    # Check gradients on original CPU tensor
    assert x_cpu.grad is not None, "CPU tensor should have gradients"
    print(f"CPU gradient shape: {x_cpu.grad.shape}")
    print(f"Gradient values: min={x_cpu.grad.min():.4f}, max={x_cpu.grad.max():.4f}")

    # Verify gradient values (derivative of 2*x + 1 summed = 2 for all elements)
    expected_grad = torch.full_like(x_cpu, 2.0)
    assert torch.allclose(x_cpu.grad, expected_grad, rtol=1e-4, atol=1e-6)
    print("✓ Gradient propagation verified")


def test_gradient_propagation_remote_to_cpu(shared_machines, provider):
    """Integration test: Operations spanning remote and CPU with gradients."""
    print("\n=== Integration: Remote->CPU Gradient Flow ===")

    # Create tensors
    x_cpu = torch.randn(2, 2, requires_grad=True)
    device_type = "cpu" if provider == "mock" else "cuda"
    x_remote = x_cpu.to(shared_machines["T4"].device(device_type))

    # Remote operations
    y_remote = x_remote**2

    # Transfer back to CPU for final operations
    y_cpu = y_remote.cpu()
    z_cpu = y_cpu.sum()

    print(f"Final CPU tensor: {z_cpu.shape}, requires_grad: {z_cpu.requires_grad}")

    # Backward pass
    z_cpu.backward()

    # Verify gradients
    assert x_cpu.grad is not None, "Should have gradients"
    print(f"Gradient computed: {x_cpu.grad.shape}")

    # Verify gradient values (derivative of x^2 = 2*x)
    expected_grad = 2 * x_cpu
    assert torch.allclose(x_cpu.grad, expected_grad, rtol=1e-4, atol=1e-6)
    print("✓ Remote->CPU gradient flow verified")


def test_mixed_device_gradient_computation(shared_machines, provider):
    """Integration test: Complex mixed device gradient computation."""
    print("\n=== Integration: Mixed Device Gradients ===")

    # Multiple tensors with gradients
    a_cpu = torch.randn(2, 2, requires_grad=True)
    b_cpu = torch.randn(2, 2, requires_grad=True)

    # Transfer to remote
    device_type = "cpu" if provider == "mock" else "cuda"
    a_remote = a_cpu.to(shared_machines["T4"].device(device_type))
    b_remote = b_cpu.to(shared_machines["T4"].device(device_type))

    # Remote operations
    c_remote = a_remote + b_remote
    d_remote = c_remote * a_remote

    # Transfer back and final operation
    d_cpu = d_remote.cpu()
    loss = d_cpu.sum()

    print(f"Loss: {loss.item()}")

    # Backward pass
    loss.backward()

    # Verify both tensors have gradients
    assert a_cpu.grad is not None, "Tensor 'a' should have gradients"
    assert b_cpu.grad is not None, "Tensor 'b' should have gradients"

    print(
        f"Gradient a: {a_cpu.grad.shape}, range: [{a_cpu.grad.min():.4f}, {a_cpu.grad.max():.4f}]"
    )
    print(
        f"Gradient b: {b_cpu.grad.shape}, range: [{b_cpu.grad.min():.4f}, {b_cpu.grad.max():.4f}]"
    )

    # Verify gradients are reasonable (not zero, not NaN)
    assert not torch.allclose(a_cpu.grad, torch.zeros_like(a_cpu)), (
        "Gradients should be non-zero"
    )
    assert not torch.allclose(b_cpu.grad, torch.zeros_like(b_cpu)), (
        "Gradients should be non-zero"
    )
    assert not torch.isnan(a_cpu.grad).any(), "No NaN gradients"
    assert not torch.isnan(b_cpu.grad).any(), "No NaN gradients"

    print("✓ Mixed device gradient computation verified")


def test_gradient_accumulation_across_transfers(shared_machines, provider):
    """Integration test: Gradient accumulation with device transfers."""
    print("\n=== Integration: Gradient Accumulation ===")

    x_cpu = torch.randn(2, 2, requires_grad=True)

    # First operation and backward
    device_type = "cpu" if provider == "mock" else "cuda"
    x_remote1 = x_cpu.to(shared_machines["T4"].device(device_type))
    y1 = (x_remote1 * 2).sum()
    y1.backward(retain_graph=True)

    first_grad = x_cpu.grad.clone()
    print(f"First gradient sum: {first_grad.sum().item():.4f}")

    # Second operation and backward (should accumulate)
    x_remote2 = x_cpu.to(shared_machines["T4"].device(device_type))
    y2 = (x_remote2 + 1).sum()
    y2.backward()

    print(f"Accumulated gradient sum: {x_cpu.grad.sum().item():.4f}")

    # Verify accumulation
    expected_grad = first_grad + torch.ones_like(x_cpu)
    assert torch.allclose(x_cpu.grad, expected_grad, rtol=1e-4, atol=1e-6)
    print("✓ Gradient accumulation across transfers verified")


def test_view_operations_with_gradients(shared_machines, provider):
    """Integration test: View operations preserving gradient flow."""
    print("\n=== Integration: View Operations + Gradients ===")

    # Create tensor with specific shape for view operations
    x_cpu = torch.randn(2, 3, 4, requires_grad=True)
    device_type = "cpu" if provider == "mock" else "cuda"
    x_remote = x_cpu.to(shared_machines["T4"].device(device_type))

    # Chain of view operations
    y = x_remote.view(6, 4)  # Reshape
    y = y.transpose(0, 1)  # Transpose
    y = y.contiguous().view(-1)  # Flatten

    print(f"Original shape: {x_cpu.shape}")
    print(f"Final shape: {y.shape}")

    # Sum for scalar output
    loss = y.sum()

    # Backward through view operations
    loss.backward()

    # Check gradient shape matches original
    assert x_cpu.grad is not None, "Should have gradients"
    assert x_cpu.grad.shape == x_cpu.shape, (
        f"Grad shape {x_cpu.grad.shape} should match tensor shape {x_cpu.shape}"
    )

    # All gradients should be 1 (sum derivative)
    expected_grad = torch.ones_like(x_cpu)
    assert torch.allclose(x_cpu.grad, expected_grad), "All gradients should be 1"
    print("✓ View operations preserve gradients correctly")


def test_custom_loss_function_gradients(shared_machines, provider):
    """Integration test: Custom loss function with device transfers."""
    print("\n=== Integration: Custom Loss Function ===")

    # Create prediction and target tensors
    pred_cpu = torch.randn(4, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2, 1])  # Classification targets

    # Transfer prediction to remote
    device_type = "cpu" if provider == "mock" else "cuda"
    pred_remote = pred_cpu.to(shared_machines["T4"].device(device_type))

    # Apply softmax on remote
    prob_remote = torch.softmax(pred_remote, dim=1)

    # Transfer back for loss computation
    prob_cpu = prob_remote.cpu()

    # Cross-entropy loss (manual implementation)
    log_prob = torch.log(prob_cpu + 1e-8)  # Add small epsilon for numerical stability
    loss = -log_prob[range(len(target)), target].mean()

    print(f"Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients
    assert pred_cpu.grad is not None, "Should have gradients"
    assert pred_cpu.grad.shape == pred_cpu.shape, (
        "Gradient shape should match prediction shape"
    )

    # Verify gradients are reasonable (not all zeros, not NaN)
    assert not torch.allclose(pred_cpu.grad, torch.zeros_like(pred_cpu)), (
        "Gradients should be non-zero"
    )
    assert not torch.isnan(pred_cpu.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(pred_cpu.grad).any(), "Gradients should not contain Inf"

    print("✓ Custom loss gradients computed correctly")
    print(f"Gradient range: [{pred_cpu.grad.min():.6f}, {pred_cpu.grad.max():.6f}]")


# =============================================================================
# ADVANCED INTEGRATION TESTS
# =============================================================================


def test_neural_network_training_simulation(shared_machines, provider):
    """Integration test: Simulate a simple neural network training step."""
    print("\n=== Integration: Neural Network Training ===")

    # Network parameters
    batch_size = 4
    input_size = 6
    hidden_size = 8
    output_size = 3

    # Create data
    inputs = torch.randn(batch_size, input_size)
    targets = torch.randint(0, output_size, (batch_size,), dtype=torch.long)

    # Create simple network weights
    W1 = torch.randn(input_size, hidden_size, requires_grad=True)
    b1 = torch.zeros(hidden_size, requires_grad=True)
    W2 = torch.randn(hidden_size, output_size, requires_grad=True)
    b2 = torch.zeros(output_size, requires_grad=True)

    print(f"Input: {inputs.shape}, Targets: {targets.shape}")
    print(f"W1: {W1.shape}, W2: {W2.shape}")

    # Transfer to remote
    device_type = "cpu" if provider == "mock" else "cuda"
    inputs_remote = inputs.to(shared_machines["T4"].device(device_type))
    targets_remote = targets.to(shared_machines["T4"].device(device_type))
    W1_remote = W1.to(shared_machines["T4"].device(device_type))
    b1_remote = b1.to(shared_machines["T4"].device(device_type))
    W2_remote = W2.to(shared_machines["T4"].device(device_type))
    b2_remote = b2.to(shared_machines["T4"].device(device_type))

    # Forward pass on remote
    h1 = torch.mm(inputs_remote, W1_remote) + b1_remote
    h1_relu = torch.relu(h1)
    h2 = torch.mm(h1_relu, W2_remote) + b2_remote

    # Loss computation
    loss = torch.nn.functional.cross_entropy(h2, targets_remote)

    print(f"Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Verify all parameters have gradients
    parameters = [W1, b1, W2, b2]
    param_names = ["W1", "b1", "W2", "b2"]

    for param, name in zip(parameters, param_names):
        assert param.grad is not None, f"Parameter {name} should have gradients"
        assert not torch.isnan(param.grad).any(), (
            f"Parameter {name} should not have NaN gradients"
        )
        print(
            f"✓ {name} gradient: shape={param.grad.shape}, range=[{param.grad.min():.6f}, {param.grad.max():.6f}]"
        )

    print("✓ Neural network training simulation completed successfully")
