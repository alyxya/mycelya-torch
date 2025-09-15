# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for basic autograd functionality in mycelya-torch.

This module tests fundamental gradient computation, backward passes,
and basic autograd operations on remote tensors.
"""

import pytest
import torch
from test_utilities import NumericalTestUtils, TestConstants


class TestBasicGradientComputation:
    """Tests for basic gradient computation on remote tensors."""

    def test_simple_backward_pass(self, shared_machines, provider):
        """Test simple backward pass on remote tensors."""
        # Create tensors with gradients
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Simple operation
        y_remote = x_remote.sum()

        # Backward pass
        y_remote.backward()

        # Verify gradients exist
        NumericalTestUtils.verify_gradient_flow(x_remote)

        # Verify gradient values
        expected_grad = torch.ones_like(x_cpu)
        NumericalTestUtils.assert_tensors_close(x_remote.grad.cpu(), expected_grad)

    def test_scalar_multiplication_grad(self, shared_machines, provider):
        """Test gradients for scalar multiplication."""
        x_cpu = torch.randn(3, 3, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        scalar = 2.5

        # Operation
        y_remote = x_remote * scalar
        loss_remote = y_remote.sum()

        # Backward pass
        loss_remote.backward()

        # Verify gradients
        NumericalTestUtils.verify_gradient_flow(x_remote)
        expected_grad = torch.full_like(x_cpu, scalar)
        NumericalTestUtils.assert_tensors_close(x_remote.grad.cpu(), expected_grad)

    def test_addition_gradients(self, shared_machines, provider):
        """Test gradients for tensor addition."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        y_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"

        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        y_remote = (
            y_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Operation
        z_remote = x_remote + y_remote
        loss_remote = z_remote.sum()

        # Backward pass
        loss_remote.backward()

        # Verify gradients for both tensors
        NumericalTestUtils.verify_gradient_flow(x_remote)
        NumericalTestUtils.verify_gradient_flow(y_remote)

        # Both should have gradient of ones
        expected_grad = torch.ones_like(x_cpu)
        NumericalTestUtils.assert_tensors_close(x_remote.grad.cpu(), expected_grad)
        NumericalTestUtils.assert_tensors_close(y_remote.grad.cpu(), expected_grad)

    def test_element_wise_multiplication_grad(self, shared_machines, provider):
        """Test gradients for element-wise multiplication."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        y_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"

        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        y_remote = (
            y_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Operation
        z_remote = x_remote * y_remote
        loss_remote = z_remote.sum()

        # Backward pass
        loss_remote.backward()

        # Verify gradients
        NumericalTestUtils.verify_gradient_flow(x_remote)
        NumericalTestUtils.verify_gradient_flow(y_remote)

        # x's gradient should be y, y's gradient should be x
        NumericalTestUtils.assert_tensors_close(x_remote.grad.cpu(), y_cpu)
        NumericalTestUtils.assert_tensors_close(y_remote.grad.cpu(), x_cpu)


class TestMatrixOperationGradients:
    """Tests for gradients in matrix operations."""

    def test_matrix_multiplication_gradients(self, shared_machines, provider):
        """Test gradients for matrix multiplication."""
        x_cpu = torch.randn(2, 3, requires_grad=True)
        y_cpu = torch.randn(3, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"

        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        y_remote = (
            y_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Matrix multiplication
        z_remote = x_remote.mm(y_remote)
        loss_remote = z_remote.sum()

        # Backward pass
        loss_remote.backward()

        # Verify gradients exist
        NumericalTestUtils.verify_gradient_flow(x_remote)
        NumericalTestUtils.verify_gradient_flow(y_remote)

        # Verify gradient shapes
        assert x_remote.grad.shape == x_cpu.shape
        assert y_remote.grad.shape == y_cpu.shape

    def test_batch_matrix_multiplication_gradients(self, shared_machines, provider):
        """Test gradients for batch matrix multiplication."""
        x_cpu = torch.randn(2, 3, 4, requires_grad=True)
        y_cpu = torch.randn(2, 4, 5, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"

        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        y_remote = (
            y_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        try:
            # Batch matrix multiplication
            z_remote = torch.bmm(x_remote, y_remote)
            loss_remote = z_remote.sum()

            # Backward pass
            loss_remote.backward()

            # Verify gradients exist and have correct shapes
            NumericalTestUtils.verify_gradient_flow(x_remote)
            NumericalTestUtils.verify_gradient_flow(y_remote)
            assert x_remote.grad.shape == x_cpu.shape
            assert y_remote.grad.shape == y_cpu.shape
        except (RuntimeError, NotImplementedError):
            pytest.skip("Batch matrix multiplication gradients not supported")


class TestViewOperationGradients:
    """Tests for gradients with view operations."""

    def test_view_operation_gradients(self, shared_machines, provider):
        """Test that gradients flow through view operations."""
        x_cpu = torch.randn(2, 6, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # View operation
        y_remote = x_remote.view(3, 4)
        loss_remote = y_remote.sum()

        # Backward pass
        loss_remote.backward()

        # Verify gradients flow back to original tensor
        NumericalTestUtils.verify_gradient_flow(x_remote)

        # Gradient should have original shape
        assert x_remote.grad.shape == x_cpu.shape
        expected_grad = torch.ones_like(x_cpu)
        NumericalTestUtils.assert_tensors_close(x_remote.grad.cpu(), expected_grad)

    def test_transpose_operation_gradients(self, shared_machines, provider):
        """Test that gradients flow through transpose operations."""
        x_cpu = torch.randn(3, 4, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Transpose operation
        y_remote = x_remote.transpose(0, 1)
        loss_remote = y_remote.sum()

        # Backward pass
        loss_remote.backward()

        # Verify gradients
        NumericalTestUtils.verify_gradient_flow(x_remote)
        assert x_remote.grad.shape == x_cpu.shape
        expected_grad = torch.ones_like(x_cpu)
        NumericalTestUtils.assert_tensors_close(x_remote.grad.cpu(), expected_grad)

    def test_reshape_operation_gradients(self, shared_machines, provider):
        """Test that gradients flow through reshape operations."""
        x_cpu = torch.randn(2, 3, 4, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Reshape operation
        y_remote = x_remote.reshape(6, 4)
        loss_remote = y_remote.sum()

        # Backward pass
        loss_remote.backward()

        # Verify gradients
        NumericalTestUtils.verify_gradient_flow(x_remote)
        assert x_remote.grad.shape == x_cpu.shape


class TestRetainGradFunctionality:
    """Tests for retain_grad() functionality on remote tensors."""

    def test_retain_grad_basic(self, shared_machines, provider):
        """Test basic retain_grad() functionality with remote tensors."""
        x_cpu = torch.randn(3, 3, requires_grad=True)
        y_cpu = torch.randn(3, 3, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"

        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        y_remote = (
            y_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Intermediate computation - normally gradients wouldn't be retained
        z_remote = x_remote * y_remote
        z_remote.retain_grad()  # Explicitly retain gradients for intermediate tensor

        # Final computation
        loss_remote = z_remote.sum()

        # Backward pass
        loss_remote.backward()

        # Check that gradients are available
        NumericalTestUtils.verify_gradient_flow(x_remote)
        NumericalTestUtils.verify_gradient_flow(y_remote)
        assert z_remote.grad is not None, "z should have gradients (retained)"

        # Verify gradient values are correct
        expected_z_grad = torch.ones_like(z_remote.cpu())
        expected_x_grad = y_cpu  # d/dx(x*y) = y
        expected_y_grad = x_cpu  # d/dy(x*y) = x

        NumericalTestUtils.assert_tensors_close(
            z_remote.grad.cpu(), expected_z_grad, msg="z gradient should be ones"
        )
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(), expected_x_grad, msg="x gradient should equal y"
        )
        NumericalTestUtils.assert_tensors_close(
            y_remote.grad.cpu(), expected_y_grad, msg="y gradient should equal x"
        )

    def test_retain_grad_multiple_intermediates(self, shared_machines, provider):
        """Test retain_grad() with multiple intermediate tensors."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        y_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"

        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        y_remote = (
            y_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Multiple intermediate computations
        z1_remote = x_remote + y_remote
        z1_remote.retain_grad()

        z2_remote = z1_remote * 2
        z2_remote.retain_grad()

        # Final loss
        loss_remote = z2_remote.mean()

        # Backward
        loss_remote.backward()

        # All should have gradients
        NumericalTestUtils.verify_gradient_flow(x_remote)
        NumericalTestUtils.verify_gradient_flow(y_remote)
        assert z1_remote.grad is not None, "z1 should have gradients (retained)"
        assert z2_remote.grad is not None, "z2 should have gradients (retained)"

        # Verify gradient shapes
        assert z1_remote.grad.shape == z1_remote.shape
        assert z2_remote.grad.shape == z2_remote.shape

    def test_retain_grad_comparison_with_cpu(self, shared_machines, provider):
        """Test that retain_grad() on remote tensors matches CPU behavior."""
        x_cpu = torch.randn(2, 3, requires_grad=True)
        y_cpu = torch.randn(2, 3, requires_grad=True)

        # CPU computation
        z_cpu = x_cpu * y_cpu
        z_cpu.retain_grad()
        loss_cpu = z_cpu.sum()
        loss_cpu.backward()

        # Remote computation
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        y_remote = (
            y_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        z_remote = x_remote * y_remote
        z_remote.retain_grad()
        loss_remote = z_remote.sum()
        loss_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(), x_cpu.grad, msg="Remote x gradient should match CPU"
        )
        NumericalTestUtils.assert_tensors_close(
            y_remote.grad.cpu(), y_cpu.grad, msg="Remote y gradient should match CPU"
        )
        NumericalTestUtils.assert_tensors_close(
            z_remote.grad.cpu(), z_cpu.grad, msg="Remote z gradient should match CPU"
        )

    def test_without_retain_grad_no_intermediate_gradients(
        self, shared_machines, provider
    ):
        """Test that without retain_grad(), intermediate tensors don't have gradients."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        y_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"

        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        y_remote = (
            y_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Intermediate computation WITHOUT retain_grad()
        z_remote = x_remote * y_remote
        # z_remote.retain_grad()  # <-- NOT called

        loss_remote = z_remote.sum()
        loss_remote.backward()

        # Check gradients
        NumericalTestUtils.verify_gradient_flow(x_remote)
        NumericalTestUtils.verify_gradient_flow(y_remote)

        # Accessing .grad on non-leaf tensor without retain_grad() should be None
        # (PyTorch will warn about this, but that's expected behavior)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert z_remote.grad is None, "z should NOT have gradients (not retained)"


class TestGradientAccumulation:
    """Tests for gradient accumulation behavior."""

    def test_gradient_accumulation_basic(self, shared_machines, provider):
        """Test basic gradient accumulation."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # First backward pass
        y1_remote = x_remote.sum()
        y1_remote.backward(retain_graph=True)
        first_grad = x_remote.grad.clone()

        # Second backward pass (should accumulate)
        y2_remote = (x_remote * 2).sum()
        y2_remote.backward()

        # Verify gradient accumulation
        expected_grad = first_grad + 2 * torch.ones_like(first_grad)
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(), expected_grad.cpu()
        )

    def test_gradient_zero_behavior(self, shared_machines, provider):
        """Test gradient zeroing behavior."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # First backward pass
        y_remote = x_remote.sum()
        y_remote.backward()

        # Verify gradients exist
        assert x_remote.grad is not None

        # Zero gradients
        x_remote.grad.zero_()

        # Verify gradients are zero
        expected_zero = torch.zeros_like(x_cpu)
        NumericalTestUtils.assert_tensors_close(x_remote.grad.cpu(), expected_zero)


class TestAutoGradFunction:
    """Tests for autograd function behavior."""

    def test_requires_grad_propagation(self, shared_machines, provider):
        """Test that requires_grad propagates correctly."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        y_cpu = torch.randn(2, 2, requires_grad=False)
        device_type = "cpu" if provider == "mock" else "cuda"

        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        y_remote = y_cpu.to(shared_machines["T4"].device(device_type)).detach()

        # Operations with requires_grad=True tensor
        z_remote = x_remote + y_remote
        assert z_remote.requires_grad

        # Operations with only requires_grad=False tensors
        w_remote = y_remote * 2
        assert not w_remote.requires_grad

    def test_detach_behavior(self, shared_machines, provider):
        """Test tensor detach behavior."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Detach tensor
        x_detached = x_remote.detach()
        assert not x_detached.requires_grad

        # Operations on detached tensor shouldn't require grad
        y_remote = x_detached * 2
        assert not y_remote.requires_grad

    def test_no_grad_context(self, shared_machines, provider):
        """Test torch.no_grad() context behavior."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        with torch.no_grad():
            y_remote = x_remote * 2
            assert not y_remote.requires_grad


class TestGradientNumericalVerification:
    """Tests for numerical verification of gradients."""

    def test_gradient_numerical_simple(self, shared_machines, provider):
        """Test gradient computation against CPU reference."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # CPU computation
        y_cpu = (x_cpu**2).sum()
        y_cpu.backward()
        cpu_grad = x_cpu.grad.clone()

        # Remote computation
        y_remote = (x_remote**2).sum()
        y_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(),
            cpu_grad,
            msg="Remote gradient doesn't match CPU gradient",
        )

    def test_chain_rule_verification(self, shared_machines, provider):
        """Test chain rule implementation."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # CPU computation: y = (x^2 + 1) * 3
        y_cpu = ((x_cpu**2) + 1) * 3
        loss_cpu = y_cpu.sum()
        loss_cpu.backward()
        cpu_grad = x_cpu.grad.clone()

        # Remote computation
        y_remote = ((x_remote**2) + 1) * 3
        loss_remote = y_remote.sum()
        loss_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(), cpu_grad, msg="Chain rule gradient doesn't match CPU"
        )


@pytest.mark.parametrize("shape", TestConstants.SMALL_SHAPES)
def test_parametrized_gradient_shapes(shared_machines, provider, shape):
    """Test gradient computation with various tensor shapes."""
    x_cpu = torch.randn(shape, requires_grad=True)
    device_type = "cpu" if provider == "mock" else "cuda"
    x_remote = (
        x_cpu.to(shared_machines["T4"].device(device_type)).detach().requires_grad_()
    )

    # Simple operation
    y_remote = x_remote.sum()
    y_remote.backward()

    # Verify gradient shape and values
    NumericalTestUtils.verify_gradient_flow(x_remote)
    assert x_remote.grad.shape == shape
    expected_grad = torch.ones(shape)
    NumericalTestUtils.assert_tensors_close(x_remote.grad.cpu(), expected_grad)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: x.sum(),
        lambda x: (x**2).sum(),
        lambda x: (x * 2).sum(),
        lambda x: x.mean(),
    ],
)
def test_parametrized_operations_gradients(shared_machines, provider, operation):
    """Test gradients for various operations."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    device_type = "cpu" if provider == "mock" else "cuda"
    x_remote = (
        x_cpu.clone()
        .to(shared_machines["T4"].device(device_type))
        .detach()
        .requires_grad_()
    )

    try:
        # CPU computation
        y_cpu = operation(x_cpu)
        y_cpu.backward()
        cpu_grad = x_cpu.grad.clone()

        # Remote computation
        y_remote = operation(x_remote)
        y_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(),
            cpu_grad,
            msg=f"Gradient mismatch for operation {operation}",
        )
    except (RuntimeError, NotImplementedError):
        pytest.skip(f"Operation {operation} gradients not supported")
