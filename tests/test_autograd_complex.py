# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for complex autograd functionality in mycelya-torch.

This module tests advanced gradient scenarios, cross-device gradient flow,
complex computational graphs, and integration with PyTorch optimizers.
"""

import pytest
import torch
from test_utilities import (
    DeviceTestUtils,
    NumericalTestUtils,
    TestConstants,
)


class TestComplexComputationalGraphs:
    """Tests for complex computational graphs and gradient flow."""

    def test_multiple_operations_chain(self, shared_machines, provider):
        """Test gradients through multiple chained operations."""
        x_cpu = torch.randn(3, 3, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # CPU computation: complex chain
        y_cpu = ((x_cpu * 2 + 1) ** 2).sum()
        y_cpu.backward()
        cpu_grad = x_cpu.grad.clone()

        # Remote computation
        y_remote = ((x_remote * 2 + 1) ** 2).sum()
        y_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(), cpu_grad, msg="Complex chain gradient mismatch"
        )

    def test_branching_computational_graph(self, shared_machines, provider):
        """Test gradients with branching computational graph."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # CPU computation: branching
        y1_cpu = x_cpu * 2
        y2_cpu = x_cpu + 1
        z_cpu = y1_cpu + y2_cpu
        loss_cpu = z_cpu.sum()
        loss_cpu.backward()
        cpu_grad = x_cpu.grad.clone()

        # Remote computation
        y1_remote = x_remote * 2
        y2_remote = x_remote + 1
        z_remote = y1_remote + y2_remote
        loss_remote = z_remote.sum()
        loss_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(), cpu_grad, msg="Branching graph gradient mismatch"
        )

    def test_multiple_inputs_graph(self, shared_machines, provider):
        """Test gradients with multiple input tensors."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        y_cpu = torch.randn(2, 2, requires_grad=True)
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

        # CPU computation
        z1_cpu = x_cpu * y_cpu
        z2_cpu = x_cpu + y_cpu
        loss_cpu = (z1_cpu + z2_cpu).sum()
        loss_cpu.backward()
        x_cpu_grad = x_cpu.grad.clone()
        y_cpu_grad = y_cpu.grad.clone()

        # Remote computation
        z1_remote = x_remote * y_remote
        z2_remote = x_remote + y_remote
        loss_remote = (z1_remote + z2_remote).sum()
        loss_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(), x_cpu_grad, msg="Multi-input x gradient mismatch"
        )
        NumericalTestUtils.assert_tensors_close(
            y_remote.grad.cpu(), y_cpu_grad, msg="Multi-input y gradient mismatch"
        )


class TestAdvancedMatrixOperations:
    """Tests for gradients in advanced matrix operations."""

    def test_matrix_chain_multiplication(self, shared_machines, provider):
        """Test gradients through chained matrix multiplications."""
        A_cpu = torch.randn(2, 3, requires_grad=True)
        B_cpu = torch.randn(3, 4, requires_grad=True)
        C_cpu = torch.randn(4, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"

        A_remote = (
            A_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        B_remote = (
            B_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        C_remote = (
            C_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # CPU computation: A @ B @ C
        result_cpu = A_cpu.mm(B_cpu).mm(C_cpu)
        loss_cpu = result_cpu.sum()
        loss_cpu.backward()

        # Remote computation
        result_remote = A_remote.mm(B_remote).mm(C_remote)
        loss_remote = result_remote.sum()
        loss_remote.backward()

        # Compare gradients for all matrices
        NumericalTestUtils.assert_tensors_close(
            A_remote.grad.cpu(), A_cpu.grad, msg="Matrix A gradient mismatch in chain"
        )
        NumericalTestUtils.assert_tensors_close(
            B_remote.grad.cpu(), B_cpu.grad, msg="Matrix B gradient mismatch in chain"
        )
        NumericalTestUtils.assert_tensors_close(
            C_remote.grad.cpu(), C_cpu.grad, msg="Matrix C gradient mismatch in chain"
        )

    def test_matrix_element_wise_combination(self, shared_machines, provider):
        """Test gradients with matrix operations and element-wise operations."""
        X_cpu = torch.randn(3, 3, requires_grad=True)
        Y_cpu = torch.randn(3, 3, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"

        X_remote = (
            X_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        Y_remote = (
            Y_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # CPU computation: (X @ Y) * (X + Y)
        mm_result_cpu = X_cpu.mm(Y_cpu)
        add_result_cpu = X_cpu + Y_cpu
        final_cpu = mm_result_cpu * add_result_cpu
        loss_cpu = final_cpu.sum()
        loss_cpu.backward()

        # Remote computation
        mm_result_remote = X_remote.mm(Y_remote)
        add_result_remote = X_remote + Y_remote
        final_remote = mm_result_remote * add_result_remote
        loss_remote = final_remote.sum()
        loss_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            X_remote.grad.cpu(),
            X_cpu.grad,
            msg="X gradient mismatch in combined operations",
        )
        NumericalTestUtils.assert_tensors_close(
            Y_remote.grad.cpu(),
            Y_cpu.grad,
            msg="Y gradient mismatch in combined operations",
        )


class TestHigherOrderGradients:
    """Tests for higher-order gradients."""

    def test_second_order_gradients(self, shared_machines, provider):
        """Test second-order gradient computation."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        try:
            # CPU computation
            y_cpu = (x_cpu**3).sum()
            grad_cpu = torch.autograd.grad(y_cpu, x_cpu, create_graph=True)[0]
            second_grad_cpu = torch.autograd.grad(grad_cpu.sum(), x_cpu)[0]

            # Remote computation
            y_remote = (x_remote**3).sum()
            grad_remote = torch.autograd.grad(y_remote, x_remote, create_graph=True)[0]
            second_grad_remote = torch.autograd.grad(grad_remote.sum(), x_remote)[0]

            # Compare second-order gradients
            NumericalTestUtils.assert_tensors_close(
                second_grad_remote.cpu(),
                second_grad_cpu,
                msg="Second-order gradient mismatch",
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip("Second-order gradients not supported")


class TestGradientWithViewOperations:
    """Tests for gradients with complex view operations."""

    def test_gradients_through_multiple_views(self, shared_machines, provider):
        """Test gradients through multiple view operations."""
        x_cpu = torch.randn(2, 3, 4, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # CPU computation with multiple views
        y_cpu = x_cpu.view(6, 4).transpose(0, 1).contiguous().view(-1)
        loss_cpu = y_cpu.sum()
        loss_cpu.backward()
        cpu_grad = x_cpu.grad.clone()

        # Remote computation
        try:
            y_remote = x_remote.view(6, 4).transpose(0, 1).contiguous().view(-1)
            loss_remote = y_remote.sum()
            loss_remote.backward()

            # Compare gradients
            NumericalTestUtils.assert_tensors_close(
                x_remote.grad.cpu(), cpu_grad, msg="Multi-view gradient mismatch"
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip("Complex view operations not supported")

    def test_gradients_with_permute_operations(self, shared_machines, provider):
        """Test gradients with permute operations."""
        x_cpu = torch.randn(2, 3, 4, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # CPU computation with permute
        y_cpu = x_cpu.permute(2, 0, 1)
        loss_cpu = y_cpu.sum()
        loss_cpu.backward()
        cpu_grad = x_cpu.grad.clone()

        # Remote computation
        y_remote = x_remote.permute(2, 0, 1)
        loss_remote = y_remote.sum()
        loss_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(), cpu_grad, msg="Permute gradient mismatch"
        )


class TestNeuralNetworkGradients:
    """Tests for gradients in neural network-like computations."""

    def test_linear_layer_gradients(self, shared_machines, provider):
        """Test gradients through linear layer computation."""
        # Setup
        batch_size = 3
        input_size = 4
        output_size = 2

        # Inputs
        x_cpu = torch.randn(batch_size, input_size, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Parameters
        weight_cpu = torch.randn(output_size, input_size, requires_grad=True)
        bias_cpu = torch.randn(output_size, requires_grad=True)

        weight_remote = (
            weight_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )
        bias_remote = (
            bias_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # CPU computation: y = x @ W.T + b
        output_cpu = torch.nn.functional.linear(x_cpu, weight_cpu, bias_cpu)
        loss_cpu = output_cpu.sum()
        loss_cpu.backward()

        # Remote computation
        output_remote = torch.nn.functional.linear(x_remote, weight_remote, bias_remote)
        loss_remote = output_remote.sum()
        loss_remote.backward()

        # Compare gradients
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(),
            x_cpu.grad,
            msg="Input gradient mismatch in linear layer",
        )
        NumericalTestUtils.assert_tensors_close(
            weight_remote.grad.cpu(),
            weight_cpu.grad,
            msg="Weight gradient mismatch in linear layer",
        )
        NumericalTestUtils.assert_tensors_close(
            bias_remote.grad.cpu(),
            bias_cpu.grad,
            msg="Bias gradient mismatch in linear layer",
        )

    def test_activation_function_gradients(self, shared_machines, provider):
        """Test gradients through activation functions."""
        x_cpu = torch.randn(3, 4, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        activation_functions = [
            ("relu", torch.nn.functional.relu),
            ("sigmoid", torch.sigmoid),
            ("tanh", torch.tanh),
        ]

        for name, activation_fn in activation_functions:
            try:
                # Reset gradients
                if x_cpu.grad is not None:
                    x_cpu.grad.zero_()
                if x_remote.grad is not None:
                    x_remote.grad.zero_()

                # CPU computation
                y_cpu = activation_fn(x_cpu)
                loss_cpu = y_cpu.sum()
                loss_cpu.backward(retain_graph=True)
                cpu_grad = x_cpu.grad.clone()

                # Remote computation
                y_remote = activation_fn(x_remote)
                loss_remote = y_remote.sum()
                loss_remote.backward(retain_graph=True)

                # Compare gradients
                NumericalTestUtils.assert_tensors_close(
                    x_remote.grad.cpu(),
                    cpu_grad,
                    msg=f"{name} activation gradient mismatch",
                )
            except (RuntimeError, NotImplementedError):
                pytest.skip(f"{name} activation gradients not supported")


class TestMultipleBackwardPasses:
    """Tests for multiple backward passes and gradient accumulation."""

    def test_multiple_backward_passes_same_graph(self, shared_machines, provider):
        """Test multiple backward passes on the same computational graph."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # Create computational graph
        y_remote = x_remote**2
        z_remote = y_remote.sum()

        # First backward pass
        z_remote.backward(retain_graph=True)
        first_grad = x_remote.grad.clone()

        # Second backward pass (should accumulate)
        z_remote.backward()
        second_grad = x_remote.grad.clone()

        # Verify accumulation
        expected_grad = first_grad * 2
        NumericalTestUtils.assert_tensors_close(
            second_grad.cpu(), expected_grad.cpu(), msg="Gradient accumulation failed"
        )

    def test_gradient_accumulation_different_operations(
        self, shared_machines, provider
    ):
        """Test gradient accumulation from different operations."""
        x_cpu = torch.randn(2, 2, requires_grad=True)
        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote = (
            x_cpu.clone()
            .to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_()
        )

        # First operation and backward
        y1_remote = (x_remote * 2).sum()
        y1_remote.backward(retain_graph=True)
        first_grad = x_remote.grad.clone()

        # Second operation and backward
        y2_remote = (x_remote + 1).sum()
        y2_remote.backward()

        # Verify total accumulated gradient
        expected_grad = first_grad + torch.ones_like(x_remote)
        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(),
            expected_grad.cpu(),
            msg="Multi-operation gradient accumulation failed",
        )


class TestCrossDeviceGradientRestrictions:
    """Tests for cross-device gradient restrictions and error handling."""

    def test_cross_device_gradient_restriction(self, shared_machines, provider):
        """Test that cross-device operations properly restrict gradients."""
        if len([k for k in TestConstants.DEVICE_KEYS if k in shared_machines]) < 2:
            pytest.skip("Need at least 2 devices for cross-device testing")

        # Get two different devices
        machine_keys = [k for k in TestConstants.DEVICE_KEYS if k in shared_machines][
            :2
        ]
        machine1_key, machine2_key = machine_keys[0], machine_keys[1]

        device_type = "cpu" if provider == "mock" else "cuda"
        x_remote1 = DeviceTestUtils.create_remote_tensor(
            (2, 2),
            shared_machines,
            machine1_key,
            requires_grad=True,
            device_type=device_type,
        )
        y_remote2 = DeviceTestUtils.create_remote_tensor(
            (2, 2),
            shared_machines,
            machine2_key,
            requires_grad=True,
            device_type=device_type,
        )

        # Try cross-device operation (should fail)
        try:
            z = x_remote1 + y_remote2
            loss = z.sum()
            loss.backward()
            pytest.fail("Expected cross-device gradient to fail")
        except RuntimeError as e:
            assert "different remote devices" in str(e) or "device" in str(e).lower()


@pytest.mark.parametrize(
    "batch_size,input_size,output_size",
    [
        (2, 3, 2),
        (5, 4, 3),
        (1, 10, 5),
    ],
)
def test_parametrized_linear_gradients(
    shared_machines, provider, batch_size, input_size, output_size
):
    """Test linear layer gradients with various dimensions."""
    x_cpu = torch.randn(batch_size, input_size, requires_grad=True)
    weight_cpu = torch.randn(output_size, input_size, requires_grad=True)
    bias_cpu = torch.randn(output_size, requires_grad=True)

    device_type = "cpu" if provider == "mock" else "cuda"
    x_remote = (
        x_cpu.clone()
        .to(shared_machines["T4"].device(device_type))
        .detach()
        .requires_grad_()
    )
    weight_remote = (
        weight_cpu.clone()
        .to(shared_machines["T4"].device(device_type))
        .detach()
        .requires_grad_()
    )
    bias_remote = (
        bias_cpu.clone()
        .to(shared_machines["T4"].device(device_type))
        .detach()
        .requires_grad_()
    )

    # CPU computation
    output_cpu = torch.nn.functional.linear(x_cpu, weight_cpu, bias_cpu)
    loss_cpu = output_cpu.sum()
    loss_cpu.backward()

    # Remote computation
    output_remote = torch.nn.functional.linear(x_remote, weight_remote, bias_remote)
    loss_remote = output_remote.sum()
    loss_remote.backward()

    # Compare all gradients
    NumericalTestUtils.assert_tensors_close(
        x_remote.grad.cpu(),
        x_cpu.grad,
        msg=f"Input gradient mismatch for dims ({batch_size}, {input_size}, {output_size})",
    )
    NumericalTestUtils.assert_tensors_close(
        weight_remote.grad.cpu(),
        weight_cpu.grad,
        msg=f"Weight gradient mismatch for dims ({batch_size}, {input_size}, {output_size})",
    )
    NumericalTestUtils.assert_tensors_close(
        bias_remote.grad.cpu(),
        bias_cpu.grad,
        msg=f"Bias gradient mismatch for dims ({batch_size}, {input_size}, {output_size})",
    )
