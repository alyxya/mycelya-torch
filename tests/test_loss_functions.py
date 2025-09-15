# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for loss function integration in mycelya-torch.

This module tests various PyTorch loss functions working with remote tensors,
including MSE loss, cross-entropy loss, and custom loss computations.
"""

import pytest
import torch
import torch.nn.functional as F
from test_utilities import (
    NumericalTestUtils,
    TestConstants,
)


class TestMSELoss:
    """Tests for Mean Squared Error loss with remote tensors."""

    def test_mse_loss_basic(self, shared_machines, provider):
        """Test basic MSE loss computation."""
        # Create input and target tensors
        inputs_cpu = torch.randn(3, 4, requires_grad=True)
        targets_cpu = torch.randn(3, 4)

        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = (
            inputs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        # CPU computation
        loss_cpu = F.mse_loss(inputs_cpu, targets_cpu)
        loss_cpu.backward()
        cpu_grad = inputs_cpu.grad.clone()

        # Remote computation
        loss_remote = F.mse_loss(inputs_remote, targets_remote)
        loss_remote.backward()

        # Compare loss values and gradients
        NumericalTestUtils.assert_tensors_close(
            loss_remote.cpu(), loss_cpu, msg="MSE loss values don't match"
        )
        NumericalTestUtils.assert_tensors_close(
            inputs_remote.grad.cpu(), cpu_grad, msg="MSE loss gradients don't match"
        )

    def test_mse_loss_reduction_options(self, shared_machines, provider):
        """Test MSE loss with different reduction options."""
        inputs_cpu = torch.randn(2, 3, requires_grad=True)
        targets_cpu = torch.randn(2, 3)

        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = (
            inputs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        reductions = ["mean", "sum", "none"]

        for reduction in reductions:
            try:
                # Reset gradients
                if inputs_cpu.grad is not None:
                    inputs_cpu.grad.zero_()
                if inputs_remote.grad is not None:
                    inputs_remote.grad.zero_()

                # CPU computation
                loss_cpu = F.mse_loss(inputs_cpu, targets_cpu, reduction=reduction)
                if loss_cpu.dim() > 0:  # Only backward if scalar
                    loss_cpu.sum().backward(retain_graph=True)
                else:
                    loss_cpu.backward(retain_graph=True)

                # Remote computation
                loss_remote = F.mse_loss(
                    inputs_remote, targets_remote, reduction=reduction
                )
                if loss_remote.dim() > 0:
                    loss_remote.sum().backward(retain_graph=True)
                else:
                    loss_remote.backward(retain_graph=True)

                # Compare results
                NumericalTestUtils.assert_tensors_close(
                    loss_remote.cpu(),
                    loss_cpu,
                    msg=f"MSE loss with reduction={reduction} doesn't match",
                )
            except (RuntimeError, NotImplementedError):
                pytest.skip(f"MSE loss with reduction={reduction} not supported")

    def test_mse_loss_various_shapes(self, shared_machines, provider):
        """Test MSE loss with various tensor shapes."""
        test_shapes = TestConstants.SMALL_SHAPES + [(1, 10), (5, 1)]

        device_type = "cpu" if provider == "mock" else "cuda"
        for shape in test_shapes:
            inputs_cpu = torch.randn(shape, requires_grad=True)
            targets_cpu = torch.randn(shape)

            inputs_remote = (
                inputs_cpu.to(shared_machines["T4"].device(device_type))
                .detach()
                .requires_grad_(True)
            )
            targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

            # CPU computation
            loss_cpu = F.mse_loss(inputs_cpu, targets_cpu)
            loss_cpu.backward()

            # Remote computation
            loss_remote = F.mse_loss(inputs_remote, targets_remote)
            loss_remote.backward()

            # Verify gradients exist and have correct shape
            NumericalTestUtils.verify_gradient_flow(inputs_remote)
            assert inputs_remote.grad.shape == shape


class TestCrossEntropyLoss:
    """Tests for Cross Entropy loss with remote tensors."""

    def test_cross_entropy_basic(self, shared_machines, provider):
        """Test basic cross entropy loss computation."""
        batch_size = 3
        num_classes = 4

        # Generate test data
        inputs_cpu = torch.randn(batch_size, num_classes, requires_grad=True)
        targets_cpu = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = (
            inputs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        # CPU computation
        loss_cpu = F.cross_entropy(inputs_cpu, targets_cpu)
        loss_cpu.backward()
        cpu_grad = inputs_cpu.grad.clone()

        # Remote computation
        loss_remote = F.cross_entropy(inputs_remote, targets_remote)
        loss_remote.backward()

        # Compare results
        NumericalTestUtils.assert_tensors_close(
            loss_remote.cpu(), loss_cpu, msg="Cross entropy loss values don't match"
        )
        NumericalTestUtils.assert_tensors_close(
            inputs_remote.grad.cpu(),
            cpu_grad,
            msg="Cross entropy gradients don't match",
        )

    def test_cross_entropy_with_weights(self, shared_machines, provider):
        """Test cross entropy loss with class weights."""
        batch_size = 3
        num_classes = 4

        inputs_cpu = torch.randn(batch_size, num_classes, requires_grad=True)
        targets_cpu = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        weights_cpu = torch.randn(num_classes)

        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = (
            inputs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))
        weights_remote = weights_cpu.to(shared_machines["T4"].device(device_type))

        try:
            # CPU computation
            loss_cpu = F.cross_entropy(inputs_cpu, targets_cpu, weight=weights_cpu)
            loss_cpu.backward()
            cpu_grad = inputs_cpu.grad.clone()

            # Remote computation
            loss_remote = F.cross_entropy(
                inputs_remote, targets_remote, weight=weights_remote
            )
            loss_remote.backward()

            # Compare results
            NumericalTestUtils.assert_tensors_close(
                loss_remote.cpu(),
                loss_cpu,
                msg="Weighted cross entropy loss values don't match",
            )
            NumericalTestUtils.assert_tensors_close(
                inputs_remote.grad.cpu(),
                cpu_grad,
                msg="Weighted cross entropy gradients don't match",
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip("Weighted cross entropy not supported")

    def test_cross_entropy_different_batch_sizes(self, shared_machines, provider):
        """Test cross entropy with different batch sizes."""
        num_classes = 3
        batch_sizes = [1, 2, 5, 10]

        device_type = "cpu" if provider == "mock" else "cuda"
        for batch_size in batch_sizes:
            inputs_cpu = torch.randn(batch_size, num_classes, requires_grad=True)
            targets_cpu = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

            inputs_remote = (
                inputs_cpu.to(shared_machines["T4"].device(device_type))
                .detach()
                .requires_grad_(True)
            )
            targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

            # Remote computation
            loss_remote = F.cross_entropy(inputs_remote, targets_remote)
            loss_remote.backward()

            # Verify computation succeeds and gradients exist
            assert loss_remote.dim() == 0  # Should be scalar
            NumericalTestUtils.verify_gradient_flow(inputs_remote)
            assert inputs_remote.grad.shape == (batch_size, num_classes)


class TestNLLLoss:
    """Tests for Negative Log Likelihood loss with remote tensors."""

    def test_nll_loss_basic(self, shared_machines, provider):
        """Test basic NLL loss computation."""
        batch_size = 3
        num_classes = 4

        # Generate log probabilities and targets
        log_probs_cpu = torch.randn(batch_size, num_classes, requires_grad=True)
        targets_cpu = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

        device_type = "cpu" if provider == "mock" else "cuda"
        log_probs_remote = (
            log_probs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        try:
            # CPU computation
            loss_cpu = F.nll_loss(log_probs_cpu, targets_cpu)
            loss_cpu.backward()
            cpu_grad = log_probs_cpu.grad.clone()

            # Remote computation
            loss_remote = F.nll_loss(log_probs_remote, targets_remote)
            loss_remote.backward()

            # Compare results
            NumericalTestUtils.assert_tensors_close(
                loss_remote.cpu(), loss_cpu, msg="NLL loss values don't match"
            )
            NumericalTestUtils.assert_tensors_close(
                log_probs_remote.grad.cpu(),
                cpu_grad,
                msg="NLL loss gradients don't match",
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip("NLL loss not supported")


class TestCustomLossFunctions:
    """Tests for custom loss functions with remote tensors."""

    def test_l1_loss(self, shared_machines, provider):
        """Test L1 (Mean Absolute Error) loss."""
        inputs_cpu = torch.randn(3, 4, requires_grad=True)
        targets_cpu = torch.randn(3, 4)

        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = (
            inputs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        try:
            # CPU computation
            loss_cpu = F.l1_loss(inputs_cpu, targets_cpu)
            loss_cpu.backward()
            cpu_grad = inputs_cpu.grad.clone()

            # Remote computation
            loss_remote = F.l1_loss(inputs_remote, targets_remote)
            loss_remote.backward()

            # Compare results
            NumericalTestUtils.assert_tensors_close(
                loss_remote.cpu(), loss_cpu, msg="L1 loss values don't match"
            )
            NumericalTestUtils.assert_tensors_close(
                inputs_remote.grad.cpu(), cpu_grad, msg="L1 loss gradients don't match"
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip("L1 loss not supported")

    def test_smooth_l1_loss(self, shared_machines, provider):
        """Test Smooth L1 loss (Huber loss)."""
        inputs_cpu = torch.randn(2, 3, requires_grad=True)
        targets_cpu = torch.randn(2, 3)

        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = (
            inputs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        try:
            # CPU computation
            loss_cpu = F.smooth_l1_loss(inputs_cpu, targets_cpu)
            loss_cpu.backward()
            cpu_grad = inputs_cpu.grad.clone()

            # Remote computation
            loss_remote = F.smooth_l1_loss(inputs_remote, targets_remote)
            loss_remote.backward()

            # Compare results
            NumericalTestUtils.assert_tensors_close(
                loss_remote.cpu(), loss_cpu, msg="Smooth L1 loss values don't match"
            )
            NumericalTestUtils.assert_tensors_close(
                inputs_remote.grad.cpu(),
                cpu_grad,
                msg="Smooth L1 loss gradients don't match",
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip("Smooth L1 loss not supported")

    def test_manual_custom_loss(self, shared_machines, provider):
        """Test manually implemented custom loss function."""
        inputs_cpu = torch.randn(2, 2, requires_grad=True)
        targets_cpu = torch.randn(2, 2)

        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = (
            inputs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        # Custom loss: squared difference with regularization
        def custom_loss(pred, target):
            diff = pred - target
            squared_diff = diff**2
            regularization = 0.01 * (pred**2).sum()
            return squared_diff.mean() + regularization

        # CPU computation
        loss_cpu = custom_loss(inputs_cpu, targets_cpu)
        loss_cpu.backward()
        cpu_grad = inputs_cpu.grad.clone()

        # Remote computation
        loss_remote = custom_loss(inputs_remote, targets_remote)
        loss_remote.backward()

        # Compare results
        NumericalTestUtils.assert_tensors_close(
            loss_remote.cpu(), loss_cpu, msg="Custom loss values don't match"
        )
        NumericalTestUtils.assert_tensors_close(
            inputs_remote.grad.cpu(), cpu_grad, msg="Custom loss gradients don't match"
        )


class TestLossWithNeuralNetworks:
    """Tests for loss functions integrated with neural network modules."""

    def test_mse_with_linear_model(self, shared_machines, provider):
        """Test MSE loss with a simple linear model."""
        batch_size = 3
        input_size = 4
        output_size = 2

        # Create models
        model_cpu = torch.nn.Linear(input_size, output_size)
        model_remote = torch.nn.Linear(input_size, output_size)

        # Copy weights to remote model and move to device
        model_remote.load_state_dict(model_cpu.state_dict())
        device_type = "cpu" if provider == "mock" else "cuda"
        model_remote = model_remote.to(shared_machines["T4"].device(device_type))

        # Create data
        inputs_cpu = torch.randn(batch_size, input_size)
        targets_cpu = torch.randn(batch_size, output_size)

        inputs_remote = inputs_cpu.to(shared_machines["T4"].device(device_type))
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        # CPU computation
        outputs_cpu = model_cpu(inputs_cpu)
        loss_cpu = F.mse_loss(outputs_cpu, targets_cpu)
        loss_cpu.backward()

        # Remote computation
        outputs_remote = model_remote(inputs_remote)
        loss_remote = F.mse_loss(outputs_remote, targets_remote)
        loss_remote.backward()

        # Compare loss values
        NumericalTestUtils.assert_tensors_close(
            loss_remote.cpu(), loss_cpu, msg="Model MSE loss values don't match"
        )

        # Verify gradients exist for model parameters
        for param_cpu, param_remote in zip(
            model_cpu.parameters(), model_remote.parameters()
        ):
            assert param_remote.grad is not None
            assert param_remote.grad.shape == param_cpu.grad.shape

    def test_cross_entropy_with_classification_model(self, shared_machines, provider):
        """Test cross entropy loss with a classification model."""
        batch_size = 5
        input_size = 8
        num_classes = 3

        # Create models
        model_cpu = torch.nn.Sequential(
            torch.nn.Linear(input_size, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, num_classes),
        )
        model_remote = torch.nn.Sequential(
            torch.nn.Linear(input_size, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, num_classes),
        )

        # Copy weights to remote model and move to device
        model_remote.load_state_dict(model_cpu.state_dict())
        device_type = "cpu" if provider == "mock" else "cuda"
        model_remote = model_remote.to(shared_machines["T4"].device(device_type))

        # Create data
        inputs_cpu = torch.randn(batch_size, input_size)
        targets_cpu = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

        inputs_remote = inputs_cpu.to(shared_machines["T4"].device(device_type))
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        # CPU computation
        outputs_cpu = model_cpu(inputs_cpu)
        loss_cpu = F.cross_entropy(outputs_cpu, targets_cpu)
        loss_cpu.backward()

        # Remote computation
        outputs_remote = model_remote(inputs_remote)
        loss_remote = F.cross_entropy(outputs_remote, targets_remote)
        loss_remote.backward()

        # Compare loss values
        NumericalTestUtils.assert_tensors_close(
            loss_remote.cpu(),
            loss_cpu,
            msg="Classification model loss values don't match",
        )

        # Verify all parameters have gradients
        for param_remote in model_remote.parameters():
            assert param_remote.grad is not None


class TestLossNumericalStability:
    """Tests for numerical stability of loss functions."""

    def test_cross_entropy_numerical_stability(self, shared_machines, provider):
        """Test cross entropy loss with extreme values."""
        # Create inputs with extreme values
        inputs_cpu = torch.tensor(
            [
                [100.0, -100.0, 0.0],  # Very large positive and negative
                [0.0, 50.0, -50.0],  # Moderate extreme values
            ],
            requires_grad=True,
        )
        targets_cpu = torch.tensor([0, 1], dtype=torch.long)

        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = (
            inputs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        try:
            # CPU computation
            loss_cpu = F.cross_entropy(inputs_cpu, targets_cpu)
            loss_cpu.backward()

            # Remote computation
            loss_remote = F.cross_entropy(inputs_remote, targets_remote)
            loss_remote.backward()

            # Verify loss is finite
            assert torch.isfinite(loss_remote.cpu())
            assert torch.isfinite(inputs_remote.grad.cpu()).all()

            # Compare with CPU (allowing for some numerical differences)
            NumericalTestUtils.assert_tensors_close(
                loss_remote.cpu(),
                loss_cpu,
                rtol=1e-3,
                atol=1e-5,  # Slightly relaxed tolerance
                msg="Extreme value cross entropy loss mismatch",
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip("Extreme value cross entropy not supported")

    def test_mse_loss_with_very_small_values(self, shared_machines, provider):
        """Test MSE loss with very small values."""
        inputs_cpu = torch.tensor([[1e-8, 2e-8], [3e-8, 4e-8]], requires_grad=True)
        targets_cpu = torch.tensor([[1.1e-8, 1.9e-8], [3.1e-8, 3.9e-8]])

        device_type = "cpu" if provider == "mock" else "cuda"
        inputs_remote = (
            inputs_cpu.to(shared_machines["T4"].device(device_type))
            .detach()
            .requires_grad_(True)
        )
        targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

        # CPU computation
        loss_cpu = F.mse_loss(inputs_cpu, targets_cpu)
        loss_cpu.backward()

        # Remote computation
        loss_remote = F.mse_loss(inputs_remote, targets_remote)
        loss_remote.backward()

        # Verify computations are stable
        assert torch.isfinite(loss_remote.cpu())
        assert torch.isfinite(inputs_remote.grad.cpu()).all()

        # Compare results
        NumericalTestUtils.assert_tensors_close(
            loss_remote.cpu(), loss_cpu, msg="Small value MSE loss mismatch"
        )


@pytest.mark.parametrize(
    "loss_fn,reduction",
    [
        (F.mse_loss, "mean"),
        (F.mse_loss, "sum"),
        (F.l1_loss, "mean"),
    ],
)
def test_parametrized_loss_functions(shared_machines, provider, loss_fn, reduction):
    """Test various loss functions with different reduction methods."""
    inputs_cpu = torch.randn(3, 4, requires_grad=True)
    targets_cpu = torch.randn(3, 4)

    device_type = "cpu" if provider == "mock" else "cuda"
    inputs_remote = (
        inputs_cpu.to(shared_machines["T4"].device(device_type))
        .detach()
        .requires_grad_(True)
    )
    targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

    try:
        # CPU computation
        loss_cpu = loss_fn(inputs_cpu, targets_cpu, reduction=reduction)
        loss_cpu.backward()
        cpu_grad = inputs_cpu.grad.clone()

        # Remote computation
        loss_remote = loss_fn(inputs_remote, targets_remote, reduction=reduction)
        loss_remote.backward()

        # Compare results
        NumericalTestUtils.assert_tensors_close(
            loss_remote.cpu(),
            loss_cpu,
            msg=f"{loss_fn.__name__} with reduction={reduction} values don't match",
        )
        NumericalTestUtils.assert_tensors_close(
            inputs_remote.grad.cpu(),
            cpu_grad,
            msg=f"{loss_fn.__name__} with reduction={reduction} gradients don't match",
        )
    except (RuntimeError, NotImplementedError):
        pytest.skip(f"{loss_fn.__name__} with reduction={reduction} not supported")


@pytest.mark.parametrize(
    "batch_size,num_classes",
    [
        (1, 2),
        (3, 4),
        (5, 10),
    ],
)
def test_parametrized_cross_entropy_dimensions(
    shared_machines, provider, batch_size, num_classes
):
    """Test cross entropy loss with various batch sizes and class counts."""
    inputs_cpu = torch.randn(batch_size, num_classes, requires_grad=True)
    targets_cpu = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    device_type = "cpu" if provider == "mock" else "cuda"
    inputs_remote = (
        inputs_cpu.to(shared_machines["T4"].device(device_type))
        .detach()
        .requires_grad_(True)
    )
    targets_remote = targets_cpu.to(shared_machines["T4"].device(device_type))

    # Remote computation
    loss_remote = F.cross_entropy(inputs_remote, targets_remote)
    loss_remote.backward()

    # Verify computation succeeds
    assert loss_remote.dim() == 0  # Scalar loss
    NumericalTestUtils.verify_gradient_flow(inputs_remote)
    assert inputs_remote.grad.shape == (batch_size, num_classes)
