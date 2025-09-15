# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import mycelya_torch


class SimpleTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        self.input_proj = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(x.size(0), -1, -1)

        x = self.input_proj(x) + pos_enc
        x = self.transformer(x)
        x = self.output_proj(x)
        return x


@pytest.fixture(scope="session")
def t4_device():
    return mycelya_torch.RemoteMachine("modal", "T4")


def test_simple_transformer_forward_backward(t4_device):
    device = t4_device.device("cuda")

    d_model = 512
    nhead = 8
    num_layers = 2
    batch_size = 4
    seq_len = 16

    model = SimpleTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)
    model = model.to(device)

    input_tensor = torch.randn(
        batch_size, seq_len, d_model, device=device, requires_grad=True
    )
    target = torch.randn(batch_size, seq_len, d_model, device=device)

    assert input_tensor.device == device
    assert target.device == device
    assert all(p.device == device for p in model.parameters())

    output = model(input_tensor)

    assert output.device == device
    assert output.shape == (batch_size, seq_len, d_model)

    loss = F.mse_loss(output, target)
    assert loss.device == device

    loss.backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.device == device

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert param.grad.device == device, f"Parameter {name} gradient on wrong device"


def test_transformer_attention_forward_backward(t4_device):
    device = t4_device.device("cuda")

    d_model = 256
    nhead = 4
    batch_size = 2
    seq_len = 8

    attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
    attention = attention.to(device)

    query = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    key = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    value = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    target = torch.randn(batch_size, seq_len, d_model, device=device)

    assert all(p.device == device for p in attention.parameters())

    output, weights = attention(query, key, value)

    assert output.device == device
    assert weights.device == device
    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, seq_len, seq_len)

    loss = F.mse_loss(output, target)
    loss.backward()

    assert query.grad is not None and query.grad.device == device
    assert key.grad is not None and key.grad.device == device
    assert value.grad is not None and value.grad.device == device

    for name, param in attention.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert param.grad.device == device


def test_transformer_encoder_layer_forward_backward(t4_device):
    device = t4_device.device("cuda")

    d_model = 128
    nhead = 2
    dim_feedforward = 512
    batch_size = 3
    seq_len = 10

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        batch_first=True,
    )
    encoder_layer = encoder_layer.to(device)

    input_tensor = torch.randn(
        batch_size, seq_len, d_model, device=device, requires_grad=True
    )
    target = torch.randn(batch_size, seq_len, d_model, device=device)

    assert all(p.device == device for p in encoder_layer.parameters())

    output = encoder_layer(input_tensor)

    assert output.device == device
    assert output.shape == (batch_size, seq_len, d_model)

    loss = F.mse_loss(output, target)
    loss.backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.device == device

    for name, param in encoder_layer.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert param.grad.device == device


def test_linear_layers_forward_backward(t4_device):
    device = t4_device.device("cuda")

    batch_size = 4
    input_dim = 512
    hidden_dim = 256
    output_dim = 128

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
    )
    model = model.to(device)

    input_tensor = torch.randn(batch_size, input_dim, device=device, requires_grad=True)
    target = torch.randn(batch_size, output_dim, device=device)

    assert all(p.device == device for p in model.parameters())

    output = model(input_tensor)

    assert output.device == device
    assert output.shape == (batch_size, output_dim)

    loss = F.mse_loss(output, target)
    loss.backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.device == device

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert param.grad.device == device
