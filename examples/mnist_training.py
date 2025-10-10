#!/usr/bin/env python3
# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Simple MNIST training example using mycelya-torch remote execution."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mycelya_torch


def main():
    # Setup remote GPU (use "mock" for local testing)
    machine = mycelya_torch.RemoteMachine("modal", "T4")
    device = machine.device("cuda")

    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Simple 2-layer network
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"[{batch_idx * len(data):5d}/{len(train_data)}] Loss: {loss.item():.4f}")

    print("Training completed!")


if __name__ == "__main__":
    main()
