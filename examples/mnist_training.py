#!/usr/bin/env python3
# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Simple MNIST training example using mycelya-torch remote execution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mycelya_torch


# Simple CNN model
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():
    # Configuration
    batch_size = 64
    epochs = 3
    lr = 0.001

    # Create remote machine with cloud GPU
    # Use "mock" for local testing, or "modal" with GPU type (e.g., "T4", "A100")
    machine = mycelya_torch.RemoteMachine("mock")
    device = machine.device("cpu")  # Use "cuda" for Modal with GPU

    print(f"Training on device: {device}")

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Transfer data to remote device
            data, target = data.to(device), target.to(device)

            # Forward pass, loss computation, and backprop all happen remotely
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print progress
            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
