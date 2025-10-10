#!/usr/bin/env python3
# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Simple MNIST training example using mycelya-torch remote execution."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mycelya_torch

# Setup remote GPU (use "mock" for local testing)
machine = mycelya_torch.RemoteMachine("modal", "T4")
device = machine.device("cuda")

# Load MNIST data with augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# CNN with BatchNorm for faster convergence
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, 1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(0.25),
    nn.Flatten(),
    nn.Linear(9216, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 10)
).to(device)

# AdamW with learning rate scheduler for faster convergence
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=3,
    steps_per_epoch=len(train_loader)
)

# Train for 3 epochs (faster convergence with better optimizer)
print(f"Training on {device}")
print(f"Total batches per epoch: {len(train_loader)}\n")

for epoch in range(3):
    model.train()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Only print at epoch end (no synchronous operations during training)
    print(f"Epoch {epoch+1}/3 completed")

print("\nTraining completed!")
