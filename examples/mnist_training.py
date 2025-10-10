#!/usr/bin/env python3
# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Simple MNIST training example using mycelya-torch remote execution."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mycelya_torch


class MNISTNet(nn.Module):
    """CNN for MNIST classification with BatchNorm for faster convergence."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# Setup remote GPU
machine = mycelya_torch.RemoteMachine("modal", "T4")
device = machine.device("cuda")

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# Initialize model
model = MNISTNet().to(device)

# AdamW with learning rate scheduler for faster convergence
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=3,
    steps_per_epoch=len(train_loader)
)

# Train for 3 epochs
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

    print(f"Epoch {epoch+1}/3 completed")

print("\nEvaluating on test set...")

# Evaluate accuracy
model.eval()
correct = torch.tensor(0, device=device)
total = torch.tensor(0, device=device)

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum()
        total += len(target)

# Single synchronous operation at the very end
accuracy = (correct.float() / total).item() * 100
print(f"Test Accuracy: {accuracy:.2f}%")
print("\nTraining completed!")
