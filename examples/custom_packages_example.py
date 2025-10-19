# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Example demonstrating custom package dependencies in @remote decorator.

This example shows how to use the packages parameter to specify custom
package dependencies that override auto-detection.
"""

import torch

import mycelya_torch


def main():
    """Demonstrate custom package dependencies in remote functions."""

    # Create remote machine
    machine = mycelya_torch.RemoteMachine("mock")
    device = machine.device("cpu")

    # Example 1: Auto-detection (default behavior)
    print("Example 1: Auto-detection (default)")

    @mycelya_torch.remote
    def auto_detect_function(x: torch.Tensor) -> torch.Tensor:
        """Function that relies on auto-detection of dependencies."""
        return x * 2.0

    x = torch.randn(3, 3, device=device)
    result1 = auto_detect_function(x)
    print(f"Result shape: {result1.shape}")

    # Example 2: Custom packages override auto-detection
    print("\nExample 2: Custom packages (overrides auto-detection)")

    @mycelya_torch.remote(packages=["numpy>=1.24.0"])
    def custom_packages_function(x: torch.Tensor) -> torch.Tensor:
        """Function with custom package dependencies specified."""
        # Package 'numpy>=1.24.0' will be installed
        # Auto-detection is disabled when packages is specified
        return torch.relu(x)

    y = torch.randn(3, 3, device=device) - 0.5
    result2 = custom_packages_function(y)
    print(f"Result shape: {result2.shape}")

    # Example 3: Empty packages list (no dependencies installed)
    print("\nExample 3: Empty packages list (no installations)")

    @mycelya_torch.remote(packages=[])
    def no_packages_function(x: torch.Tensor) -> torch.Tensor:
        """Function with no package dependencies."""
        # No packages will be installed (overrides auto-detection)
        return x + 1.0

    z = torch.randn(3, 3, device=device)
    result3 = no_packages_function(z)
    print(f"Result shape: {result3.shape}")

    # Example 4: Multiple custom packages
    print("\nExample 4: Multiple custom packages")

    @mycelya_torch.remote(packages=["numpy", "cloudpickle"])
    def multiple_packages_function(x: torch.Tensor) -> torch.Tensor:
        """Function with multiple custom package dependencies."""
        # Both numpy and cloudpickle will be installed
        return torch.softmax(x, dim=-1)

    w = torch.randn(3, 3, device=device)
    result4 = multiple_packages_function(w)
    print(f"Result shape: {result4.shape}")

    # Clean up resources
    machine.stop()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
