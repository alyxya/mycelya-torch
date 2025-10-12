"""
Example: Remote diffusion model inference using diffusers with Mycelya-Torch

This example demonstrates:
1. Loading a diffusion model remotely on cloud GPU
2. Running image generation remotely with automatic machine inference
3. Efficient remote function execution with @remote decorator

Copyright (C) 2025 alyxya
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import torch
import mycelya_torch
from diffusers import DiffusionPipeline


@mycelya_torch.remote
def load_pipeline(model_name: str):
    """Load the diffusion pipeline remotely on cloud GPU."""
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    return pipe


@mycelya_torch.remote
def generate_image(pipe, prompt: str, height: int, width: int, seed: int):
    """Generate image from prompt using remote diffusion model."""
    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(seed)
    ).images[0]
    return image


def main():
    # Create remote machine with cloud GPU
    machine = mycelya_torch.RemoteMachine(
        "modal", "H100", pip_packages=["diffusers", "transformers", "accelerate"]
    )

    model_name = "Qwen/Qwen-Image"

    # Load pipeline remotely
    pipe = load_pipeline(model_name)

    # Generate image remotely
    prompt = "A cat holding a sign that says hello world"
    image = generate_image(pipe, prompt, height=1024, width=1024, seed=0)

    # Save generated image locally
    image.save("cat.png")


if __name__ == "__main__":
    main()
