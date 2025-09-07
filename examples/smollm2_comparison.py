#!/usr/bin/env python3
"""
SmolLM2 loading with manual token generation.

Choose between two loading approaches:
1. --remote: Downloads HuggingFace model directly on remote GPU (efficient)
2. --local: Loads locally then transfers to remote device (standard PyTorch)

Usage:
    python smollm2_comparison.py --remote
    python smollm2_comparison.py --local
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mycelya_torch


def manual_generate(model, tokenizer, input_tokens, max_tokens=50):
    """Manual token-by-token generation."""
    model.eval()
    generated_tokens = input_tokens.clone()

    for _i in range(max_tokens):
        with torch.no_grad():
            logits = model(generated_tokens).logits[0, -1]  # Get last token logits
            next_token = torch.argmax(logits, dim=-1, keepdim=True).unsqueeze(0)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

    return generated_tokens


def load_model_remote(machine):
    """Load model directly on remote GPU (efficient for large models)."""
    print("🚀 Loading model directly on remote GPU...")

    # Create model architecture locally (no weights)
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        torch_dtype=torch.float32,
        device_map=None,  # Don't load weights yet
    )

    # Load weights directly onto remote GPU
    print("   Loading weights onto remote GPU...")
    remote_state_dict = mycelya_torch.load_huggingface_state_dict(
        "HuggingFaceTB/SmolLM2-135M-Instruct", machine.device(), torch_dtype="float32"
    )

    # Load remote state dict into model - should have all parameters now
    missing_keys, unexpected_keys = model.load_state_dict(
        remote_state_dict, strict=True
    )

    # Check if we still need model.to(device) - ideally this should do nothing now
    cpu_params_before = sum(
        1 for _, p in model.named_parameters() if p.device.type == "cpu"
    )
    if cpu_params_before > 0:
        print(f"   ⚠️  Still have {cpu_params_before} CPU parameters, moving to device")
        model = model.to(machine.device())
    else:
        print("   ✅ All parameters already on remote device")

    print(f"   ✅ Loaded {len(remote_state_dict)} parameters on remote GPU")
    return model


def load_model_local(machine):
    """Load model locally then transfer to remote (standard PyTorch approach)."""
    print("📦 Loading model locally then transferring to remote...")
    return AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    ).to(machine.device())


def main():
    parser = argparse.ArgumentParser(description="SmolLM2 with manual token generation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--remote", action="store_true", help="Load model directly on remote GPU"
    )
    group.add_argument(
        "--local",
        action="store_true",
        help="Load model locally then transfer to remote",
    )

    args = parser.parse_args()

    print("🌍 SmolLM2 Manual Generation")

    # Load tokenizer (stays on CPU)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create remote machine
    machine = mycelya_torch.RemoteMachine("modal", "T4")
    # machine = mycelya_torch.RemoteMachine("mock")  # For testing

    print(f"🚀 Starting remote machine: {machine.machine_id}")
    machine.start()

    # Load model based on chosen method
    if args.remote:
        model = load_model_remote(machine)
    else:  # args.local
        model = load_model_local(machine)

    # Prepare input
    input_text = "what is gravity?"
    tokens = tokenizer.encode(input_text, return_tensors="pt").to(machine.device())

    print(f"Input: '{input_text}'")
    print("Generating 50 tokens manually...")

    # Manual generation - shorter for testing
    generated_tokens = manual_generate(model, tokenizer, tokens, max_tokens=10)

    # Decode and show result
    final_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    response = final_text[len(input_text) :].strip()

    print(f"\nGenerated response:\n{response}")
    print("✅ Generation completed successfully!")


if __name__ == "__main__":
    main()
