#!/usr/bin/env python3
"""
Minimal example showing HuggingFace model loading with direct remote weight loading
and manual token generation for "what is gravity?" question.

This example demonstrates the new state dict approach for loading HuggingFace models
directly onto remote GPUs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mycelya_torch


def main():
    print("🌍 Generating response to 'what is gravity?' with remote state dict loading")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create remote machine and load model weights as state dict
    machine = mycelya_torch.RemoteMachine("modal", "T4")
    machine.start()

    print("Loading model architecture...")
    # Create model architecture locally (no weights loaded)
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        torch_dtype=torch.float32,
        device_map=None,  # Don't load weights yet
    )

    print("Loading weights remotely...")
    # Load weights directly onto remote GPU as mycelya tensors
    remote_state_dict = mycelya_torch.load_huggingface_state_dict(
        "HuggingFaceTB/SmolLM2-135M-Instruct", machine.device(), torch_dtype="float32"
    )

    print(f"Loaded {len(remote_state_dict)} parameters onto remote {machine.device()}")

    # Load the remote state dict into the local model
    # (should now contain all parameters including tied weights)
    missing_keys, _ = model.load_state_dict(remote_state_dict, strict=True)

    # Check if any parameters are still on CPU (shouldn't happen now)
    cpu_params = sum(1 for _, p in model.named_parameters() if p.device.type == "cpu")
    if cpu_params > 0:
        print(f"⚠️  Still have {cpu_params} CPU parameters, moving to device")
        model = model.to(machine.device())
    else:
        print("✅ All parameters correctly loaded on remote device")

    # Prepare input
    input_text = "what is gravity?"
    tokens = tokenizer.encode(input_text, return_tensors="pt").to(machine.device())

    print(f"Input: '{input_text}'")
    print("Generating 50 tokens...")

    model.eval()
    generated_tokens = tokens.clone()

    # Generate exactly 50 tokens
    for _i in range(50):
        with torch.no_grad():
            logits = model(generated_tokens).logits[0, -1]  # Get last token logits

            # Sample next token (greedy for reproducibility)
            next_token = torch.argmax(logits, dim=-1, keepdim=True).unsqueeze(0)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

    # Decode and show result
    final_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    response = final_text[len(input_text) :].strip()

    print(f"\nGenerated response:\n{response}")


if __name__ == "__main__":
    main()
