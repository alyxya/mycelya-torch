#!/usr/bin/env python3
"""
Minimal example showing HuggingFace model loading with direct remote weight loading
and manual token generation for "what is gravity?" question.
"""

import torch
from transformers import AutoTokenizer

import mycelya_torch
import mycelya_torch.huggingface as mhf


def main():
    print("🌍 Generating response to 'what is gravity?' with HuggingFace loader")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create remote machine and load model with new HuggingFace integration
    machine = mycelya_torch.create_modal_machine("T4")
    model = mhf.load_model_remote(
        "HuggingFaceTB/SmolLM2-135M-Instruct", machine, torch_dtype=torch.float32
    )

    # Show model parameters with metadata hashes
    print("\nModel parameters with metadata hashes:")
    for name, param in model.named_parameters():
        if "embed" in name or "lm_head" in name:  # Show key parameters
            print(f"  {name}: hash={param.metadata_hash}, shape={param.shape}, device={param.device}")

    # Prepare input
    input_text = "what is gravity?"
    tokens = tokenizer.encode(input_text, return_tensors="pt").to(machine.device())

    print(f"\nInput: '{input_text}'")
    print(f"Input tokens metadata hash: {tokens.metadata_hash}")
    print("Generating 50 tokens...")

    model.eval()
    generated_tokens = tokens.clone()
    print(f"Generated tokens metadata hash: {generated_tokens.metadata_hash}")

    # Generate exactly 50 tokens
    for i in range(50):
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
