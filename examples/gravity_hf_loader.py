#!/usr/bin/env python3
"""
Minimal example showing HuggingFace model loading with direct remote weight loading
and manual token generation for "what is gravity?" question.
"""

import torch
from transformers import AutoTokenizer

import mycelya_torch


def main():
    print("🌍 Generating response to 'what is gravity?' with HuggingFace loader")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create remote machine and load model with HuggingFace integration
    machine = mycelya_torch.RemoteMachine("modal", "T4")
    machine.start()
    model = mycelya_torch.load_huggingface_model(
        "HuggingFaceTB/SmolLM2-135M-Instruct", machine, torch_dtype="float32"
    )

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
