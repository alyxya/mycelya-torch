"""
Example: Remote LLM inference using transformers with Mycelya-Torch

This example demonstrates:
1. Loading a language model remotely on cloud GPU
2. Running inference remotely with automatic machine inference
3. Efficient remote function execution with @remote decorator

Copyright (C) 2025 alyxya
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import mycelya_torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@mycelya_torch.remote
def load_model(model_name: str):
    """Load the tokenizer and model remotely on cloud GPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer


@mycelya_torch.remote
def generate_text(model, tokenizer, prompt: str):
    """Conduct text completion remotely on cloud GPU."""
    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content


def main():
    # Create remote machine with cloud GPU
    machine = mycelya_torch.RemoteMachine(
        "modal", "A100", pip_packages=["transformers", "accelerate"]
    )

    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    # load the tokenizer and the model remotely
    model, tokenizer = load_model(model_name)

    # conduct text completion remotely
    prompt = "Give me a short introduction to large language model."
    content = generate_text(model, tokenizer, prompt)

    print("content:", content)


if __name__ == "__main__":
    main()
