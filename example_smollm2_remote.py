#!/usr/bin/env python3
"""
Example: Remote SmolLM2 135M LLM Loading and Inference

This example demonstrates:
1. Creating a single Modal remote machine
2. Loading SmolLM2 135M model remotely (once)
3. Performing multiple inference calls on the loaded model
4. Automatic dependency installation and single machine inference
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mycelya_torch

# Create a single Modal remote machine with A100 GPU
print("🚀 Creating Modal remote machine with A100...")
machine = mycelya_torch.RemoteMachine("modal", "A100")
print(f"✅ Created machine: {machine}")

@mycelya_torch.remote
def load_smollm2():
    """Load SmolLM2 135M model and tokenizer on the remote machine."""
    print("📦 Loading SmolLM2 135M model...")

    # Load model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )
    model = model.to("cuda")

    print("✅ Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    return model, tokenizer

@mycelya_torch.remote
def generate_response(model, tokenizer, prompt: str, max_length: int = 100):
    """Generate response from the model for the given prompt."""
    print(f"🤖 Generating response for: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")

    # Prepare the prompt with chat template if available
    try:
        # Try to use the chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback to simple prompt
        formatted_prompt = f"User: {prompt}\nAssistant:"

    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    elif formatted_prompt in response:
        response = response[len(formatted_prompt):].strip()

    return response

def main():
    """Main function to demonstrate the remote LLM example."""
    print("\n" + "="*60)
    print("🦙 SmolLM2 Remote Inference Example")
    print("="*60)

    # Step 1: Load the model once
    print("\n📋 Step 1: Loading SmolLM2 135M model remotely...")
    model, tokenizer = load_smollm2()
    print("✅ Model and tokenizer loaded successfully!")

    # Step 2: Perform multiple inference calls
    print("\n📋 Step 2: Performing multiple inference calls...")

    test_prompts = [
        "What is machine learning?",
        "Write a haiku about coding.",
        "Explain Python in one sentence.",
        "What's the meaning of life?",
        "How do neural networks work?"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔮 Inference {i}/5:")
        print(f"📝 Prompt: {prompt}")

        response = generate_response(model, tokenizer, prompt, max_length=80)
        print(f"🤖 Response: {response}")
        print("-" * 50)

    print("\n✅ Example completed successfully!")
    print("🎉 All inferences used the same loaded model on the remote machine!")

if __name__ == "__main__":
    main()
