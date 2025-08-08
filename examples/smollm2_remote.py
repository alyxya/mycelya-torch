"""
SmolLM2 example using optimized remote HuggingFace model loading.

This example demonstrates the new efficient model loading that downloads
HuggingFace models directly on the remote GPU, avoiding slow network uploads.
"""

from transformers import AutoTokenizer

import mycelya_torch

# HuggingFace checkpoint to load
checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Create remote machine
machine = mycelya_torch.create_modal_machine("T4")
# machine = mycelya_torch.create_mock_machine("T4")  # For testing

print(f"🚀 Starting remote machine: {machine.machine_id}")
machine.start()

try:
    print(f"📦 Loading HuggingFace model {checkpoint} directly on remote GPU...")

    # Load model with optimized remote loading
    # This downloads weights directly on the remote GPU, then creates local stubs
    model = mycelya_torch.load_huggingface_model(
        checkpoint=checkpoint,
        machine=machine,
        torch_dtype="auto",  # Uses float16 on GPU, float32 on CPU
    )

    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load tokenizer (this stays on CPU as usual)
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Prepare input
    messages = [{"role": "user", "content": "What is gravity?"}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"Input: {input_text}")

    # Tokenize and move to remote device
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    inputs = inputs.to(machine.device())

    print("🧠 Generating response with remote model...")

    # Generate response (all computation happens on remote GPU)
    with machine.device():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")

    print("✅ Generation completed successfully!")

finally:
    print("🛑 Stopping remote machine...")
    machine.stop()

print("🎉 Example completed!")
