#!/usr/bin/env python3
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import mycelya_torch

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Create mock machine and device
# machine = mycelya_torch.RemoteMachine("mock")
machine = mycelya_torch.RemoteMachine("modal", "T4")
device = machine.device()

# Load model config and create model architecture
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_config(config)

# Load state dict directly onto mycelya device
state_dict = mycelya_torch.load_huggingface_state_dict(
    model_name, device, torch_dtype="float32"
)

model.load_state_dict(state_dict, strict=False, assign=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test generation using chat template
messages = [
    {"role": "user", "content": "What is gravity?"}
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(device)

model.eval()
generated_tokens = inputs["input_ids"].clone()

# Generate 30 tokens manually
for _ in range(30):
    with torch.no_grad():
        logits = model(generated_tokens).logits[0, -1]  # Get last token logits
        next_token = torch.argmax(logits, dim=-1, keepdim=True).unsqueeze(0)
        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

response = tokenizer.decode(generated_tokens[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

print(f"Response: {response}")
