#!/usr/bin/env python3
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import mycelya_torch

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Create mock machine and device
machine = mycelya_torch.RemoteMachine("mock")
# machine = mycelya_torch.RemoteMachine("modal", "T4")
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

with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=30, do_sample=False)
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

print(f"Response: {response}")
