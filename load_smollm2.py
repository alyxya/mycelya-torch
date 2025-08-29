#!/usr/bin/env python3
import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Load model and tokenizer
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_config(config)

# Download only the model weights file
model_file = hf_hub_download(repo_id=model_name, filename="model.safetensors")
state_dict = safetensors.torch.load_file(model_file)

model.load_state_dict(state_dict, strict=False)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test generation using chat template
messages = [
    {"role": "user", "content": "What is gravity?"}
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_new_tokens=30, do_sample=False)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print(f"Response: {response}")
