from transformers import AutoModelForCausalLM, AutoTokenizer

import mycelya_torch

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"


# device = mycelya_torch.create_modal_machine("T4").device()
device = mycelya_torch.create_mock_machine("T4").device()
# device = "cpu"

# device = "mps" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [{"role": "user", "content": "What is gravity?"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Demonstrate metadata hash functionality
print(f"Input tensor metadata hash: {inputs.metadata_hash}")
print(f"Input tensor device: {inputs.device}")

outputs = model.generate(
    inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True
)
print(f"Output tensor metadata hash: {outputs.metadata_hash}")
print(f"{outputs=}")
print(tokenizer.decode(outputs[0]))
