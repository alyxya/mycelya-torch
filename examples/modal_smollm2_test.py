import modal

app = modal.App("smollm2-test")

# Define the image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "numpy",
    "accelerate",  # Often needed for model loading
    "tokenizers",  # Required by transformers
)

@app.function(
    image=image,
    gpu="T4",
    timeout=300,  # 5 minutes timeout
)
def run_smollm2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"{checkpoint=}")

    device = "cuda"  # T4 GPU
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    messages = [{"role": "user", "content": "What is gravity?"}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(input_text)

    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)

    result = tokenizer.decode(outputs[0])
    print(result)
    return result

def main():
    with app.run():
        result = run_smollm2.remote()
        print(f"Generated text: {result}")

if __name__ == "__main__":
    main()
