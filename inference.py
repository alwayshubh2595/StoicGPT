import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "stoicgpt-qwen-qlora"

SYSTEM_PROMPT = "You are a Stoic philosopher. Answer with wisdom, clarity, and practical guidance rooted in Stoic teachings."


def load_model(device="auto"):
    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model ready.")
    return model, tokenizer


def generate(model, tokenizer, question, max_new_tokens=300, temperature=0.7, top_p=0.9):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.15,
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    model, tokenizer = load_model()
    print("\nStoicGPT (Qwen fine-tuned) ready. Type your question (Ctrl+C to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            answer = generate(model, tokenizer, question)
            print(f"\nStoicGPT: {answer}\n")
        except KeyboardInterrupt:
            print("\nFarewell.")
            break


if __name__ == "__main__":
    main()
