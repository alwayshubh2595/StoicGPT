import os
import torch
import tiktoken
from flask import Flask, request, jsonify, render_template
from model import GPTModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)

device = torch.device("cpu")

# ── Tab 1: The Forge (from-scratch 15M model) ──

FORGE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

forge_tokenizer = tiktoken.get_encoding("gpt2")

print("Loading Forge model (from scratch)...")
forge_model = GPTModel(FORGE_CONFIG)
forge_model.load_state_dict(torch.load("model.pth", map_location=device))
forge_model.to(device)
forge_model.eval()
print("Forge model ready.")


def generate_forge(prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    input_ids = torch.tensor(forge_tokenizer.encode(prompt)).unsqueeze(0).to(device)
    eos_id = forge_tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -FORGE_CONFIG["context_length"]:]
        with torch.no_grad():
            logits = forge_model(input_cond)

        logits = logits[:, -1, :] / temperature
        top_k_logits, _ = torch.topk(logits, top_k)
        logits[logits < top_k_logits[:, -1:]] = -torch.inf
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_id), dim=1)

        if next_id.item() == eos_id:
            break

    full_text = forge_tokenizer.decode(input_ids.squeeze(0).tolist())
    return full_text[len(prompt):].strip()


# ── Tab 2: The Oracle (fine-tuned Qwen) ──

ORACLE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ORACLE_ADAPTER = "stoic-gpt-qwen-qlora"
ORACLE_SYSTEM = "You are a Stoic philosopher. Answer with wisdom, clarity, and practical guidance rooted in Stoic teachings."

print("Loading Oracle model (fine-tuned Qwen)...")
oracle_base = AutoModelForCausalLM.from_pretrained(
    ORACLE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map=None,
    trust_remote_code=True,
)
oracle_model = PeftModel.from_pretrained(oracle_base, ORACLE_ADAPTER)
oracle_model.to(device)
oracle_model.eval()

oracle_tokenizer = AutoTokenizer.from_pretrained(ORACLE_ADAPTER, trust_remote_code=True)
if oracle_tokenizer.pad_token is None:
    oracle_tokenizer.pad_token = oracle_tokenizer.eos_token
print("Oracle model ready.")


def generate_oracle(question, max_new_tokens=300, temperature=0.7, top_p=0.9):
    messages = [
        {"role": "system", "content": ORACLE_SYSTEM},
        {"role": "user", "content": question},
    ]
    text = oracle_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = oracle_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = oracle_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.15,
        )

    response = oracle_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ── Routes ──

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    tab = data.get("tab", "oracle")

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    if tab == "forge":
        response = generate_forge(prompt)
    else:
        response = generate_oracle(prompt)

    return jsonify({"response": response})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
