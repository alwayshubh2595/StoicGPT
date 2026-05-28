from flask import Flask, request, jsonify, render_template
import torch
import tiktoken
from model import GPTModel

app = Flask(__name__)

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

print("Loading model...")
model = GPTModel(GPT_CONFIG)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()
print("Model ready.")


def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -GPT_CONFIG["context_length"]:]
        with torch.no_grad():
            logits = model(input_cond)

        logits = logits[:, -1, :] / temperature
        top_k_logits, _ = torch.topk(logits, top_k)
        logits[logits < top_k_logits[:, -1:]] = -torch.inf
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_id), dim=1)

        if next_id.item() == eos_id:
            break

    full_text = tokenizer.decode(input_ids.squeeze(0).tolist())
    return full_text[len(prompt):].strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400
    response = generate(prompt)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=False)
