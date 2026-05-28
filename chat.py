import torch
import tiktoken

from model import GPTModel

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def load_model(path="model.pth", device="cpu"):
    model = GPTModel(GPT_CONFIG)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40, device="cpu"):
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -GPT_CONFIG["context_length"]:]
        with torch.no_grad():
            logits = model(input_cond)

        logits = logits[:, -1, :] / temperature

        # top-k filtering
        top_k_logits, _ = torch.topk(logits, top_k)
        logits[logits < top_k_logits[:, -1:]] = -torch.inf

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_id), dim=1)

        # stop at end-of-text
        if next_id.item() == tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]:
            break

    return tokenizer.decode(input_ids.squeeze(0).tolist())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    print("Loading model...")
    model = load_model("model.pth", device)
    print("StoicGPT ready. Type your prompt (Ctrl+C to quit).\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            response = generate(model, tokenizer, prompt, device=device)
            # print only the generated part after the prompt
            print(f"\nStoicGPT: {response[len(prompt):].strip()}\n")
        except KeyboardInterrupt:
            print("\nFarewell.")
            break


if __name__ == "__main__":
    main()
