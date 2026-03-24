from importlib.metadata import version
import os
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

# Quick environment sanity checks.
print("tiktoken version:", version("tiktoken"))
print("PyTorch version:", torch.__version__)

# Merge all Stoic texts into one corpus.
files = [
    "The Project Gutenberg.txt",
    "Letters.txt",
    "Seneca's Morals.txt",
    "Discourses.txt",
    "Enchridion.txt",
]

combined = ""
for fname in files:
    with open(fname, "r", encoding="utf-8") as f:
        combined += f.read() + "\n<|endoftext|>\n"

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(combined)

print(f"Corpus created! Total characters: {len(combined):,}")

# Load the merged corpus.
with open("corpus.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# GPT-2 tokenizer.
tokenizer = tiktoken.get_encoding("gpt2")


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        if len(token_ids) <= max_length:
            raise ValueError("Tokenized text must contain at least max_length + 1 tokens.")

        for start_idx in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[start_idx : start_idx + max_length]
            target_chunk = token_ids[start_idx + 1 : start_idx + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128,
                          shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)


# Test the dataloader.
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# Test the embedding layers.
vocab_size = 50257
embedding_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
pos_embedding_layer = torch.nn.Embedding(4, embedding_dim)  # 4 = max_length used above

token_embeds = token_embedding_layer(inputs)
pos_embeds = pos_embedding_layer(torch.arange(inputs.shape[1]))
x = token_embeds + pos_embeds
print("Embedding output shape:", x.shape)  # expected: (8, 4, 256)