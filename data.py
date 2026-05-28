import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

STOIC_FILES = [
    "Stoic Knowledge/The Project Gutenberg.txt",
    "Stoic Knowledge/Letters.txt",
    "Stoic Knowledge/Seneca_s Morals.txt",
    "Stoic Knowledge/Discourses.txt",
    "Stoic Knowledge/Enchridion.txt",
]


def load_text(files=STOIC_FILES):
    combined = ""
    for fname in files:
        with open(fname, "r", encoding="utf-8") as f:
            combined += f.read() + "\n<|endoftext|>\n"
    return combined


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
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)
