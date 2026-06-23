from huggingface_hub import hf_hub_download, snapshot_download

print("Downloading Forge model (from-scratch)...")
hf_hub_download(
    repo_id="ShubhWorking/stoicgpt-forge-model",
    filename="model.pth",
    local_dir=".",
)

print("Downloading Oracle adapter (QLoRA)...")
snapshot_download(
    repo_id="ShubhWorking/stoic-gpt-qwen-qlora",
    local_dir="stoic-gpt-qwen-qlora",
    ignore_patterns=["checkpoint-*"],
)

print("All models downloaded.")
