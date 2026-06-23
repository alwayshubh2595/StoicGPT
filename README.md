# StoicGPT

A language model project exploring the full spectrum of LLM development — from building a transformer by hand to fine-tuning a modern open-source model — trained exclusively on ancient Stoic philosophy.

**[Live Demo](https://huggingface.co/spaces/ShubhWorking/StoicGPT)**

---

## Two Models, One Philosophy

### The Forge — Built from Scratch
A 15M parameter decoder-only transformer implemented entirely in PyTorch, with no library abstractions. Custom multi-head causal self-attention, learned positional embeddings, pre-norm residual connections, GELU activations, and KV caching for efficient inference.

- **Architecture:** 6 layers, 6 heads, 384 embedding dim, 256 context length
- **Tokenizer:** GPT-2 BPE (50,257 vocab) via tiktoken
- **Training:** 10 epochs on ~400K tokens of Stoic text, AdamW optimizer
- **Inference:** KV cache with prefill/decode separation for 2–3x speedup

### The Oracle — Fine-Tuned Qwen 2.5 1.5B
Qwen 2.5 1.5B-Instruct fine-tuned with QLoRA (4-bit NF4 quantization + LoRA rank 16) on 7,348 Stoic Q&A pairs. Speaks with clarity and practical wisdom grounded in the original texts.

- **Base model:** Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0)
- **Fine-tuning:** QLoRA, 3 epochs, RTX 3090 via RunPod (~25 min)
- **Dataset:** Generated from the Stoic corpus using GPT-4o-mini
- **Adapter weights:** [ShubhWorking/stoic-gpt-qwen-qlora](https://huggingface.co/ShubhWorking/stoic-gpt-qwen-qlora)

---

## Training Corpus

~400,000 tokens from five public domain Stoic texts:

| Text | Author | Tokens |
|------|--------|--------|
| Meditations | Marcus Aurelius | ~90K |
| Letters to Lucilius | Seneca | ~105K |
| Seneca's Morals | Seneca | ~135K |
| Discourses | Epictetus | ~55K |
| Enchiridion | Epictetus | ~15K |

---

## Project Structure

```
StoicGPT/
├── attention.py          # Multi-head causal self-attention with KV cache
├── model.py              # GPTModel — LayerNorm, GELU, FeedForward, TransformerBlock
├── data.py               # Dataset, dataloader, text loading
├── train.py              # Training loop for the from-scratch model
│
├── generate_qa.py        # Q&A pair generation from Stoic texts via OpenAI API
├── finetune.py           # QLoRA fine-tuning script for Qwen 2.5 1.5B
├── inference.py          # CLI inference for the fine-tuned model
│
├── app.py                # Flask web app serving both models
├── chat.py               # CLI chat interface for the from-scratch model
├── templates/
│   └── index.html        # Multi-page Stoic-themed UI
├── static/
│   └── css/style.css     # Design system
│
├── Stoic Knowledge/      # Source texts (public domain)
├── data/
│   ├── stoic_qa.jsonl    # 7,348 raw Q&A pairs
│   └── stoic_qa_chat.jsonl  # Chat-formatted for fine-tuning
├── configs/
│   └── config.yaml       # Model hyperparameters
│
├── deploy/               # HF Spaces deployment files
│   ├── Dockerfile
│   ├── download_models.py
│   ├── requirements-deploy.txt
│   └── README_SPACE.md
│
├── requirements.txt
└── LICENSE
```

---

## Setup

```bash
git clone https://github.com/alwayshubh2595/StoicGPT.git
cd StoicGPT
pip install -r requirements.txt
```

---

## Training the From-Scratch Model

```bash
python train.py
```

Trains for 10 epochs, saves `model.pth`. ~2–4 hours on CPU, ~15 minutes on GPU.

## Generating Q&A Dataset

```bash
export OPENAI_API_KEY="your-key"
python generate_qa.py
```

Generates ~7K Stoic Q&A pairs from the source texts. Cost: ~$0.40.

## Fine-Tuning (requires GPU)

```bash
pip install transformers peft bitsandbytes accelerate datasets
python finetune.py
```

QLoRA fine-tuning on Qwen 2.5 1.5B. ~25 min on RTX 3090.

## Running the Web App

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860). Features:
- Dual-tab chat (The Oracle + The Forge)
- Interactive architecture visualization
- Stoicism knowledge base
- Bibliography with technical references

---

## Technical Highlights

- **KV Caching:** Implemented from scratch in the attention layer — prefill phase processes the full prompt, decode phase passes only the new token and reuses cached Keys/Values
- **Pre-Norm Transformer:** LayerNorm applied before attention and FFN sub-layers for stable training (Xiong et al., 2020)
- **QLoRA:** 4-bit NF4 quantization with double quantization + LoRA adapters targeting all attention and MLP layers
- **Crash-resistant data pipeline:** Q&A generation script saves after every chunk and resumes from existing progress

---

## References

- Raschka, S. *Build a Large Language Model (From Scratch).* Manning, 2024.
- Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
- Radford, A. et al. *Language Models are Unsupervised Multitask Learners.* OpenAI, 2019.
- Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized Language Models.* NeurIPS 2023.
- Qwen Team. *Qwen 2.5: A Party of Foundation Models.* Alibaba Cloud, 2024.

---

## License

Apache 2.0

---

Built by [Shubh Srivastava](https://www.linkedin.com/in/shubh-srivastava-06a636322)
