# StoicGPT — The Oracle

A small GPT (~15M parameters) trained from scratch on the writings of Marcus Aurelius, Seneca, and Epictetus. Ask it anything and receive wisdom in the voice of the ancient Stoics.

---

## Texts used for training

- **Marcus Aurelius** — Meditations (via Project Gutenberg)
- **Seneca** — Letters to Lucilius, Seneca's Morals
- **Epictetus** — Discourses, Enchiridion

---

## Project structure

```
StoicGPT/
├── Stoic Knowledge/        # Training texts
├── configs/config.yaml     # Model hyperparameters
├── attention.py            # Multi-head attention
├── model.py                # GPT model architecture
├── data.py                 # Dataset and dataloader
├── train.py                # Training loop
├── chat.py                 # CLI chat interface
├── app.py                  # Flask web app
└── templates/index.html    # Stoic Oracle UI
```

---

## Setup

```bash
git clone https://github.com/alwayshubh2595/StoicGPT.git
cd StoicGPT
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

- Trains for 10 epochs on the Stoic corpus (~400k tokens)
- Saves the model to `model.pth` when done
- Prints loss every 5 steps and generates a sample sentence each epoch
- **CPU**: ~2–4 hours (Intel Core Ultra or equivalent)
- **GPU (T4 on Kaggle)**: ~15 minutes

### Training on Kaggle (recommended)

1. Go to [kaggle.com](https://kaggle.com) and create a new Notebook
2. Set Accelerator to **GPU T4 x2** in session options
3. Upload all project files
4. Run `!pip install tiktoken` then `!python train.py`
5. Download the output `model.pth` into your local project folder

---

## Running the web app

Make sure `model.pth` is in the project root, then:

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

The **Stoic Oracle** UI lets you ask questions and receive responses with a typewriter effect. Full chat history is preserved for the session.

---

## CLI chat (no browser)

```bash
python chat.py
```

Type your prompt and press Enter. `Ctrl+C` to quit.

---

## Model config

| Parameter | Value |
|---|---|
| Layers | 6 |
| Attention heads | 6 |
| Embedding dim | 384 |
| Context length | 256 tokens |
| Vocab size | 50,257 (GPT-2 tokenizer) |
| Parameters | ~15M |

---

## Note on model quality

This is a language model trained from scratch on a small corpus — it is not fine-tuned on instruction following. Responses will be Stoic in tone and style but may not directly answer questions. Think of it as a Stoic text generator, not a chatbot.
