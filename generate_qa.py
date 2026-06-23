import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

STOIC_DIR = "Stoic Knowledge"
OUTPUT_FILE = "data/stoic_qa.jsonl"

SYSTEM_PROMPT = """You are an expert on Stoic philosophy. Given a passage from a Stoic text, generate 3-5 high-quality question-answer pairs.

Rules:
- Questions should be the kind a curious person would ask about life, suffering, virtue, emotions, death, adversity, discipline, etc.
- Answers must be grounded in the passage but written as a Stoic philosopher speaking directly to the reader. Use "you" voice.
- Answers should be 2-5 sentences. Wise, practical, not preachy.
- Do NOT mention "the passage" or "the text" — the philosopher is speaking from their own wisdom.
- Vary question types: some about practical advice, some about concepts, some about emotions.

Respond ONLY with a JSON array of objects like:
[{"question": "...", "answer": "..."}]"""


def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 100:
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def generate_qa_from_chunk(chunk, source_name):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Source: {source_name}\n\nPassage:\n{chunk}"}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        pairs = json.loads(content)
        return pairs
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Error: {e}")
        return []


def main():
    files = {
        "The Project Gutenberg.txt": "Marcus Aurelius",
        "Letters.txt": "Seneca",
        "Seneca_s Morals.txt": "Seneca",
        "Discourses.txt": "Epictetus",
        "Enchridion.txt": "Epictetus",
    }

    # resume from existing progress if any
    existing_pairs = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_pairs = [json.loads(line) for line in f if line.strip()]
        print(f"Resuming — found {len(existing_pairs)} existing QA pairs")

    all_pairs = existing_pairs
    processed_chunks = len(all_pairs) // 4  # rough estimate: ~4 pairs per chunk

    for fname, philosopher in files.items():
        path = os.path.join(STOIC_DIR, fname)
        if not os.path.exists(path):
            print(f"Skipping {fname} — not found")
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"\n{philosopher} ({fname}): {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            pairs = generate_qa_from_chunk(chunk, philosopher)
            for pair in pairs:
                pair["source"] = philosopher
                all_pairs.append(pair)

                # save every pair immediately
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(chunks)} chunks — {len(all_pairs)} QA pairs so far")

            time.sleep(0.5)

        print(f"  Done — {len(all_pairs)} total QA pairs")

    print(f"\nFinished! {len(all_pairs)} QA pairs saved to {OUTPUT_FILE}")

    # save chat-formatted version for fine-tuning
    formatted = []
    for pair in all_pairs:
        formatted.append({
            "messages": [
                {"role": "system", "content": "You are a Stoic philosopher. Answer with wisdom, clarity, and practical guidance rooted in Stoic teachings."},
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]}
            ]
        })

    with open("data/stoic_qa_chat.jsonl", "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Chat-formatted version saved to stoic_qa_chat.jsonl")


if __name__ == "__main__":
    main()
