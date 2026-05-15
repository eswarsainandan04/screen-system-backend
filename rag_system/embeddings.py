import os
import re
import requests
import chromadb
import time

# ==========================================
# HF API CONFIG
# ==========================================

HF_TOKEN = "your_token_here"


API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ==========================================
# FILE CONFIG
# ==========================================

TEXT_FILE = "../extracted_text/ml_book.txt"

CHROMA_DB_PATH = "../chroma_db"

COLLECTION_NAME = "ml_book_collection"

MAX_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 200

# ==========================================
# CLEANING + CHUNKING
# ==========================================

def _should_drop_line(line: str) -> bool:

    stripped = line.strip()
    if not stripped:
        return False

    lower = stripped.lower()
    noise_phrases = (
        "format:",
        "suggested audience",
        "author:",
        "published by",
        "copyright",
    )
    if any(phrase in lower for phrase in noise_phrases):
        return True

    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    if re.fullmatch(r"page\s*\d{1,4}", lower):
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
        return True

    return False


def _clean_text(text: str) -> str:

    cleaned_lines = []
    for raw_line in text.splitlines():
        if _should_drop_line(raw_line):
            continue
        cleaned_lines.append(raw_line.rstrip())

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_paragraphs(text: str) -> list[str]:

    return [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n+", text)
        if chunk.strip()
    ]


def _split_sentences(text: str) -> list[str]:

    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:

    cleaned = _clean_text(text)
    paragraphs = _split_paragraphs(cleaned)

    chunks = []
    buffer = []
    buffer_len = 0

    def flush_buffer(force: bool = False) -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        if buffer_len >= min_chars or force:
            chunks.append(" ".join(buffer).strip())
            buffer = []
            buffer_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            sentences = _split_sentences(paragraph)
            for sentence in sentences:
                if buffer_len + len(sentence) + 1 > max_chars:
                    flush_buffer(force=True)
                buffer.append(sentence)
                buffer_len += len(sentence) + 1
            flush_buffer(force=True)
            continue

        if buffer_len + len(paragraph) + 1 > max_chars:
            flush_buffer(force=False)
        buffer.append(paragraph)
        buffer_len += len(paragraph) + 1

    flush_buffer(force=True)
    return [chunk for chunk in chunks if chunk.strip()]

# ==========================================
# HF EMBEDDING FUNCTION
# ==========================================


def generate_embedding(text):

    payload = {
        "inputs": text
    }

    for attempt in range(3):

        try:

            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            print("STATUS:", response.status_code)

            if response.status_code != 200:

                print(response.text)

                time.sleep(5)

                continue

            result = response.json()

            return result

        except Exception as e:

            print(f"\nAttempt {attempt+1} failed:")
            print(e)

            time.sleep(5)

    return None


# ==========================================
# LOAD TEXT
# ==========================================

with open(
    TEXT_FILE,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

print("\nText loaded successfully.")

# ==========================================
# CREATE CHUNKS
# ==========================================

chunks = chunk_text(text)

print(f"\nGenerated {len(chunks)} chunks")

# ==========================================
# CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# ==========================================
# PROCESS CHUNKS
# ==========================================

for i, chunk in enumerate(chunks):

    print(f"\nEmbedding chunk {i+1}/{len(chunks)}")

    embedding = generate_embedding(chunk)

    if embedding is None:

        continue

    collection.add(
        ids=[f"chunk_{i}"],
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[{
            "source": "ds_book.txt"
        }]
    )

print("\n================================")
print("ML BOOK EMBEDDINGS COMPLETE")
print("================================")