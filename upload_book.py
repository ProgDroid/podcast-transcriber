"""
upload_book.py — one-off script to chunk, embed and upload a PDF book to ChromaDB.

Usage:
    python upload_book.py path/to/book.pdf

Requirements:
    pip install pdfplumber sentence-transformers chromadb torch
"""

import sys
import os
import re

# --- Configuration ---
BOOK_TITLE = "Geopolitical Alpha"
BOOK_AUTHOR = "Marko Papic"
CHROMA_TENANT = os.environ["CHROMA_TENANT"]
CHROMA_DATABASE = os.environ["CHROMA_DATABASE"]
CHROMA_API_KEY = os.environ["CHROMA_API_KEY"]

MAX_CHUNK_WORDS = 400
CHUNK_OVERLAP_WORDS = 50


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """Extract text from each page, returning list of {page, text} dicts."""
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"Extracting text from {total} pages...")
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i, "text": text})
            if i % 50 == 0:
                print(f"  {i}/{total} pages processed")
    print(f"Extracted text from {len(pages)} non-empty pages.")
    return pages


def build_chunks(pages: list[dict]) -> list[dict]:
    """
    Sliding window chunker over pages.
    Splits at MAX_CHUNK_WORDS with CHUNK_OVERLAP_WORDS overlap.
    Carries page number of the first word in each chunk.
    """
    chunks = []
    word_buffer = []  # list of (word, page_number)

    for page in pages:
        words = page["text"].split()
        for word in words:
            word_buffer.append((word, page["page"]))

    print(f"Total words: {len(word_buffer)}")

    i = 0
    while i < len(word_buffer):
        window = word_buffer[i : i + MAX_CHUNK_WORDS]
        if not window:
            break
        text = " ".join(w for w, _ in window)
        start_page = window[0][1]
        chunks.append(
            {
                "text": text,
                "metadata": {
                    "source": "book",
                    "title": BOOK_TITLE,
                    "author": BOOK_AUTHOR,
                    # Use a high date_ts so book content doesn't get buried
                    # by recency sorting — it's foundational, not time-sensitive
                    "date": "2021-01-01",  # publication year of Geopolitical Alpha
                    "date_ts": 20210101,
                    "page": start_page,
                    "show": BOOK_TITLE,  # reuse show field so filters work consistently
                    "episode_number": "N/A",
                    "episode_title": BOOK_TITLE,
                    "speaker": BOOK_AUTHOR,
                    "start_time": float(start_page),
                },
            }
        )
        i += MAX_CHUNK_WORDS - CHUNK_OVERLAP_WORDS

    print(f"Built {len(chunks)} chunks.")
    return chunks


def embed_and_store(chunks: list[dict], collection):
    """Embed chunks in batches and upsert into ChromaDB."""
    from sentence_transformers import SentenceTransformer

    print("Loading BGE embedding model (this may take a minute on first run)...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [
        f"{BOOK_TITLE.replace(' ', '_')}-p{m['page']}-{i}"
        for i, m in enumerate(metadatas)
    ]

    batch_size = 32
    total = len(texts)
    print(f"Embedding and uploading {total} chunks in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_meta = metadatas[start:end]
        batch_ids = ids[start:end]

        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"  Uploaded {end}/{total} chunks")

    print("Done.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_book.py path/to/book.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    import chromadb

    print("Connecting to ChromaDB...")
    client = chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY,
    )
    collection = client.get_or_create_collection(
        name="podcast_transcripts",  # same collection as podcasts
        metadata={"hnsw:space": "cosine"},
    )
    print("Connected.")

    pages = extract_text_by_page(pdf_path)
    chunks = build_chunks(pages)
    embed_and_store(chunks, collection)

    print(f"\nSuccessfully uploaded '{BOOK_TITLE}' by {BOOK_AUTHOR} to ChromaDB.")


if __name__ == "__main__":
    main()
