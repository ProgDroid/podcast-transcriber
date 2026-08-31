"""
Self-test the copy + validation logic against two LOCAL PersistentClient DBs.

Proves pagination (>300 records), schema round-trip, and the validation gate
before we ever touch Chroma Cloud. Two PersistentClients with DISTINCT dirs
(note: two EphemeralClients would share one backend, so we must not use those).

Usage:
  python selftest.py <base_dir>
"""

import shutil
import sys

import numpy as np
import chromadb

from chroma_migrate import (
    copy_collection,
    create_dest_collection,
    get_collection_readonly,
    validate_collection,
)

N = 350  # > 300 to exercise the get() page cap
DIM = 16
NAME = "podcast_transcripts"


def main():
    base = sys.argv[1]
    src_dir = f"{base}/selftest_src"
    dst_dir = f"{base}/selftest_dst"
    shutil.rmtree(src_dir, ignore_errors=True)
    shutil.rmtree(dst_dir, ignore_errors=True)

    # --- Seed source ---
    src = chromadb.PersistentClient(path=src_dir)
    col = src.get_or_create_collection(name=NAME, metadata={"hnsw:space": "cosine"})

    rng = np.random.default_rng(0)
    embs = rng.normal(size=(N, DIM)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)  # normalize (cosine)
    ids = [f"rec-{i}" for i in range(N)]
    docs = [f"[SPEAKER_00] document number {i} about geopolitics" for i in range(N)]
    metas = [
        {"show": "Test Show", "episode_number": str(i % 7), "date_ts": 20240000 + i}
        for i in range(N)
    ]
    col.add(ids=ids, embeddings=embs.tolist(), documents=docs, metadatas=metas)
    print(f"Seeded source with {col.count()} records (dim={DIM}).")

    # --- Migrate (same functions used against Cloud) ---
    dst = chromadb.PersistentClient(path=dst_dir)
    src_col = get_collection_readonly(src, NAME)
    dst_col = create_dest_collection(dst, src_col)
    copied = copy_collection(src_col, dst_col)
    print(f"Copied {copied} records.")

    # --- Validate ---
    problems = validate_collection(src_col, dst_col)
    if problems:
        print(f"\nSELF-TEST FAIL: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nSELF-TEST PASS: copy + schema + validation logic is sound.")


if __name__ == "__main__":
    main()
