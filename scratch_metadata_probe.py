"""Step 5 probe: does a metadata-only write preserve the document and the
embedding on Chroma Cloud? Runs against a throwaway collection, never
`podcast_transcripts_v2`. Deletes the scratch collection at the end.

Two calls are probed because the first attempt (a metadata-only `upsert`,
matching the plan in the task brief) turned out to be REJECTED client-side by
chromadb 1.5.9 before any network call: "Exactly one of documents, images
must be provided in upsert." `update()`'s validation only requires
documents/images when there are no embeddings AND no documents/images at all
being embedded -- a metadata-only `update()` skips that check entirely.
"""

from __future__ import annotations

import os

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("chromadb==1.5.9")

app = modal.App("speaker-backfill-probe")

SCRATCH_NAME = "scratch_speaker_id_probe"


def _eq_embeddings(a, b):
    if a is None or b is None:
        return a == b
    return [list(x) for x in a] == [list(x) for x in b]


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=300,
)
def run_probe() -> dict:
    import chromadb

    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    collection = client.get_or_create_collection(
        name=SCRATCH_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    result: dict = {}
    try:
        collection.upsert(
            ids=["probe::0"],
            embeddings=[[0.1] * 8],
            documents=["[SPEAKER_00] hello"],
            metadatas=[{"show": "S", "episode_number": "1", "date": "2026-01-01"}],
        )

        # --- Sub-probe A: metadata-only upsert(), as originally planned ---
        try:
            collection.upsert(ids=["probe::0"], metadatas=[{"speaker_id": "Someone"}])
            result["upsert_metadata_only"] = "SUCCEEDED (unexpected)"
        except ValueError as e:
            result["upsert_metadata_only"] = f"REJECTED: {e}"
        print("metadata-only upsert():", result["upsert_metadata_only"])

        # --- Sub-probe B: metadata-only update() ---
        before = collection.get(
            ids=["probe::0"], include=["documents", "embeddings", "metadatas"]
        )
        collection.update(ids=["probe::0"], metadatas=[{"speaker_id": "Someone"}])
        after = collection.get(
            ids=["probe::0"], include=["documents", "embeddings", "metadatas"]
        )

        doc_preserved = before["documents"] == after["documents"]
        emb_preserved = _eq_embeddings(before["embeddings"], after["embeddings"])
        merged = after["metadatas"][0]

        result["update_document_preserved"] = doc_preserved
        result["update_embedding_preserved"] = emb_preserved
        result["update_merged_metadata"] = merged

        print("update() document preserved:", doc_preserved)
        print("update() embedding preserved:", emb_preserved)
        print("update() metadata merged:", merged)

        return result
    finally:
        client.delete_collection(SCRATCH_NAME)
        print(f"deleted scratch collection {SCRATCH_NAME!r}")


@app.local_entrypoint()
def main() -> None:
    print(run_probe.remote())
