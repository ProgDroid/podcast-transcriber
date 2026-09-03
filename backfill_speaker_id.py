"""Write `speaker_id` into Chroma. Add a key; never touch another.

WHY ADDITIVE IS A CORRECTNESS REQUIREMENT, not a preference. `chunking.py:84`
bakes `[SPEAKER_01] ` into the document text and `rendering.py` strips it
using the metadata value, so overwriting `speaker` with a person's name leaves
the raw cluster label visible in the text it was meant to replace. And per
`corpus/writing.py`, Chroma MERGES metadata on upsert -- a key written here
can never be removed by rewriting the episode, only by deleting the record.
Both point the same way: add `speaker_id`, leave everything else alone.

This never writes a document and never writes an embedding.

WRITES VIA `update()`, NOT `upsert()` -- measured, not assumed (spec UNKNOWN
6.3, closed here). A metadata-only `collection.upsert(ids=..., metadatas=...)`
against real Chroma Cloud (chromadb 1.5.9) is REJECTED client-side, before any
network call: "Exactly one of documents, images must be provided in upsert."
`update()` has no such requirement -- its validation only demands
documents/images when there ARE documents/images to embed, so a call carrying
only `ids` and `metadatas` skips that path entirely. Measured against a
throwaway `scratch_speaker_id_probe` collection: `update()` preserves the
existing document and the existing embedding, and merges the metadata, same as
`upsert()` does when documents/embeddings ARE supplied. See
`scratch_metadata_probe.py` and the Task 8 report for the raw probe output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from corpus.store import BATCH, episode_where

# BATCH comes from corpus/store.py:20 (250), under a measured Cloud cap of
# 300. A second constant that happened to differ would be a silent
# inconsistency between two writers against the same collection.


def plan_updates(collection, assignments: list[dict]) -> list[tuple[str, dict]]:
    """Every (chunk_id, {"speaker_id": person}) this backfill would write.

    A cluster with `person: None` yields NOTHING. Writing a placeholder would
    create a value a filter could match, which is the false confidence the
    whole refusal rule exists to avoid.
    """
    updates: list[tuple[str, dict]] = []
    for record in assignments:
        missing = {"show", "episode_number", "date"} - set(record)
        if missing:
            # Refuse rather than skip. A record without the triple would build
            # a filter that selects nothing, plan zero updates, and report
            # success -- the failure mode with no symptom, and the one that
            # broke the first draft of this task.
            raise ValueError(
                f"assignment is missing {sorted(missing)}: {record.get('episode')!r}"
            )
        person = record.get("person")
        if person is None:
            continue
        where = episode_where(record["show"], record["episode_number"], record["date"])
        where = {"$and": [*where["$and"], {"speaker": {"$eq": record["cluster"]}}]}
        offset = 0
        while True:
            got = collection.get(
                where=where,
                limit=BATCH,
                offset=offset,
                include=["metadatas"],
            )
            ids = got["ids"]
            if not ids:
                break
            updates.extend((chunk_id, {"speaker_id": person}) for chunk_id in ids)
            if len(ids) < BATCH:
                break
            offset += BATCH
    return updates


def apply_updates(collection, updates: list[tuple[str, dict]], *, expected: int) -> int:
    """Write the planned metadata, refusing if the count moved.

    The count is asserted rather than trusted: a collection that changed since
    the plan was built should cause a refusal, not an unknown-size write.

    Uses `collection.update()`, not `.upsert()` -- see the module docstring.
    A metadata-only `upsert()` is rejected client-side by chromadb 1.5.9
    before any request reaches Cloud.
    """
    if len(updates) != expected:
        raise RuntimeError(
            f"REFUSING: expected {expected} updates, planned {len(updates)}. "
            f"The collection moved since the plan was built; re-plan rather "
            f"than write."
        )
    written = 0
    for start in range(0, len(updates), BATCH):
        batch = updates[start : start + BATCH]
        collection.update(
            ids=[chunk_id for chunk_id, _ in batch],
            metadatas=[metadata for _, metadata in batch],
        )
        written += len(batch)
    return written


# Runs ON MODAL, not locally. The Chroma credentials live in the
# `podcast-secrets` Modal secret and are not on the developer's machine, and
# `mcp_server.py` exposes no importable collection helper -- it builds a
# CloudClient inside its Modal class's load(). The importable one is
# transcribe.py:137, which takes three arguments.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb==1.5.9")
    .add_local_python_source("corpus")
)

app = modal.App("speaker-backfill")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=1800,
)
def run_backfill(assignments: list[dict], apply: bool) -> dict:
    from transcribe import get_chroma_collection

    collection = get_chroma_collection(
        os.environ["CHROMA_API_KEY"],
        os.environ["CHROMA_TENANT"],
        os.environ["CHROMA_DATABASE"],
    )
    print(f"collection holds {collection.count()} records")

    updates = plan_updates(collection, assignments)
    print(f"planned {len(updates)} chunk updates")
    if not apply:
        print("DRY RUN — nothing written.")
        print("BACKFILL PLAN COMPLETE")
        return {"planned": len(updates), "written": 0}

    written = apply_updates(collection, updates, expected=len(updates))
    print(f"wrote {written}")
    print("BACKFILL COMPLETE")
    return {"planned": len(updates), "written": written}


@app.local_entrypoint()
def main(apply: bool = False) -> None:
    payload = json.loads(Path("speakers/assignments.json").read_text(encoding="utf-8"))
    print(run_backfill.remote(payload["assignments"], apply))
