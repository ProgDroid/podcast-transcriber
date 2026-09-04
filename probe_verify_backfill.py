"""THROWAWAY PROBE -- did the backfill actually land, read back from Chroma?

`backfill_speaker_id.py` reported `written 22483`. That is the writer's own
count of calls it believes it made. It is not evidence that the collection
holds them, that the values are the right ones, or that `update()` preserved
the documents and embeddings it was supposed to leave alone.

PRE-REGISTERED, before running:

  * sum of per-person counts        == 22483   (matches what was written)
  * nonsense value control          == 0       (proves the filter
                                                discriminates rather than
                                                matching everything)
  * unfiltered collection count     == 29160   (unchanged -- a backfill that
                                                created records instead of
                                                updating them would grow it)
  * JSP Episode 100 SPEAKER_01      == "Marko Papic"
                                                (the corrupt run's smoking
                                                gun, attributed to Jacob)
  * sampled updated chunk           document non-empty and embedding present
                                                (`update()` must preserve
                                                both; an upsert would have
                                                blanked the document)

A per-person sum that lands short means some writes did not stick. A sum that
EXCEEDS 22483 means the filter is matching records it should not, which is
the more alarming direction.

Run:  modal run probe_verify_backfill.py
"""

from __future__ import annotations

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb==1.5.9")
    .add_local_python_source("corpus", "transcribe")
)

app = modal.App("verify-backfill")

PEOPLE = [
    "Jacob Shapiro",
    "Marko Papic",
    "Matt Gertken",
    "Rob Larity",
    "Tobias Harris",
]
EXPECTED_TOTAL = 22483
EXPECTED_RECORDS = 29160


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=1800,
)
def verify() -> dict:
    from corpus.store import paged_get_ids
    from transcribe import get_chroma_collection

    collection = get_chroma_collection(
        os.environ["CHROMA_API_KEY"],
        os.environ["CHROMA_TENANT"],
        os.environ["CHROMA_DATABASE"],
    )

    total_records = collection.count()
    print(f"collection holds {total_records} records (expect {EXPECTED_RECORDS})")

    counts: dict[str, int] = {}
    for person in PEOPLE:
        ids = paged_get_ids(collection, {"speaker_id": {"$eq": person}})
        counts[person] = len(ids)
        print(f"  {person}: {len(ids)}")

    # Positive control. If a value nobody wrote comes back non-zero, the
    # filter is not doing what the counts above assume it is.
    control = len(paged_get_ids(collection, {"speaker_id": {"$eq": "__nobody__"}}))
    print(f"CONTROL nonsense value: {control} (expect 0)")

    total = sum(counts.values())
    print(f"sum of per-person counts: {total} (expect {EXPECTED_TOTAL})")

    # The corrupt run put this cluster under Jacob Shapiro at 0.879.
    gun = collection.get(
        where={
            "$and": [
                {"show": {"$eq": "The Jacob Shapiro Podcast"}},
                {"episode_number": {"$eq": "100"}},  # a STRING in the metadata
                {"speaker": {"$eq": "SPEAKER_01"}},
            ]
        },
        limit=5,
        include=["metadatas", "documents"],
    )
    gun_ids = gun.get("ids", [])
    gun_speakers = [m.get("speaker_id") for m in gun.get("metadatas", [])]
    print(f"\nJSP Episode 100 SPEAKER_01: {len(gun_ids)} chunks sampled")
    print(f"  speaker_id values: {set(gun_speakers) or 'NONE'}")

    # `update()` must have preserved the document and the embedding. An
    # `upsert` without documents would have been rejected client-side, but a
    # blanked document would be silent.
    preserved = None
    if gun_ids:
        doc = gun.get("documents", [""])[0]
        with_emb = collection.get(ids=[gun_ids[0]], include=["embeddings"])
        # Chroma returns embeddings as a numpy array; `or []` on one raises
        # "truth value of an array is ambiguous" rather than falling back.
        emb = with_emb.get("embeddings")
        n_emb = 0 if emb is None else len(emb)
        preserved = {
            "document_chars": len(doc or ""),
            "embedding_dims": len(emb[0]) if n_emb else 0,
        }
        print(f"  preserved: {preserved}")

    print("VERIFY COMPLETE")  # terminal marker
    return {
        "total_records": total_records,
        "counts": counts,
        "control": control,
        "total": total,
        "gun_speakers": sorted({s for s in gun_speakers if s}),
        "preserved": preserved,
    }


@app.local_entrypoint()
def main() -> None:
    result = verify.remote()

    checks = [
        ("record count unchanged", result["total_records"] == EXPECTED_RECORDS),
        ("per-person sum matches", result["total"] == EXPECTED_TOTAL),
        ("nonsense control is zero", result["control"] == 0),
        ("episode 100 is Marko Papic", result["gun_speakers"] == ["Marko Papic"]),
        (
            "document preserved",
            bool(result["preserved"])
            and result["preserved"]["document_chars"] > 0
            and result["preserved"]["embedding_dims"] > 0,
        ),
    ]
    print()
    for label, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
    print("ALL PASS" if all(ok for _, ok in checks) else "SOME CHECKS FAILED")
