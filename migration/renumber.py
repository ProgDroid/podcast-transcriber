"""Re-key a renumbered feed's episodes in place. No GPU, no re-transcription.

See `corpus/renumber.py` for why (the 2026-11 JSP +72 renumber) and how the
match is made (by guid, never by the shift).

Usage (from the repo root):
  modal run migration/renumber.py
      dry run: prints the plan and the counts to pass back, writes nothing
  modal run migration/renumber.py --apply --expect-renames N --expect-orphans M
      applies only if the fresh plan matches both pre-registered counts

Per episode, in this order:
  1. upsert the records under the new ids, embeddings copied, not recomputed
  2. read back the new triple's count; stop on any mismatch
  3. write the renamed transcript and move the old file to /renumbered/
  4. delete the old ids, then read back that none survive

A crash after 1 or 2 leaves duplicates a re-run completes (upsert is
idempotent and the plan still sees the file missing). A crash between 3 and 4
leaves the old records behind; the re-run's plan then reports that guid as
stored under two triples, by name, rather than silently skipping it.

Orphans (old copies of episodes already re-transcribed) are moved to
/renumbered/ too: reversible, and out of `bulk_embed`'s top-level listing.

RUN WITH NOTHING ELSE WRITING: not between 09:00 and 11:00 UTC.
"""

from __future__ import annotations

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb==1.5.9", "feedparser==6.0.12")
    # transcribe: for parse_all_episodes itself, not a copy of it.
    .add_local_python_source("corpus", "transcribe")
)

app = modal.App("podcast-renumber-migration", image=image)
volume = modal.Volume.from_name("podcast-transcripts")
VOLUME_PATH = "/transcripts"
ARCHIVE_DIR = f"{VOLUME_PATH}/renumbered"

FEEDS = {
    "The Jacob Shapiro Podcast": "https://feeds.captivate.fm/jacob-shapiro/",
}


def _fetch_full(collection, where: dict) -> dict:
    """Paged get INCLUDING embeddings, which `paged_get` does not carry."""
    from corpus.store import PAGE

    out: dict = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
    offset = 0
    while True:
        page = collection.get(
            where=where,
            include=["embeddings", "documents", "metadatas"],
            limit=PAGE,
            offset=offset,
        )
        if not page["ids"]:
            break
        out["ids"].extend(page["ids"])
        out["embeddings"].extend([list(e) for e in page["embeddings"]])
        out["documents"].extend(page["documents"])
        out["metadatas"].extend(page["metadatas"])
        offset += len(page["ids"])
    return out


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("podcast-secrets"),
        modal.Secret.from_name("chroma-selfhosted"),
    ],
    timeout=3600,
    memory=2048,
)
def run(show: str, apply: bool, expect_renames: int, expect_orphans: int) -> None:
    from transcribe import parse_all_episodes

    from corpus.chroma_client import make_client
    from corpus.identity import parse_transcript_filename, transcript_filename
    from corpus.renumber import plan_renumber, rekey, rewrite_header
    from corpus.store import (
        BATCH,
        batched,
        episode_where,
        paged_get,
        paged_get_ids,
        resolve_collection_name,
    )

    volume.reload()
    name = resolve_collection_name(os.environ["CHROMA_COLLECTION"])
    collection = make_client().get_collection(name)
    print(f"collection={name} count={collection.count()} apply={apply}")

    feed = parse_all_episodes(FEEDS[show])
    volume_triples = set()
    for entry in os.listdir(VOLUME_PATH):
        parsed = parse_transcript_filename(entry)
        if parsed and parsed[0] == show:
            volume_triples.add(parsed)

    stored = paged_get(collection, {"show": {"$eq": show}}, include=["metadatas"])
    stored_triples = set()
    stored_by_guid: dict[str, set[tuple[str, str]]] = {}
    for meta in stored["metadatas"]:
        pair = (meta["episode_number"], meta["date"])
        stored_triples.add((show, *pair))
        if meta.get("episode_guid"):
            stored_by_guid.setdefault(meta["episode_guid"], set()).add(pair)
    print(
        f"feed {len(feed)} entries; volume {len(volume_triples)} files; "
        f"store {len(stored['ids'])} records / {len(stored_triples)} triples"
    )

    plan = plan_renumber(
        show,
        feed,
        volume_triples=volume_triples,
        stored_triples=stored_triples,
        stored_by_guid=stored_by_guid,
    )
    record_total = sum(
        len(paged_get_ids(collection, episode_where(show, r.old_number, r.date)))
        for r in plan.renames
    )
    print(f"renames {len(plan.renames)} ({record_total} records)")
    for r in plan.renames[:5]:
        print(f"   ep{r.old_number} -> ep{r.new_number}  {r.date}  {r.guid}")
    print(f"new (untranscribed, left for the nightly run) {plan.new}")
    print(f"unresolved {len(plan.unresolved)}")
    for line in plan.unresolved:
        print(f"   {line}")
    print(f"orphans {len(plan.orphans)}")
    for _, number, date in plan.orphans:
        print(f"   ep{number} {date}")
    print("RENUMBER PLAN COMPLETE")

    if not apply:
        print(
            "DRY RUN -- nothing written. To apply: --apply "
            f"--expect-renames {len(plan.renames)} "
            f"--expect-orphans {len(plan.orphans)}"
        )
        return
    if (len(plan.renames), len(plan.orphans)) != (expect_renames, expect_orphans):
        raise SystemExit(
            f"REFUSED: plan has {len(plan.renames)} renames / "
            f"{len(plan.orphans)} orphans, expected {expect_renames} / "
            f"{expect_orphans}. Nothing written."
        )

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    for done, r in enumerate(plan.renames, 1):
        old = _fetch_full(collection, episode_where(show, r.old_number, r.date))
        new_ids, new_metas = rekey(
            old["ids"],
            old["metadatas"],
            show=show,
            old_number=r.old_number,
            new_number=r.new_number,
            date=r.date,
        )
        for batch in batched(list(range(len(new_ids))), BATCH):
            collection.upsert(
                ids=[new_ids[i] for i in batch],
                embeddings=[old["embeddings"][i] for i in batch],
                documents=[old["documents"][i] for i in batch],
                metadatas=[new_metas[i] for i in batch],
            )
        landed = paged_get_ids(collection, episode_where(show, r.new_number, r.date))
        if sorted(landed) != sorted(new_ids):
            raise SystemExit(
                f"STOPPED at ep{r.old_number}->ep{r.new_number}: wrote "
                f"{len(new_ids)}, read back {len(landed)}. Old records intact."
            )

        old_path = f"{VOLUME_PATH}/{transcript_filename(show, r.old_number, r.date)}"
        new_path = f"{VOLUME_PATH}/{transcript_filename(show, r.new_number, r.date)}"
        with open(old_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        with open(new_path, "w", encoding="utf-8") as fh:
            fh.write(rewrite_header(text, show, r.old_number, r.new_number))
        os.replace(old_path, f"{ARCHIVE_DIR}/{os.path.basename(old_path)}")

        for batch in batched(old["ids"], BATCH):
            collection.delete(ids=batch)
        left = paged_get_ids(collection, episode_where(show, r.old_number, r.date))
        if left:
            raise SystemExit(
                f"STOPPED at ep{r.old_number}: {len(left)} old records survived "
                f"the delete. New records and file are in place."
            )
        if done % 25 == 0:
            volume.commit()
            print(f"  renamed {done}/{len(plan.renames)}")

    for _, number, date in plan.orphans:
        path = f"{VOLUME_PATH}/{transcript_filename(show, number, date)}"
        os.replace(path, f"{ARCHIVE_DIR}/{os.path.basename(path)}")
    volume.commit()
    print(
        f"RENUMBER COMPLETE: {len(plan.renames)} renamed, "
        f"{len(plan.orphans)} orphans archived; count={collection.count()}"
    )


@app.local_entrypoint()
def main(
    show: str = "The Jacob Shapiro Podcast",
    apply: bool = False,
    expect_renames: int = -1,
    expect_orphans: int = -1,
):
    run.remote(show, apply, expect_renames, expect_orphans)
