"""THROWAWAY PROBE -- did the nightly run stamp speaker_id on new episodes?

The ingest wiring (1f9bb83) deployed 2026-09-04. Nothing has yet confirmed it
works on real data; the mount probe proved the container CAN derive centroids,
not that an episode came out the other end with an identity on it.

PRE-REGISTERED, before running:

  * The most likely outcome is ZERO new episodes. The feeds publish ~2.3 a
    week and the last known are 2026-08-31 (JSP) and 2026-08-22 (GC). Zero is
    a NO-OP and says NOTHING about the wiring -- it must be reported UNKNOWN,
    never as a pass. Only an episode dated after the cutoff, carrying
    speaker_id, is evidence the wiring fires.
  * cutoff: episodes with date_ts > 20260904 are ingest-era. Everything on or
    before that got speaker_id from the backfill instead, so mixing the two
    would let old backfilled rows masquerade as new stamped ones.
  * collection count >= 29160. It can only grow.
  * control: a nonsense speaker_id value must return 0, proving the filter
    discriminates rather than matching everything.

Reports per new episode: chunk count, how many carry speaker_id, and which
names. A new episode with zero stamped chunks is a REAL FAILURE and is the
outcome this probe exists to catch -- it is also invisible to any query,
because Chroma's $eq skips records missing the key.

Run:  modal run probe_nightly_identity.py
"""

from __future__ import annotations

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb==1.5.9")
    .add_local_python_source("corpus", "transcribe")
)

app = modal.App("probe-nightly-identity")

# Ingest-era boundary. The backfill ran 2026-09-04; the wiring deployed the
# same day. Anything strictly after belongs to the new path.
CUTOFF_TS = 20260904
BACKFILL_RECORDS = 29160


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=1800,
)
def check() -> dict:
    from collections import defaultdict

    from corpus.store import paged_get
    from transcribe import get_chroma_collection

    collection = get_chroma_collection(
        os.environ["CHROMA_API_KEY"],
        os.environ["CHROMA_TENANT"],
        os.environ["CHROMA_DATABASE"],
    )
    total = collection.count()
    print(f"collection holds {total} records (was {BACKFILL_RECORDS} at backfill)")

    fresh = paged_get(
        collection,
        {"date_ts": {"$gt": CUTOFF_TS}},
        ["metadatas"],
    )
    metas = fresh.get("metadatas", [])
    print(f"chunks dated after {CUTOFF_TS}: {len(metas)}")

    # Positive control: the same machinery on a value nobody ever wrote.
    control = paged_get(
        collection, {"speaker_id": {"$eq": "__nobody__"}}, ["metadatas"]
    )
    n_control = len(control.get("metadatas", []))
    print(f"CONTROL nonsense speaker_id: {n_control} (expect 0)")

    episodes: dict[tuple, dict] = defaultdict(
        lambda: {"chunks": 0, "stamped": 0, "names": set(), "clusters": set()}
    )
    for m in metas:
        key = (m.get("show"), m.get("episode_number"), m.get("date"))
        row = episodes[key]
        row["chunks"] += 1
        row["clusters"].add(m.get("speaker"))
        sid = m.get("speaker_id")
        if sid:
            row["stamped"] += 1
            row["names"].add(sid)

    out = []
    for (show, number, date), row in sorted(episodes.items()):
        print(
            f"  {show} ep{number} ({date}): {row['stamped']}/{row['chunks']} "
            f"stamped, names={sorted(row['names']) or 'NONE'}, "
            f"clusters={len(row['clusters'])}"
        )
        out.append(
            {
                "show": show,
                "episode_number": number,
                "date": date,
                "chunks": row["chunks"],
                "stamped": row["stamped"],
                "names": sorted(row["names"]),
                "clusters": len(row["clusters"]),
            }
        )

    print("CHECK COMPLETE")  # terminal marker
    return {"total": total, "control": n_control, "episodes": out}


@app.local_entrypoint()
def main() -> None:
    r = check.remote()
    episodes = r["episodes"]

    print()
    print(f"collection: {r['total']} records")
    print(f"control:    {r['control']} (must be 0)")

    if r["control"] != 0:
        print("VERDICT: BROKEN PROBE -- the control matched. Believe nothing above.")
        return

    if not episodes:
        # The honest answer, and the likely one.
        print(
            "VERDICT: UNKNOWN -- no episodes dated after the cutoff, so the "
            "nightly run had nothing to ingest. This is NOT evidence the "
            "wiring works; it is evidence it was never exercised."
        )
        return

    unstamped = [e for e in episodes if e["stamped"] == 0]
    partial = [e for e in episodes if 0 < e["stamped"] < e["chunks"]]
    print(f"new episodes: {len(episodes)}, of which {len(unstamped)} wholly unstamped")
    for e in partial:
        print(
            f"  partial (expected -- unmatched clusters get nothing): "
            f"{e['show']} ep{e['episode_number']} {e['stamped']}/{e['chunks']}"
        )
    if unstamped:
        print("VERDICT: FAILED -- new episodes carry no speaker_id at all:")
        for e in unstamped:
            print(f"  {e['show']} ep{e['episode_number']} ({e['date']})")
    else:
        print("VERDICT: PASS -- every new episode carries at least one speaker_id.")
