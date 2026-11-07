"""One-time: stamp `identity_version` onto the episodes the archive pass ran.

WHY THIS EXISTS. The signal is worthless while `identity_version` is absent
everywhere, and treating absent as "needs identity" would queue all 433
archive episodes on night one to re-download 56 GB of audio and recompute
names that are already correct. This pass writes the key onto the episodes
identification demonstrably already ran over, so the planner's first real
night reports only the episodes published SINCE that pass.

THE STAMP SET IS PROVENANCE, NOT OUTCOME. It is the 433 distinct triples in
`speakers/assignments.json` -- the archive pass's own record of what it
processed. Measured 2026-09-05 and re-measured during review: 970 cluster
records over 433 episodes, 704 named and 266 not, every one carrying the full
(show, episode_number, date) triple, 0 records missing any part of it.

"Has a `speaker_id`" would have selected the same 433 here -- 0 of the 433
have zero named clusters, so the two selectors coincide exactly -- but it
reads the wrong field, and it would silently exclude any episode the pass
processed and matched nobody in. Those are precisely the episodes that must
reach SKIP rather than loop forever, so selecting them by outcome would defeat
the key's whole purpose.

WHY IT NEEDS ITS OWN PLANNER. `backfill_speaker_id.plan_updates` filters by
cluster and skips `person: None`, so reusing it would leave the 266 unmatched
clusters' chunks unstamped. `episode_state` reads ONE arbitrary record of an
episode, so a partly-stamped episode reports whichever record the paging
happened to return -- uniformity is a requirement, not a nicety.

The `--expect` gate is IMPORTED from `backfill_speaker_id` rather than
restated, so the dry-run -> `--expect` flow is the same one and not a
lookalike that drifts.

THE SIX-EPISODE GAP. 439 transcripts exist on the volume and 433 were
identified. The gap is The Jacob Shapiro Podcast episodes 1-6, dated
2022-05-02 to 2022-05-27, and `speakers/unavailable.json` holds `unresolved`
clips spanning exactly those six with `failed: []`. They are ABSENT from the
feed, so `parse_all_episodes` never emits them and they can never be
classified IDENTIFY_ONLY. Their contribution to the first-night count is 0,
and they are deliberately not stamped: nothing ever asks about them.

THE GAP GROWS, AND IT HAS. The feed keeps a fixed window, so the set of
volume episodes it no longer emits gets bigger on its own. **Measured
2026-09-07 against the live feed: episode 7 has also rolled off — the oldest
entry emitted is now episode 8 (2022-06-03).**

This does NOT change the stamp set or `--expect`. Selection is by triple
against the STORE, not the feed, so episode 7 is still stamped and N is still
433. What it changes is the FIRST-NIGHT prediction: an episode that is
stamped but unreachable from the feed can never be counted, so the expected
first-night IDENTIFY_ONLY is 433 minus however many have rolled off — **432
as of 2026-09-07, not 433.** Re-derive it from the feed before the run rather
than reusing this number; that is what it is for.

    modal run stamp_identity_version.py                      # dry run
    modal run stamp_identity_version.py --apply --expect N   # writes
"""

from __future__ import annotations

import json
from pathlib import Path

import modal
from backfill_speaker_id import _require_expect, apply_updates

from corpus.identity import IDENTITY_VERSION
from corpus.remap import is_non_episode
from corpus.store import BATCH, episode_where

TRIPLE_KEYS = ("show", "episode_number", "date")


def episode_triples(assignments: list[dict]) -> list[tuple[str, str, str]]:
    """The distinct episodes an assignments file records a run over.

    `assignments.json` holds one record per CLUSTER -- 970 of them over 433
    episodes -- so this collapses to episodes.

    SORTED, not in file order. The dry run prints a count the operator
    carries by hand into `--apply --expect N`, so the plan has to be
    reproducible against a re-exported assignments.json whose records were
    written in a different order. File order is stable only for one
    particular file; sorting is stable for the set.

    A record missing part of the triple is REFUSED, not skipped. Skipping it
    would build a filter that selects nothing, plan zero updates for that
    episode, and report success -- the failure mode with no symptom.
    """
    seen: set[tuple[str, str, str]] = set()
    for record in assignments:
        missing = [key for key in TRIPLE_KEYS if key not in record]
        if missing:
            raise ValueError(
                f"assignment is missing {missing}: {record.get('episode')!r}"
            )
        seen.add(tuple(record[key] for key in TRIPLE_KEYS))
    return sorted(seen)


def _episode_records(collection, triple: tuple[str, str, str]) -> list[str]:
    """Every id of this episode's OWN records, paged.

    PAGED because an unlimited `get()` silently returns 300: an unpaged plan
    against a 431-chunk episode would stamp 300 records, leave 131 unstamped,
    report success, and leave the episode in the one state `episode_state`'s
    head-record read cannot survive -- non-uniform.

    Non-episode records are excluded. `upload_book.py`'s 191 records can share
    a triple by construction, and stamping one would claim a book chunk was
    identified as a podcast episode.
    """
    show, episode_number, date_str = triple
    where = episode_where(show, episode_number, date_str)
    ids: list[str] = []
    offset = 0
    while True:
        got = collection.get(
            where=where, limit=BATCH, offset=offset, include=["metadatas"]
        )
        page = got["ids"]
        if not page:
            break
        ids.extend(
            chunk_id
            for chunk_id, meta in zip(page, got["metadatas"], strict=True)
            if not is_non_episode(meta)
        )
        if len(page) < BATCH:
            break
        offset += BATCH
    return ids


def plan_stamp_updates(
    collection,
    triples: list[tuple[str, str, str]],
    *,
    identity_version: str = IDENTITY_VERSION,
) -> list[tuple[str, dict]]:
    """Every (chunk_id, {"identity_version": ...}) this pass would write.

    EVERY chunk of a stamped episode, including the chunks of clusters that
    matched nobody. The key records that identification RAN, never what it
    found, and uniformity across the episode is what makes the planner's
    single-record read sound.

    Exactly one key is written. Chroma merges metadata, so a key written here
    cannot later be removed by rewriting the episode -- only by deleting the
    record -- which makes "add one key, overwrite nothing" a correctness
    requirement rather than a style preference.
    """
    return [
        (chunk_id, {"identity_version": identity_version})
        for triple in triples
        for chunk_id in _episode_records(collection, triple)
    ]


def verify_stamped(
    collection,
    triples: list[tuple[str, str, str]],
    *,
    identity_version: str = IDENTITY_VERSION,
) -> tuple[int, int]:
    """Records carrying the current version, and records that should, read back.

    READ BACK from the collection rather than trusting the writer's return
    value. A metadata-only `update()` invites exactly one failure -- reporting
    success and writing nothing -- and a writer cannot detect it about itself.
    This is the pattern `tests/test_backfill.py` already establishes.
    """
    stamped = 0
    total = 0
    for triple in triples:
        show, episode_number, date_str = triple
        where = episode_where(show, episode_number, date_str)
        offset = 0
        while True:
            got = collection.get(
                where=where, limit=BATCH, offset=offset, include=["metadatas"]
            )
            page = got["ids"]
            if not page:
                break
            for meta in got["metadatas"]:
                if is_non_episode(meta):
                    continue
                total += 1
                if meta.get("identity_version") == identity_version:
                    stamped += 1
            if len(page) < BATCH:
                break
            offset += BATCH
    return stamped, total


# Runs ON MODAL, not locally: the Chroma credentials live in the
# `podcast-secrets` Modal secret and are not on the developer's machine.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb==1.5.9")
    # transcribe.py is mounted because get_chroma_collection lives there, and
    # it is the one helper that applies resolve_collection_name -- writing to
    # a phantom collection is the failure this import exists to prevent.
    .add_local_python_source("corpus", "transcribe", "backfill_speaker_id")
)

app = modal.App("identity-version-stamp")


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("podcast-secrets"),
        modal.Secret.from_name("chroma-selfhosted"),
    ],
    timeout=1800,
)
def run_stamp(assignments: list[dict], apply: bool, expect: int = -1) -> dict:
    from transcribe import get_chroma_collection

    collection = get_chroma_collection()
    print(f"collection holds {collection.count()} records")

    triples = episode_triples(assignments)
    print(f"{len(triples)} distinct episodes in the stamp set")
    shows = sorted({t[0] for t in triples})
    for show in shows:
        print(f"  {show}: {sum(1 for t in triples if t[0] == show)}")

    updates = plan_stamp_updates(collection, triples, identity_version=IDENTITY_VERSION)
    print(
        f"planned {len(updates)} chunk updates at identity_version={IDENTITY_VERSION}"
    )

    if not apply:
        print("DRY RUN -- nothing written.")
        print(f"Re-run with: --apply --expect {len(updates)}")
        print("STAMP PLAN COMPLETE")
        return {"episodes": len(triples), "planned": len(updates), "written": 0}

    written = apply_updates(collection, updates, expected=_require_expect(expect))
    stamped, total = verify_stamped(
        collection, triples, identity_version=IDENTITY_VERSION
    )
    print(f"wrote {written}; read back {stamped}/{total} records carrying the version")
    if stamped != total:
        raise RuntimeError(
            f"VERIFICATION FAILED: {total - stamped} record(s) of the stamp set "
            f"do not carry identity_version={IDENTITY_VERSION}. The episodes "
            f"are non-uniform, which is the one state episode_state's "
            f"head-record read cannot survive -- re-run before deploying the "
            f"planner."
        )
    print("STAMP COMPLETE")
    return {
        "episodes": len(triples),
        "planned": len(updates),
        "written": written,
        "verified": stamped,
    }


@app.local_entrypoint()
def main(apply: bool = False, expect: int = -1) -> None:
    payload = json.loads(Path("speakers/assignments.json").read_text(encoding="utf-8"))
    print(run_stamp.remote(payload["assignments"], apply, expect))
