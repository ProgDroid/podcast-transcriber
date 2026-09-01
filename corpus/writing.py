"""Full-replacement episode writes.

UPSERT FIRST, THEN PRUNE -- not delete-then-upsert. A new episode's delete
matches nothing so the trickle case has no window either way, but a
full-archive re-embed opens one destructive window per healthy episode. And
the failure that matters is not an exception: transcribe runs with a 7200s
timeout covering a whole show, and a Modal timeout kills the container without
raising anything an `except` clause can see. Under delete-first that is a
deleted-and-never-rewritten episode with no log line. This way a crash leaves
a SUPERSET, never a hole, and the next run prunes it.
"""

from __future__ import annotations

from corpus.identity import chunk_id, episode_id_prefix
from corpus.store import (
    BATCH,
    batched,
    episode_where,
    guid_where,
    paged_get_ids,
    stale_ids,
)


def upsert_then_prune(
    collection,
    chunks: list[dict],
    embeddings: list[list[float]],
    *,
    show: str,
    episode_number: str,
    date_str: str,
    episode_guid: str | None,
    _skip_prune: bool = False,
) -> dict:
    """Write an episode's chunks, then remove whatever it no longer occupies.

    `_skip_prune` exists only so a test can simulate a crash between the two
    phases. Never pass it from production code.
    """
    prefix = episode_id_prefix(show, episode_number, date_str)
    new_ids = [chunk_id(prefix, i) for i in range(len(chunks))]

    for batch in batched(list(range(len(new_ids))), BATCH):
        collection.upsert(
            ids=[new_ids[i] for i in batch],
            embeddings=[embeddings[i] for i in batch],
            documents=[chunks[i]["text"] for i in batch],
            metadatas=[chunks[i]["metadata"] for i in batch],
        )

    if _skip_prune:
        return {"written": len(new_ids), "pruned": 0}

    # UNION, never "guid if present else triple". Every record written before
    # the migration is triple-keyed with no guid, so a guid-only prune matches
    # nothing and strands the entire old record set.
    existing = set(
        paged_get_ids(collection, episode_where(show, episode_number, date_str))
    )
    if episode_guid:
        # $and with show: a guid is only unique WITHIN a feed. Two different
        # shows can carry the same guid on a cross-posted episode (see
        # corpus/exclusions.py), and an unscoped guid arm would prune the
        # other show's records too.
        existing |= set(
            paged_get_ids(
                collection,
                {"$and": [guid_where(episode_guid), {"show": {"$eq": show}}]},
            )
        )

    to_prune = stale_ids(existing, new_ids)
    for batch in batched(to_prune, BATCH):
        collection.delete(ids=batch)
    return {"written": len(new_ids), "pruned": len(to_prune)}
