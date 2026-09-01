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
from corpus.remap import is_non_episode
from corpus.store import (
    BATCH,
    batched,
    episode_where,
    guid_where,
    paged_get,
    stale_ids,
)


def _matched_ids_and_non_episode_ids(
    collection, where: dict
) -> tuple[set[str], set[str]]:
    """ids matching `where`, split out the ones that aren't episode chunks.

    Fetches metadata in the same paged pass rather than a second round trip.
    A non-episode id (e.g. a book record sharing this triple/guid) is never
    eligible for pruning -- see `upsert_then_prune`'s guard below.
    """
    result = paged_get(collection, where, include=["metadatas"])
    ids = set(result["ids"])
    non_episode = {
        _id
        for _id, meta in zip(result["ids"], result["metadatas"], strict=True)
        if is_non_episode(meta)
    }
    return ids, non_episode


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
    existing, non_episode = _matched_ids_and_non_episode_ids(
        collection, episode_where(show, episode_number, date_str)
    )
    if episode_guid:
        # $and with show: a guid is only unique WITHIN a feed. Two different
        # shows can carry the same guid on a cross-posted episode (see
        # corpus/exclusions.py), and an unscoped guid arm would prune the
        # other show's records too. Cost of that narrowing: a genuine feed
        # SHOW RENAME now strands the old records under the old show instead
        # of the guid arm cleaning them up -- accepted, because a stranded
        # (recoverable) orphan is a smaller failure than a cross-show
        # over-delete (not recoverable from the guid arm alone).
        guid_ids, guid_non_episode = _matched_ids_and_non_episode_ids(
            collection,
            {"$and": [guid_where(episode_guid), {"show": {"$eq": show}}]},
        )
        existing |= guid_ids
        non_episode |= guid_non_episode

    # A non-episode record (e.g. upload_book.py's book chunks) can share this
    # exact triple or guid by construction -- see corpus/remap.py's docstring
    # -- and must never be pruned just because it isn't in `new_ids`. Cost of
    # that guard: any record stamping a `source` other than "podcast" becomes
    # permanently un-prunable by this function, forever -- accepted, because
    # the alternative is the 191-record book destruction this guard exists
    # to prevent, and an un-prunable stray record is recoverable by hand
    # while a wrongly deleted corpus is not.
    to_prune = [i for i in stale_ids(existing, new_ids) if i not in non_episode]
    for batch in batched(to_prune, BATCH):
        collection.delete(ids=batch)
    return {"written": len(new_ids), "pruned": len(to_prune)}
