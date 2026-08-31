"""Chroma-shaped helpers, written against a duck-typed collection.

Nothing here imports chromadb. Callers pass anything with `get`, `upsert`,
`delete` and `count`, which is what makes the whole layer testable against a
fake that enforces the real caps.

The caps below were measured against Chroma Cloud (chromadb 1.5.9), not read
from documentation. The dangerous one is that an unlimited `get()` returns 300
records and raises nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

# Chroma Cloud rejects get(limit>300) and upsert of >300 records per request.
MAX_REQUEST = 300
# Page and batch below the cap, per the value already used in migration/.
PAGE = 250
BATCH = 250


def batched(items: list, size: int = BATCH) -> Iterator[list]:
    """Split a list into request-sized batches."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def episode_where(show: str, episode_number: str, date_str: str) -> dict:
    """A filter selecting one episode by the unique triple."""
    return {
        "$and": [
            {"show": {"$eq": show}},
            {"episode_number": {"$eq": episode_number}},
            {"date": {"$eq": date_str}},
        ]
    }


def guid_where(episode_guid: str) -> dict:
    """A filter selecting one episode by its RSS guid."""
    return {"episode_guid": {"$eq": episode_guid}}


def paged_get(collection, where: dict, include: list[str]) -> dict:
    """Page a filtered get, assembling the complete result."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    offset = 0
    while True:
        page = collection.get(where=where, include=include, limit=PAGE, offset=offset)
        if not page["ids"]:
            break
        ids.extend(page["ids"])
        if "documents" in include:
            documents.extend(page.get("documents", []))
        if "metadatas" in include:
            metadatas.extend(page.get("metadatas", []))
        offset += len(page["ids"])
    out: dict = {"ids": ids}
    if "documents" in include:
        out["documents"] = documents
    if "metadatas" in include:
        out["metadatas"] = metadatas
    return out


def paged_get_ids(collection, where: dict) -> list[str]:
    """Every id matching the filter. Never call the unpaged form."""
    return paged_get(collection, where, include=[])["ids"]


def stale_ids(existing_ids: Iterable[str], new_ids: Iterable[str]) -> list[str]:
    """Records to prune after an upsert: what was there and no longer is.

    Upsert cannot shrink a record set, so a re-embed producing fewer chunks
    strands every index above the new count. Those survivors keep the old
    document text, which after a speaker rename means stale labels and
    duplicated passages.
    """
    return sorted(set(existing_ids) - set(new_ids))


def is_complete(stored_ids: list[str], expected_n_chunks: int | None) -> bool:
    """Whether an episode is fully stored.

    Presence is not existence. A collision clobber leaves an episode with one
    surviving chunk, and a boolean "does any chunk exist" check calls that
    healthy -- so the self-healing branch never repairs it and reconciliation
    passes.

    A missing expected count means a pre-migration record, which is treated as
    INCOMPLETE. The alternative reading -- absent means satisfied -- would mean
    old episodes are never completeness-checked at all.
    """
    if expected_n_chunks is None:
        return False
    return len(stored_ids) == expected_n_chunks
