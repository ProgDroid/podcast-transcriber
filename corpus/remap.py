# corpus/remap.py
"""Mapping old-scheme record ids onto the new scheme.

Old: {show}-ep{episode_number}-{index}
New: {show}-ep{episode_number}-{date}-{index}

Only records whose id matches the old episode pattern AND whose metadata
reconstructs that exact prefix are remapped. Everything else passes through
untouched and is counted -- the book records written by upload_book.py have
their own scheme and no episode concept, and an id that disagrees with its own
metadata is a fact to report, never one to guess at.

A record is recognised as non-episode by its `source` metadata, checked
before the id-shape gate -- not by its id or title. upload_book.py stamps
`source="book"` but also reuses the episode fields ("so filters work
consistently": show=title, episode_number="N/A", date="2021-01-01"), and its
ids are `{title}-p{page}-{i}`, which DOES match the old episode id shape. A
book record that reached the id-shape/metadata gate would land in
`passthrough_unmatched` -- the alarm class for an id that disagrees with its
own metadata -- even though nothing is actually wrong with it. Episode
records never write a `source` field at all (see corpus/chunking.py), so
"source present and not 'podcast'" is unambiguous today and stays correct if
a future uploader adds its own `source`.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from corpus.identity import chunk_id, episode_id_prefix

# The trailing component of the OLD scheme is integer-only, which is exactly
# why an old id can never equal a new one.
_OLD_ID_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d+)$")


class RemapResult(NamedTuple):
    new_id: str
    classification: str


def remap_id(old_id: str, metadata: dict) -> RemapResult:
    """Compute this record's id under the new scheme."""
    source = metadata.get("source")
    if source is not None and source != "podcast":
        # Checked before the id-shape gate: a non-episode record (e.g. the
        # book) can carry any id shape, including one that happens to match
        # the old episode pattern.
        return RemapResult(old_id, "passthrough_non_episode")

    m = _OLD_ID_RE.match(old_id)
    if not m:
        return RemapResult(old_id, "passthrough_non_episode")

    show = metadata.get("show")
    episode_number = metadata.get("episode_number")
    date_str = metadata.get("date")
    if not (show and episode_number and date_str):
        return RemapResult(old_id, "passthrough_unmatched")

    old_prefix = f"{show}-ep{episode_number}".replace(" ", "_")
    new_prefix = episode_id_prefix(show, episode_number, date_str)
    index = int(m.group("index"))

    if m.group("prefix") == new_prefix:
        return RemapResult(old_id, "remapped")  # already migrated
    if m.group("prefix") != old_prefix:
        return RemapResult(old_id, "passthrough_unmatched")

    return RemapResult(chunk_id(new_prefix, index), "remapped")
