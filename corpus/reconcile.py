"""Reconcile the transcript volume against the vector store, both directions.

A one-directional check keeps passing while the corpus rots: the original
version could only see records that were MISSING, never records that should
not exist and never an episode whose indices had holes in them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

from corpus.exclusions import EXCLUDED_EPISODES, is_excluded
from corpus.remap import is_non_episode

Key = tuple[str, str, str]


@dataclass
class ReconcileReport:
    missing: list[Key] = field(default_factory=list)
    extra: list[Key] = field(default_factory=list)
    non_contiguous: list[Key] = field(default_factory=list)
    shared_prefixes: list[str] = field(default_factory=list)
    excluded_with_records: list[Key] = field(default_factory=list)
    feed_unreachable: list[Key] = field(default_factory=list)
    # NEVER POPULATED TODAY. `reconcile()` only ever sees (show,
    # episode_number, date) triples -- no episode titles reach it from the
    # volume, Chroma or the feeds -- and cross-post detection needs a fuzzy
    # title-and-date match (see corpus/exclusions.py's module docstring).
    # An empty list here means "this check does not run", NOT "no cross-posts
    # were found". Do not read it as a clean result; it is not a result at
    # all until something populates it.
    suspected_cross_posts: list[tuple[Key, Key]] = field(default_factory=list)

    def is_clean(self) -> bool:
        """feed_unreachable and suspected_cross_posts are INFORMATIONAL.

        Those two are facts to know, not faults: six episodes have aged off
        their feed and cross-posts are a human decision, not an inference.
        """
        return not (
            self.missing
            or self.extra
            or self.non_contiguous
            or self.shared_prefixes
            or self.excluded_with_records
        )


def reconcile(
    volume_keys: set[Key],
    chroma_records: list[tuple[str, dict]],
    feed_keys: set[Key],
) -> ReconcileReport:
    """Compare the volume, the store and the feeds."""
    report = ReconcileReport()

    by_key: dict[Key, list[str]] = defaultdict(list)
    prefixes: dict[str, set[Key]] = defaultdict(set)
    excluded_present: set[Key] = set()

    for record_id, meta in chroma_records:
        # upload_book.py's records reuse episode-shaped metadata fields
        # (show/episode_number/date) plus source="book" -- and its ids match
        # the same digit-suffix shape episode chunk ids do. Left in, they'd
        # report as `extra` (and pollute non_contiguous/shared_prefixes) on
        # every single run, training the operator to stop reading the report.
        if is_non_episode(meta):
            continue

        # meta is a duck-typed Chroma metadata dict -- runtime-untyped, but
        # every episode chunk record carries these three fields as strings.
        show = cast(str, meta.get("show"))
        episode_number = cast(str, meta.get("episode_number"))
        date_str = cast(str, meta.get("date"))
        key: Key = (show, episode_number, date_str)
        by_key[key].append(record_id)

        # An excluded episode that came back under a BACKFILLED
        # episode_number no longer matches the triple; is_excluded's guid
        # arm is what still catches it. Without this it lands in `extra`
        # and the alarm points at the wrong class entirely.
        if is_excluded(show, episode_number, date_str, meta.get("episode_guid")):
            excluded_present.add(key)

        prefix, _, index = record_id.rpartition("-")
        if index.isdigit():
            prefixes[prefix].add(key)

    expected = volume_keys - EXCLUDED_EPISODES
    report.missing = sorted(k for k in expected if k not in by_key)
    report.extra = sorted(
        k for k in by_key if k not in volume_keys and k not in excluded_present
    )
    report.excluded_with_records = sorted(excluded_present)
    report.feed_unreachable = sorted(volume_keys - feed_keys)
    report.shared_prefixes = sorted(p for p, keys in prefixes.items() if len(keys) > 1)

    for key, ids in by_key.items():
        indices = sorted(
            int(i.rpartition("-")[2]) for i in ids if i.rpartition("-")[2].isdigit()
        )
        if indices and indices != list(range(len(indices))):
            report.non_contiguous.append(key)
    report.non_contiguous.sort()

    return report
