"""Pure planning helpers for the `podcast_transcripts` -> `_v2` re-ID.

Split out of `migration/reid.py` so these can be tested without importing
`modal`: `migration/reid.py` builds a `modal.Image` / `modal.App` at module
scope, which needs the `modal` package installed, and this package must not
depend on it.
"""

from __future__ import annotations

from corpus.chunking import count_chunks_from_text
from corpus.identity import RULES_VERSION


def build_episode_facts(
    transcript_texts: dict[tuple[str, str, str], str],
    feed_guids: dict[tuple[str, str, str], str],
) -> dict[tuple[str, str, str], dict]:
    """Per-episode facts to stamp onto every one of its records.

    `n_chunks` is derived by RE-PARSING THE TRANSCRIPT, never by counting
    stored records: counting would stamp a torn episode's truncated count as
    its expected count and freeze it as permanently complete.
    """
    facts: dict[tuple[str, str, str], dict] = {}
    for key, text in transcript_texts.items():
        show, episode_number, date_str = key
        entry: dict = {
            "n_chunks": count_chunks_from_text(
                text, show, episode_number, "", date_str
            ),
            "rules_version": RULES_VERSION,
        }
        guid = feed_guids.get(key)
        if guid is not None:
            entry["episode_guid"] = guid
        facts[key] = entry
    return facts


def enrich_metadata(metadata: dict, facts: dict | None) -> dict:
    """Add the declared new keys. Records with no facts pass through."""
    if not facts:
        return metadata
    return {**metadata, **facts}
