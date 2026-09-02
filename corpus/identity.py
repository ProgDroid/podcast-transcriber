"""Episode identity.

`episode_number` is a display attribute, not a key: the feeds omit it (5
episodes carry the literal string "Unknown") and occasionally repeat it
(`Episode 243` exists on both 2024-11-07 and 2024-11-08). Neither is
`date` a key on its own -- four (show, date) pairs are duplicated. Only the
full triple is unique, at 439 distinct keys across 439 transcripts
(re-verified 2026-09-02 against the volume).

That a publisher's own numbering can be self-contradictory is not
hypothetical: see the comment at `transcribe.py`'s episode-number fallback
for a feed whose `<itunes:episode>` disagrees with its own titles by one.
Keying on the triple is what makes that harmless.
"""

from __future__ import annotations

import re

# Bumped whenever anything that derives chunk CONTENT from a transcript
# changes -- today the speaker labels. Completeness answers "does this episode
# have all its chunks"; this answers "were those chunks built by current
# rules". They are different questions, and conflating them makes a
# rules-change re-embed a silent no-op.
RULES_VERSION = "1"

_FILENAME_RE = re.compile(
    r"^(?P<show>.+) - Episode (?P<ep>\w+) - "
    r"(?P<date>\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\.txt$"
)


def episode_id_prefix(show: str, episode_number: str, date_str: str) -> str:
    """Stable, unique per-episode ID prefix, keyed on the full triple."""
    return f"{show}-ep{episode_number}-{date_str}".replace(" ", "_")


def chunk_id(prefix: str, index: int) -> str:
    """The ID of one chunk within an episode."""
    return f"{prefix}-{index}"


def transcript_filename(show: str, episode_number: str, date_str: str) -> str:
    """The volume filename the pipeline writes for an episode."""
    return f"{show} - Episode {episode_number} - {date_str}.txt"


def parse_transcript_filename(filename: str) -> tuple[str, str, str] | None:
    """Inverse of `transcript_filename`. None if the name is not a transcript."""
    m = _FILENAME_RE.match(filename)
    if not m:
        return None
    return m.group("show"), m.group("ep"), m.group("date")
