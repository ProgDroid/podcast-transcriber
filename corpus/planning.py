"""What to do with one episode.

Pure: takes booleans and a version string, returns an action. No Modal, no
Chroma, no filesystem -- so every branch is tested in milliseconds.
"""

from __future__ import annotations

from enum import StrEnum

from corpus.identity import RULES_VERSION


class Action(StrEnum):
    TRANSCRIBE = "TRANSCRIBE"
    EMBED_ONLY = "EMBED_ONLY"
    SKIP = "SKIP"
    EXCLUDE = "EXCLUDE"
    UNPARSEABLE = "UNPARSEABLE"


def decide_action(
    *,
    transcript_exists: bool,
    complete_in_chroma: bool,
    stored_rules_version: str | None,
    excluded: bool,
    parses_to_chunks: bool,
) -> Action:
    """Decide what this episode needs.

    Order matters. EXCLUDE is checked first because both excluded episodes
    have transcripts on the volume and are incomplete in Chroma, so any later
    branch would re-embed them -- reverting an approved deletion every night.

    transcript_exists is checked before parses_to_chunks: UNPARSEABLE means
    "a transcript exists on disk that will never produce chunks" and is
    terminal precisely because such an episode can never become complete, so
    treating it as incomplete would re-embed it forever. That reasoning
    presupposes a transcript exists to have failed parsing. If no transcript
    exists at all, the correct action is TRANSCRIBE, not mark it terminal.
    Measured: 0 of 438 transcripts currently parse to zero chunks, but a total
    state machine is not the same as one no current input breaks.
    """
    if excluded:
        return Action.EXCLUDE
    if not transcript_exists:
        return Action.TRANSCRIBE
    if not parses_to_chunks:
        return Action.UNPARSEABLE
    if not complete_in_chroma:
        return Action.EMBED_ONLY
    if stored_rules_version != RULES_VERSION:
        return Action.EMBED_ONLY
    return Action.SKIP
