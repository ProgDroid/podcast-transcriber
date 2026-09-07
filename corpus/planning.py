"""What to do with one episode.

Pure: takes booleans and version strings, returns an action. No Modal, no
Chroma, no filesystem -- so every branch is tested in milliseconds.
"""

from __future__ import annotations

from enum import StrEnum

from corpus.identity import IDENTITY_VERSION, RULES_VERSION

# How many episodes one run may repair identity for, per show.
#
# Deliberately a constant and not an env var: a bandwidth bound changeable in
# the Modal dashboard with no code diff is the shape `resolve_collection_name`
# exists to refuse. Ships at 0 so the planner classifies and reports against
# production for one night with no bandwidth spent; raised by a one-line
# follow-up commit once the first night's counts match Part 5's
# pre-registration.
IDENTITY_REPAIR_CAP = 0


class Action(StrEnum):
    TRANSCRIBE = "TRANSCRIBE"
    EMBED_ONLY = "EMBED_ONLY"
    IDENTIFY_ONLY = "IDENTIFY_ONLY"
    SKIP = "SKIP"
    EXCLUDE = "EXCLUDE"
    UNPARSEABLE = "UNPARSEABLE"


def decide_action(
    *,
    transcript_exists: bool,
    complete_in_chroma: bool,
    stored_rules_version: str | None,
    stored_identity_version: str | None,
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

    IDENTIFY_ONLY is placed LAST, after both chunk checks. Identity is derived
    from chunks, so chunks settle first: repairing identity over an episode
    that is torn or rules-stale stamps a chunk set that is about to be
    rewritten, and the stamp would be positional rather than earned.

    `stored_identity_version` is taken as a value rather than a pre-computed
    boolean, for symmetry with `stored_rules_version` -- the comparison
    against the current version is this function's to make, and a caller that
    made it would be a second place the version constant is read.
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
    if stored_identity_version != IDENTITY_VERSION:
        return Action.IDENTIFY_ONLY
    return Action.SKIP


def cap_work(episodes: list[dict], cap: int) -> tuple[list[dict], int]:
    """The episodes to act on now, and how many are left over.

    Ordering is the feed's, preserved: re-ordering would let an episode at the
    tail starve while the head is repaired repeatedly.

    A new list is returned rather than a slice alias, so a caller mutating the
    work list cannot reach back into the plan it came from.
    """
    if cap <= 0:
        return [], len(episodes)
    return list(episodes[:cap]), max(0, len(episodes) - cap)
