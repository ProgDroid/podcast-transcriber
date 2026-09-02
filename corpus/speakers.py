"""Turns and speech shares, derived from transcripts alone.

No modal, no audio library, no network -- so the derivation that decides
"who dominates this episode" runs in the test suite on CPU, the same split
`corpus/showplan.py` argues for.

**Every duration here is derived, and the derivation is a choice.** A
transcript line carries a start time and nothing else, so a segment's
duration can only be the gap to the next segment's start, and the last
segment of an episode has no successor to subtract from. Two consequences
are load-bearing:

  - Silence, music and advertising are indistinguishable from speech in a
    start-only transcript, and uncapped they are charged in full to whoever
    spoke before them. `gap_cap_s` is therefore REQUIRED rather than
    defaulted: a caller that does not state its cap cannot produce a number
    anyone can compare against another run. The figures this replaces were
    computed by a method nobody wrote down, which is the entire reason this
    module exists.
  - Seconds are derived; words are counted. `speech_shares` reports both and
    never computes one from the other, so a show where they disagree about
    who dominates is visible rather than averaged away.

This module deliberately stops at measurement. The dominant-cluster rule,
the co-host surname check and the eval scoring are gated on a labelled set
that does not exist yet, and none of them belong here until it does.
"""

from __future__ import annotations

import random

# The clip a labeller hears. 10s is enough to recognise a familiar voice and
# short enough that 400-odd of them stay a single sitting. The lead-in skips
# the head of the turn, where the previous speaker's tail bleeds across a
# diarisation boundary -- a clip that opens on the wrong voice is worse than
# no clip, because it is labelled rather than discarded. The minimum turn
# length is the sum: below it there is no full clip to cut.
CLIP_LENGTH_S = 10.0
CLIP_LEAD_IN_S = 2.0
CLIP_MIN_TURN_S = CLIP_LEAD_IN_S + CLIP_LENGTH_S

# The ceiling on a single segment's derived duration. Measured over the 400
# transcripts of the 2026-05-06 snapshot: 262,402 gaps, p50 4.2s, p90 11.1s,
# p99 22.3s, p99.9 29.4s, max 127.1s. At 30s the cap removes 0.01% of derived
# time (117 gaps) and only one gap in the corpus exceeds 60s -- so whisper
# segments are effectively contiguous and this corpus has no ad-break or
# music problem to correct for. The cap is kept anyway, because it bounds the
# damage a single pathological episode can do to a speech share, and because
# a stated bound is comparable between runs where an absent one is not.
GAP_CAP_S = 30.0

# Turns shorter than this are dropped before shares are computed:
# back-channel noise ("mm", "right", "yeah") is not a claim on the floor,
# and at a 4.2s median segment length it would otherwise inflate whichever
# speaker interjects most. Matches the filter the published figures used.
MIN_TURN_S = 1.5


def merge_turns(segments: list[dict], *, gap_cap_s: float) -> list[dict]:
    """Merge consecutive same-speaker segments into turns.

    A segment's duration is the gap to the next segment's start, clamped to
    `gap_cap_s`. This is NOT what `corpus.chunking.build_chunks` produces:
    that splits a long same-speaker run at MAX_CHUNK_WORDS and overlaps
    words across the boundary, so its chunks are not turns and its count is
    not a turn count.

    The episode's final segment contributes 0.0 seconds, because there is no
    next start to subtract from. Imputing one would be a model rather than a
    measurement; the turn keeps its words either way.
    """
    if not segments:
        return []

    durations: list[float] = []
    for i, segment in enumerate(segments):
        if i + 1 >= len(segments):
            durations.append(0.0)
            continue
        gap = segments[i + 1]["start"] - segment["start"]
        # A negative gap means the transcript's starts are not monotonic,
        # which is a data fault rather than a short turn. Clamping to zero
        # keeps it out of the totals; `speaker_stats.py` counts them so the
        # fault stays visible instead of being silently absorbed.
        durations.append(min(max(gap, 0.0), gap_cap_s))

    turns: list[dict] = []
    for i, segment in enumerate(segments):
        speaker = segment.get("speaker", "UNKNOWN")
        n_words = len(segment.get("text", "").split())
        if turns and turns[-1]["speaker"] == speaker:
            turn = turns[-1]
            turn["duration_s"] += durations[i]
            turn["n_words"] += n_words
            turn["n_segments"] += 1
        else:
            turns.append(
                {
                    "speaker": speaker,
                    "start": segment.get("start", 0.0),
                    "duration_s": durations[i],
                    "n_words": n_words,
                    "n_segments": 1,
                }
            )
    return turns


def count_non_monotonic(segments: list[dict]) -> int:
    """Segments whose successor starts earlier than they do."""
    return sum(
        1
        for i in range(len(segments) - 1)
        if segments[i + 1]["start"] < segments[i]["start"]
    )


def speech_shares(turns: list[dict], *, min_turn_s: float) -> dict:
    """Per-speaker seconds and words, and the top-1/top-2 concentration.

    Turns under `min_turn_s` leave BOTH the numerator and the denominator.
    Dropping them from one side only inflates the remaining speakers'
    shares by a few points -- large enough to move a conclusion, small
    enough to look right.
    """
    kept = [t for t in turns if t["duration_s"] >= min_turn_s]

    by_speaker_s: dict[str, float] = {}
    by_speaker_words: dict[str, int] = {}
    for turn in kept:
        speaker = turn["speaker"]
        by_speaker_s[speaker] = by_speaker_s.get(speaker, 0.0) + turn["duration_s"]
        by_speaker_words[speaker] = by_speaker_words.get(speaker, 0) + turn["n_words"]

    total_s = float(sum(by_speaker_s.values()))
    total_words = int(sum(by_speaker_words.values()))
    ranked_s = sorted(by_speaker_s.values(), reverse=True)
    ranked_words = sorted(by_speaker_words.values(), reverse=True)

    return {
        "by_speaker_s": by_speaker_s,
        "by_speaker_words": by_speaker_words,
        "total_s": total_s,
        "total_words": total_words,
        "n_speakers": len(by_speaker_s),
        "n_turns": len(kept),
        "top1_s_share": (ranked_s[0] / total_s) if total_s else 0.0,
        "top2_s_share": (sum(ranked_s[:2]) / total_s) if total_s else 0.0,
        "top1_words_share": (ranked_words[0] / total_words) if total_words else 0.0,
    }


def pick_clip_turn(
    turns: list[dict],
    speaker: str,
    *,
    seed: str,
    min_turn_s: float,
    draw: int = 0,
) -> dict | None:
    """Choose which of a speaker's turns to cut a clip from.

    Uniform at random over the eligible turns, seeded so the choice is
    reproducible from the record alone. **Not the longest turn**: the longest
    turn is the cleanest, longest-uninterrupted speech in the episode, which
    is exactly the condition the matcher performs best under. Selecting it
    would bias the precision estimate the same way picking easy-looking
    episodes would, one level further down.

    `seed` is a string the caller composes from a campaign seed and the
    episode's identity, so one campaign draws differently per episode while
    staying reproducible as a whole.

    `draw` is the redraw index, for when a clip turns out to be unusable
    (crosstalk, music over the voice). Successive draws walk a seeded
    permutation, so a redraw cannot return the turn just rejected -- without
    that, "redraw until it sounds clean" quietly becomes the
    longest-clearest-turn selection this function exists to avoid. Every
    redraw must be recorded with its draw index; an unrecorded one is a
    silent resample.

    Returns None when the speaker has no turn long enough. None, never a
    truncated clip: two seconds of a voice is not evidence about who is
    speaking, and one of those inside a set whose whole claim is zero errors
    would be a label nobody could stand behind.
    """
    eligible = [
        t for t in turns if t["speaker"] == speaker and t["duration_s"] >= min_turn_s
    ]
    if not eligible:
        return None
    order = list(range(len(eligible)))
    random.Random(seed).shuffle(order)
    return eligible[order[draw % len(order)]]


def clip_window(
    turn: dict, *, lead_in_s: float, length_s: float
) -> tuple[float, float]:
    """The (start, length) in seconds to cut from the audio for this turn.

    Never runs past the turn's end. Overrunning would capture the NEXT
    speaker, which is precisely how a clip labelled for one cluster ends up
    containing another -- and a contaminated clip scores the matcher as
    correct or incorrect for the wrong reason, invisibly.
    """
    duration = turn["duration_s"]
    if duration <= lead_in_s:
        # Too short to skip into. Take it from the top and let it be short;
        # the caller decides whether a stub is worth showing.
        return (turn["start"], min(length_s, duration))
    return (turn["start"] + lead_in_s, min(length_s, duration - lead_in_s))


# Shows with two recurring hosts. Tier 1 assigns one name per episode, so a
# two-host show is not a Tier 1 case at all -- it is Tier 2's entire reason
# for existing (§1.1). Routing the whole show away is deliberate: the
# alternative is naming one of two hosts and being right half the time.
TWO_HOST_SHOWS = frozenset({"Geopolitical Cousins"})

# Co-host SURNAMES, per show. Surnames, never forenames: measured over 341
# Jacob Shapiro transcripts, `marco` hit 36 episodes and `papic` hit 10, and
# 27 of the 37 `marco OR papic` hits never say `papic` anywhere -- the signal
# is dominated by Marco Rubio, in a geopolitics podcast. Routing on the
# forename would send a quarter of the archive to Tier 2 over a different
# person.
CO_HOST_SURNAMES: dict[str, tuple[str, ...]] = {
    "The Jacob Shapiro Podcast": ("papic",),
}


def routes_to_tier2(show: str, transcript_text: str) -> bool:
    """Whether this episode is Tier 2's problem rather than Tier 1's.

    The window is the WHOLE transcript, not the opening segments: 14 of the
    24 episodes with real co-host evidence say the name only later, so a
    windowed check misses more than half of them.

    This fails safe. A false positive costs Tier 1 an episode it could have
    named; a false negative hands Tier 1 a two-voice episode and invites the
    confident misattribution the design forbids.
    """
    if show in TWO_HOST_SHOWS:
        return True
    lowered = transcript_text.lower()
    return any(surname in lowered for surname in CO_HOST_SURNAMES.get(show, ()))


def evenly_spaced(items: list, target: int) -> list:
    """`target` items spread across the whole of `items`, in order.

    Not `items[::stride]` with an integer stride. When the pool is less than
    twice the target that stride collapses to 1 and the result is the pool's
    OLDEST `target` entries -- not a sample of the archive but its beginning,
    which for a date-ordered pool hands the eval precisely the era the corpus
    has moved away from.
    """
    if target <= 0 or not items:
        return []
    if target >= len(items):
        return list(items)
    step = len(items) / target
    return [items[int(i * step)] for i in range(target)]
