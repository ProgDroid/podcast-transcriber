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
