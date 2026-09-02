"""Turn merging and speech shares.

Every duration in this module is DERIVED. A transcript line carries only a
start time (`[SPEAKER_00] 12.3s - ...`), so a segment's duration is the gap to
the next segment's start and the episode's last segment has no successor at
all. These tests pin that derivation, because the figures it feeds -- who
dominates an episode -- are the input to the Tier 1 hypothesis.
"""

from __future__ import annotations

import math

from corpus.chunking import build_chunks
from corpus.speakers import count_non_monotonic, merge_turns, speech_shares

# Four segments, three turns: A speaks twice, B interrupts, A returns last.
SEGMENTS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "text": "one two"},
    {"speaker": "SPEAKER_00", "start": 10.0, "text": "three"},
    {"speaker": "SPEAKER_01", "start": 15.0, "text": "four five six"},
    {"speaker": "SPEAKER_00", "start": 20.0, "text": "seven"},
]


def test_consecutive_same_speaker_segments_become_one_turn():
    turns = merge_turns(SEGMENTS, gap_cap_s=30.0)
    assert [t["speaker"] for t in turns] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]
    assert turns[0]["n_segments"] == 2
    assert turns[0]["n_words"] == 3


def test_turn_duration_sums_the_gaps_to_the_next_segment():
    turns = merge_turns(SEGMENTS, gap_cap_s=30.0)
    # 0.0->10.0 plus 10.0->15.0. The turn's own end is the next speaker's
    # start, so the gap that crosses the speaker change belongs to the
    # speaker who was still talking.
    assert turns[0]["duration_s"] == 15.0
    assert turns[0]["start"] == 0.0
    assert turns[1]["duration_s"] == 5.0


def test_the_final_segment_of_an_episode_contributes_no_duration():
    turns = merge_turns(SEGMENTS, gap_cap_s=30.0)
    # There is no next start to subtract from, and inventing one would be a
    # model rather than a measurement. The turn still exists and still
    # carries its words -- only its seconds are unknowable.
    assert turns[-1]["duration_s"] == 0.0
    assert turns[-1]["n_words"] == 1


def test_a_long_gap_is_capped():
    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "text": "x"},
        {"speaker": "SPEAKER_00", "start": 600.0, "text": "y"},
        {"speaker": "SPEAKER_01", "start": 610.0, "text": "z"},
    ]
    turns = merge_turns(segments, gap_cap_s=30.0)
    # 600s of silence, music or advertising is not speech, and uncapped it
    # would all be charged to whoever happened to speak before it.
    assert turns[0]["duration_s"] == 40.0


def test_an_infinite_cap_reproduces_the_naive_derivation():
    # The uncapped run is the positive control against the published
    # figures: it isolates "our method differs" from "the cap differs".
    turns = merge_turns(
        [
            {"speaker": "SPEAKER_00", "start": 0.0, "text": "x"},
            {"speaker": "SPEAKER_00", "start": 600.0, "text": "y"},
            {"speaker": "SPEAKER_01", "start": 610.0, "text": "z"},
        ],
        gap_cap_s=math.inf,
    )
    assert turns[0]["duration_s"] == 610.0


def test_turns_are_not_chunks():
    # build_chunks splits a long same-speaker run at MAX_CHUNK_WORDS and
    # overlaps CHUNK_OVERLAP_WORDS into the next chunk, so chunk boundaries
    # are not speaker-change boundaries. A turn count derived from chunks
    # would be wrong by however often a speaker talks past 400 words.
    long_run = [
        {"speaker": "SPEAKER_00", "start": 0.0, "text": " ".join(["w"] * 300)},
        {"speaker": "SPEAKER_00", "start": 60.0, "text": " ".join(["w"] * 300)},
    ]
    assert len(merge_turns(long_run, gap_cap_s=30.0)) == 1
    chunks = build_chunks(long_run, "Show", "1", "Title", "2026-01-01")
    assert len(chunks) > 1


def test_no_segments_yields_no_turns():
    assert merge_turns([], gap_cap_s=30.0) == []


def test_a_backwards_start_contributes_zero_rather_than_a_negative():
    # Non-monotonic starts are a data fault, not a short turn. A negative
    # duration would silently subtract from a speaker's total and could
    # even flip which cluster looks dominant.
    segments = [
        {"speaker": "SPEAKER_00", "start": 100.0, "text": "x"},
        {"speaker": "SPEAKER_00", "start": 40.0, "text": "y"},
        {"speaker": "SPEAKER_00", "start": 50.0, "text": "z"},
    ]
    turns = merge_turns(segments, gap_cap_s=30.0)
    assert turns[0]["duration_s"] == 10.0
    # Clamped, but counted -- absorbing the fault without reporting it is
    # how a corrupt transcript produces a confident number.
    assert count_non_monotonic(segments) == 1


# Hand-computed: A 60s/100w, B 30s/50w, C 10s/20w, plus a 1.0s A turn that
# the minimum-length filter must remove.
TURNS = [
    {"speaker": "A", "start": 0.0, "duration_s": 60.0, "n_words": 100, "n_segments": 3},
    {"speaker": "B", "start": 60.0, "duration_s": 30.0, "n_words": 50, "n_segments": 2},
    {"speaker": "C", "start": 90.0, "duration_s": 10.0, "n_words": 20, "n_segments": 1},
    {"speaker": "A", "start": 100.0, "duration_s": 1.0, "n_words": 2, "n_segments": 1},
]


def test_shares_match_a_hand_computed_confusion_of_seconds():
    s = speech_shares(TURNS, min_turn_s=1.5)
    assert s["total_s"] == 100.0
    assert s["top1_s_share"] == 0.6
    assert s["top2_s_share"] == 0.9
    assert s["n_speakers"] == 3


def test_a_short_turn_leaves_both_numerator_and_denominator():
    # If the dropped 1.0s turn stayed in the denominator the total would be
    # 101.0 and top-1 would read 59.4%. Filtering one side only is the
    # quiet way to get a plausible wrong number.
    s = speech_shares(TURNS, min_turn_s=1.5)
    assert s["total_s"] == 100.0
    assert s["by_speaker_s"]["A"] == 60.0
    assert s["n_turns"] == 3


def test_words_are_tracked_separately_from_seconds():
    # Seconds are derived and words are counted. When the two disagree about
    # who dominates, the disagreement is the finding -- so they must never
    # be computed from one another.
    s = speech_shares(TURNS, min_turn_s=1.5)
    assert s["total_words"] == 170
    assert s["by_speaker_words"]["A"] == 100
    assert s["top1_words_share"] == 100 / 170


def test_no_turns_yields_zeroes_rather_than_a_division_error():
    s = speech_shares([], min_turn_s=1.5)
    assert s["total_s"] == 0.0
    assert s["top1_s_share"] == 0.0
    assert s["n_speakers"] == 0
