"""Crop selection and centroid maths.

Two quantities are deliberately different here and conflating them is the
defect this module exists to avoid: a centroid is weighted by a turn's TRUE
duration, while the audio embedded is a duration-CAPPED crop of it. Weighting
by the crop length would silently under-weight the longest, cleanest turns --
exactly the turns that carry the most evidence about a voice.
"""

from __future__ import annotations

import math

import pytest

from corpus.embedding import (
    CROP_LENGTHS_S,
    cluster_crop_plan,
    select_crop_turns,
    weighted_centroid,
)

TURNS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "duration_s": 40.0, "n_words": 100},
    {"speaker": "SPEAKER_00", "start": 100.0, "duration_s": 15.0, "n_words": 40},
    {"speaker": "SPEAKER_00", "start": 200.0, "duration_s": 60.0, "n_words": 150},
    {"speaker": "SPEAKER_00", "start": 300.0, "duration_s": 5.0, "n_words": 10},
    {"speaker": "SPEAKER_01", "start": 400.0, "duration_s": 90.0, "n_words": 200},
    {"speaker": "UNKNOWN", "start": 500.0, "duration_s": 80.0, "n_words": 180},
]


def test_selects_the_longest_turns_descending():
    picked = select_crop_turns(TURNS, "SPEAKER_00", k=3, min_turn_s=12.0)
    assert [t["duration_s"] for t in picked] == [60.0, 40.0, 15.0]


def test_excludes_turns_below_the_floor():
    picked = select_crop_turns(TURNS, "SPEAKER_00", k=10, min_turn_s=12.0)
    assert 5.0 not in [t["duration_s"] for t in picked]


def test_returns_fewer_than_k_when_the_speaker_is_short_of_turns():
    picked = select_crop_turns(TURNS, "SPEAKER_01", k=3, min_turn_s=12.0)
    assert len(picked) == 1


def test_returns_empty_when_no_turn_is_long_enough():
    # 44 real clusters archive-wide are in exactly this position. They must
    # produce no crops rather than a short one -- a two-second crop is not
    # evidence about who is speaking.
    assert select_crop_turns(TURNS, "SPEAKER_00", k=3, min_turn_s=120.0) == []


def test_equal_durations_break_ties_by_start_so_the_plan_is_deterministic():
    turns = [
        {"speaker": "A", "start": 50.0, "duration_s": 20.0},
        {"speaker": "A", "start": 10.0, "duration_s": 20.0},
    ]
    assert [t["start"] for t in select_crop_turns(turns, "A", k=2, min_turn_s=1.0)] == [
        10.0,
        50.0,
    ]


def test_crop_plan_excludes_non_speaker_buckets():
    # chunking.py:71 writes UNKNOWN for segments with no speaker. Embedding it
    # would build a centroid from audio belonging to nobody in particular.
    plan = cluster_crop_plan({"name": "ep.txt", "turns": TURNS}, min_turn_s=12.0)
    assert {r["cluster"] for r in plan} == {"SPEAKER_00", "SPEAKER_01"}


def test_crop_plan_emits_one_record_per_cluster_turn_and_length():
    plan = cluster_crop_plan({"name": "ep.txt", "turns": TURNS}, min_turn_s=12.0)
    # SPEAKER_00 contributes 3 turns, SPEAKER_01 contributes 1, times 2 lengths.
    assert len(plan) == (3 + 1) * len(CROP_LENGTHS_S)


def test_crop_never_runs_past_the_end_of_its_turn():
    plan = cluster_crop_plan({"name": "ep.txt", "turns": TURNS}, min_turn_s=12.0)
    for record in plan:
        end = record["crop_start_s"] + record["crop_length_s"]
        assert end <= record["turn_start"] + record["turn_duration_s"] + 1e-9


def test_crop_plan_records_the_true_turn_duration_not_the_crop_length():
    plan = cluster_crop_plan({"name": "ep.txt", "turns": TURNS}, min_turn_s=12.0)
    longest = max(plan, key=lambda r: r["turn_duration_s"])
    assert longest["turn_duration_s"] == 90.0
    assert longest["crop_length_s"] <= 30.0


def test_centroid_is_unit_length():
    out = weighted_centroid([[3.0, 4.0], [0.0, 1.0]], [1.0, 1.0])
    assert math.isclose(math.sqrt(sum(v * v for v in out)), 1.0, rel_tol=1e-9)


def test_centroid_weighting_pulls_toward_the_heavier_vector():
    light = weighted_centroid([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    heavy = weighted_centroid([[1.0, 0.0], [0.0, 1.0]], [9.0, 1.0])
    assert heavy[0] > light[0]


def test_centroid_rejects_empty_input():
    with pytest.raises(ValueError, match="no vectors"):
        weighted_centroid([], [])


def test_centroid_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights"):
        weighted_centroid([[1.0, 0.0]], [1.0, 2.0])


def test_centroid_rejects_ragged_vectors():
    with pytest.raises(ValueError, match="dimension"):
        weighted_centroid([[1.0, 0.0], [1.0]], [1.0, 1.0])


def test_centroid_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="weight"):
        weighted_centroid([[1.0, 0.0]], [0.0])
