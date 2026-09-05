"""`identify_clusters` — the ingest-time path from segments to speaker_id.

This is the composition that removes the need for a backfill: the nightly
job already holds the diarised segments and the audio file, so identity can
be decided before the chunks are ever written rather than repaired
afterwards.

The load-bearing property is not "does it assign someone". It is **does it
build a cluster centroid the same way `assign_speakers.py` does**. The
enrolled centroids it scores against were built from clips; the archive
centroids that validated the thresholds were built from crops weighted by
TRUE TURN DURATION, not by crop length. Weight the ingest centroid any other
way and every vector it produces is quietly incomparable to the 0.516/0.074
that were measured — assignments would still be produced, and they would
still look reasonable.

`test_centroid_is_weighted_by_turn_duration` is the one that catches that,
and it is built to fail rather than to pass: cluster A holds a 150s turn
that embeds to [1,0] and a 30s turn that embeds to [0,1]. Duration-weighted
those combine 5:1 and land firmly on Alice. Weighted equally — one vote per
crop — they combine 2:2, the two scores tie, and the margin rule correctly
refuses to assign anyone. So the mutation from `turn_duration_s` to
`crop_length_s` turns an assignment into a None.

The fake embedder keys on crop start time, which is the only thing that
distinguishes one turn from another here, and never touches audio.
"""

from __future__ import annotations

import pytest

from corpus.identify import identify_clusters

SEGMENT_GAP_S = 30.0  # equals GAP_CAP_S, so each segment contributes its full gap


def _segments(runs: list[tuple[str, int]]) -> list[dict]:
    """Segments `SEGMENT_GAP_S` apart, as `(speaker, count)` runs.

    `merge_turns` derives a segment's duration from the gap to the NEXT
    segment's start, so a run of n segments becomes one turn of n * gap --
    except the episode's final segment, which contributes 0.0 because there
    is no next start.
    """
    segments: list[dict] = []
    start = 0.0
    for speaker, count in runs:
        for _ in range(count):
            segments.append({"speaker": speaker, "start": start, "text": "words here"})
            start += SEGMENT_GAP_S
    return segments


def _embedder(fn):
    """Anything with `.embed(path, start_s, length_s)`; audio is never touched."""

    class _Fake:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float, float]] = []

        def embed(self, path: str, start_s: float, length_s: float) -> list[float]:
            self.calls.append((path, start_s, length_s))
            return fn(start_s)

    return _Fake()


ALICE = [1.0, 0.0]
BOB = [0.0, 1.0]
CENTROIDS = {"Alice": ALICE, "Bob": BOB}
TUNING = {"threshold": 0.516, "margin": 0.074}


def test_a_cluster_matching_an_enrolled_voice_is_assigned():
    segments = _segments([("SPEAKER_00", 5), ("SPEAKER_01", 2), ("SPEAKER_00", 1)])
    embedder = _embedder(lambda _start: ALICE)

    out = identify_clusters(
        segments, "episode.mp3", embedder, {"Alice": ALICE}, **TUNING
    )

    assert out["SPEAKER_00"] == "Alice"


def test_a_cluster_matching_nobody_is_absent_from_the_result():
    """No match writes NOTHING. An absent key is the unidentified state."""
    segments = _segments([("SPEAKER_00", 5), ("SPEAKER_01", 2)])
    embedder = _embedder(lambda _start: [0.7071, 0.7071])

    out = identify_clusters(segments, "episode.mp3", embedder, CENTROIDS, **TUNING)

    # Equidistant from both enrolled voices: over threshold, but the margin
    # rule refuses rather than picking one of two equally good answers.
    assert "SPEAKER_00" not in out
    assert out == {}


def test_centroid_is_weighted_by_turn_duration():
    """The subtle one: crop-length weighting would silently break comparability.

    Cluster SPEAKER_00 holds a 150s turn embedding to Alice and a 30s turn
    embedding to Bob. Weighted by turn duration that is 5:1 and resolves to
    Alice. Weighted per-crop it is 2:2, the scores tie, and nobody is
    assigned -- so this test distinguishes the two rules by outcome.
    """
    segments = _segments(
        [("SPEAKER_00", 5), ("SPEAKER_01", 1), ("SPEAKER_00", 1), ("SPEAKER_01", 1)]
    )
    # The first SPEAKER_00 turn starts at 0s, the second at 180s.
    embedder = _embedder(lambda start: ALICE if start < 100.0 else BOB)

    out = identify_clusters(segments, "episode.mp3", embedder, CENTROIDS, **TUNING)

    assert out["SPEAKER_00"] == "Alice"


def test_non_speaker_buckets_are_never_enrolled():
    """`UNKNOWN` is audio belonging to nobody; a centroid from it is a merge."""
    segments = _segments([("UNKNOWN", 5), ("SPEAKER_00", 5)])
    embedder = _embedder(lambda _start: ALICE)

    out = identify_clusters(
        segments, "episode.mp3", embedder, {"Alice": ALICE}, **TUNING
    )

    assert "UNKNOWN" not in out
    assert out == {"SPEAKER_00": "Alice"}


def test_turns_too_short_to_crop_yield_no_centroid():
    """Below CLIP_MIN_TURN_S there is no full crop, so there is no evidence."""
    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "text": "a"},
        {"speaker": "SPEAKER_01", "start": 3.0, "text": "b"},
        {"speaker": "SPEAKER_00", "start": 6.0, "text": "c"},
        {"speaker": "SPEAKER_01", "start": 9.0, "text": "d"},
    ]
    embedder = _embedder(lambda _start: ALICE)

    out = identify_clusters(
        segments, "episode.mp3", embedder, {"Alice": ALICE}, **TUNING
    )

    assert out == {}
    assert embedder.calls == [], "nothing should have been embedded"


def test_no_segments_is_not_an_error():
    """An episode diarisation produced nothing for is unidentified, not fatal."""
    embedder = _embedder(lambda _start: ALICE)

    out = identify_clusters([], "episode.mp3", embedder, CENTROIDS, **TUNING)

    assert out == {}


def test_the_audio_path_is_passed_through_untouched():
    """The crop reads the file the caller named, at the planned offsets."""
    segments = _segments([("SPEAKER_00", 5), ("SPEAKER_01", 2)])
    embedder = _embedder(lambda _start: ALICE)

    identify_clusters(segments, "/tmp/ep.mp3", embedder, {"Alice": ALICE}, **TUNING)

    assert embedder.calls, "no crops were embedded at all"
    assert {c[0] for c in embedder.calls} == {"/tmp/ep.mp3"}
    # Two windows per drawn turn (corpus.embedding.CROP_LENGTHS_S), but
    # `clip_window` CLAMPS a window to what the turn affords after the
    # lead-in -- a 30s window on a 30s turn yields 28s, not 30s. So the
    # invariant is a bound, not an equality.
    lengths = [c[2] for c in embedder.calls]
    assert len(lengths) % 2 == 0, "crops should come in one pair per drawn turn"
    assert all(0.0 < n <= 30.0 for n in lengths), lengths
    assert 10.0 in lengths, "the short window should be unclamped here"


def test_an_embedder_failure_propagates():
    """Identity must not silently degrade to 'unidentified' on a broken model.

    The caller decides what to do with the episode; swallowing it here would
    make a model fault indistinguishable from a genuinely unknown voice.
    """
    segments = _segments([("SPEAKER_00", 5), ("SPEAKER_01", 2)])

    def _boom(_start):
        raise RuntimeError("model is not loaded")

    with pytest.raises(RuntimeError, match="model is not loaded"):
        identify_clusters(
            segments, "episode.mp3", _embedder(_boom), CENTROIDS, **TUNING
        )


# --- stamp_chunks -----------------------------------------------------------
#
# The join that silently regresses. `identify_clusters` can be perfect and
# the ingest still write nothing, if the metadata key `chunking.py` uses ever
# drifts from the one looked up here. Every log line would still report
# clusters identified.


def _chunk(speaker: str) -> dict:
    return {"text": f"[{speaker}] words", "metadata": {"speaker": speaker}}


def test_stamp_writes_speaker_id_onto_matching_chunks():
    from corpus.identify import stamp_chunks

    chunks = [_chunk("SPEAKER_00"), _chunk("SPEAKER_01"), _chunk("SPEAKER_00")]

    stamped = stamp_chunks(chunks, {"SPEAKER_00": "Alice"})

    assert stamped == 2
    assert [c["metadata"].get("speaker_id") for c in chunks] == [
        "Alice",
        None,
        "Alice",
    ]


def test_stamp_leaves_unidentified_chunks_without_the_key():
    """Absent IS the unidentified state; a placeholder would be filterable."""
    from corpus.identify import stamp_chunks

    chunks = [_chunk("SPEAKER_01")]

    stamped = stamp_chunks(chunks, {"SPEAKER_00": "Alice"})

    assert stamped == 0
    assert "speaker_id" not in chunks[0]["metadata"]


def test_stamp_reads_the_key_chunking_actually_writes():
    """Pins the join to `build_chunks`' real output, not to an assumption."""
    from corpus.chunking import build_chunks
    from corpus.identify import stamp_chunks

    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "text": "hello there friend"},
        {"speaker": "SPEAKER_01", "start": 5.0, "text": "and hello to you"},
    ]
    chunks = build_chunks(segments, "A Show", "1", "A Title", "2026-01-01")
    assert chunks, "build_chunks produced nothing; the fixture is wrong"

    stamped = stamp_chunks(chunks, {"SPEAKER_00": "Alice"})

    assert stamped > 0, (
        "stamped nothing against real build_chunks output -- the metadata key "
        "this joins on has drifted from the one chunking.py writes"
    )
    assert any(c["metadata"].get("speaker_id") == "Alice" for c in chunks)


# --- raw whisperx segment shape ---------------------------------------------
#
# identify_clusters is the ONLY consumer of raw result["segments"] from
# whisperx.assign_word_speakers. Every other merge_turns caller feeds
# segments parsed from a saved transcript, where `start` is guaranteed by the
# line format. corpus/speakers.py indexes segments[i+1]["start"] unguarded --
# while defending `speaker` with .get() two lines later -- so one malformed
# segment would raise KeyError and lose identity for the whole episode.


def test_a_segment_without_start_does_not_kill_the_episode():
    from corpus.identify import identify_clusters

    segments = _segments([("SPEAKER_00", 5), ("SPEAKER_01", 2)])
    segments.insert(2, {"speaker": "SPEAKER_00", "text": "no start key"})
    embedder = _embedder(lambda _start: ALICE)

    out = identify_clusters(
        segments, "episode.mp3", embedder, {"Alice": ALICE}, **TUNING
    )

    assert out.get("SPEAKER_00") == "Alice"


def test_a_non_numeric_start_is_dropped_too():
    """whisperx is not the only possible producer; a string start is garbage."""
    from corpus.identify import usable_segments

    kept, dropped = usable_segments(
        [
            {"speaker": "SPEAKER_00", "start": 0.0},
            {"speaker": "SPEAKER_00", "start": "12.5"},
            {"speaker": "SPEAKER_00", "start": None},
            {"speaker": "SPEAKER_00", "start": 30.0},
        ]
    )

    assert [s["start"] for s in kept] == [0.0, 30.0]
    assert dropped == 2


def test_a_bool_start_is_not_a_number():
    """True == 1 in Python; a bool here is a shape error, not a timestamp."""
    from corpus.identify import usable_segments

    kept, dropped = usable_segments([{"speaker": "SPEAKER_00", "start": True}])

    assert kept == []
    assert dropped == 1


def test_well_formed_segments_are_untouched():
    """The guard must not quietly thin a healthy episode."""
    from corpus.identify import usable_segments

    segments = _segments([("SPEAKER_00", 5), ("SPEAKER_01", 2)])

    kept, dropped = usable_segments(segments)

    assert kept == segments
    assert dropped == 0
