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

    stamped = stamp_chunks(chunks, {"SPEAKER_00": "Alice"}, identity_version="1@1")

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

    stamped = stamp_chunks(chunks, {"SPEAKER_00": "Alice"}, identity_version="1@1")

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

    stamped = stamp_chunks(chunks, {"SPEAKER_00": "Alice"}, identity_version="1@1")

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


# --- stamp_chunks writes identity_version -----------------------------------


def test_the_version_lands_on_every_chunk_including_unmatched_clusters():
    # Uniformity is a REQUIREMENT, not a nicety: episode_state reads metadata
    # from one arbitrary record (ids[:1] after an offset-paged get with no
    # ordering guarantee, so "the first id" is not "chunk 0"). That read is
    # only sound if every record of an episode agrees.
    from corpus.identify import stamp_chunks

    chunks = [_chunk("SPEAKER_00"), _chunk("SPEAKER_01"), _chunk("UNKNOWN")]

    stamp_chunks(chunks, {"SPEAKER_00": "Alice"}, identity_version="1@1")

    assert [c["metadata"].get("identity_version") for c in chunks] == [
        "1@1",
        "1@1",
        "1@1",
    ]


def test_nobody_matching_still_counts_as_a_completed_identification():
    # 27% of archive clusters legitimately match nobody. If an empty
    # `identified` withheld the version, every such episode would be re-queued
    # for audio every night forever -- the non-terminating shape the whole key
    # exists to kill.
    from corpus.identify import stamp_chunks

    chunks = [_chunk("SPEAKER_00")]

    stamped = stamp_chunks(chunks, {}, identity_version="1@1")

    assert stamped == 0
    assert chunks[0]["metadata"]["identity_version"] == "1@1"
    assert "speaker_id" not in chunks[0]["metadata"]


def test_a_broken_stamp_join_withholds_the_version():
    # Clusters named and not one chunk taking a value can only mean the
    # `speaker` metadata key drifted between chunking.py and this function.
    # Stamping the version there would record "identification ran" for an
    # episode that stored no identity, retiring it from repair FOREVER and
    # converting a loud nightly failure into a silent permanent one.
    from corpus.identify import stamp_chunks

    chunks = [{"text": "words", "metadata": {"speaker_label": "SPEAKER_00"}}]

    stamped = stamp_chunks(chunks, {"SPEAKER_00": "Alice"}, identity_version="1@1")

    assert stamped == 0
    assert "identity_version" not in chunks[0]["metadata"]


def test_no_chunks_at_all_returns_zero_rather_than_raising():
    # Named for what it can actually catch. With no chunks the version
    # is unwritten either way, so this cannot distinguish the broken-join
    # exclusion from its absence -- it only pins that an empty episode
    # does not index chunks[0].
    from corpus.identify import stamp_chunks

    assert stamp_chunks([], {"SPEAKER_00": "Alice"}, identity_version="1@1") == 0


# --- carry_identity ---------------------------------------------------------
#
# The cluster->person mapping is invariant under re-chunking: EMBED_ONLY and
# RECHUNK both re-chunk the SAME transcript, so its [SPEAKER_XX] labels are
# unchanged and only the positional layout moves. The existing records already
# hold the mapping, so it is carried forward rather than re-earned from audio.


def _stored(speaker: str, person: str | None = None, **extra) -> dict:
    meta = {"speaker": speaker}
    if person is not None:
        meta["speaker_id"] = person
    meta.update(extra)
    return meta


def test_the_map_is_joined_on_the_label_never_on_the_position():
    # THE POINT OF THE WHOLE FUNCTION. The stored records and the incoming
    # chunks deliberately disagree on position AND on count -- a fixture
    # where chunk i lines up with stored record i would pass against a
    # positional implementation and be a check that cannot fail.
    from corpus.identify import carry_identity

    stored = [
        _stored("SPEAKER_00", "Alice"),
        _stored("SPEAKER_01", "Bob"),
        _stored("SPEAKER_00", "Alice"),
    ]
    chunks = [
        _chunk("SPEAKER_01"),
        _chunk("SPEAKER_01"),
        _chunk("SPEAKER_00"),
        _chunk("SPEAKER_01"),
    ]

    stamped, refused, unmapped = carry_identity(stored, chunks, identity_version="1@1")

    assert (refused, unmapped) == ([], [])
    assert stamped == 4
    assert [c["metadata"]["speaker_id"] for c in chunks] == [
        "Bob",
        "Bob",
        "Alice",
        "Bob",
    ]


def test_the_version_is_written_on_every_carried_chunk():
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice"), _stored("SPEAKER_01")]
    chunks = [_chunk("SPEAKER_00"), _chunk("SPEAKER_01")]

    carry_identity(stored, chunks, identity_version="1@1")

    assert [c["metadata"].get("identity_version") for c in chunks] == ["1@1", "1@1"]


def test_a_cluster_that_matched_nobody_is_carried_as_considered_not_named():
    # The map is keyed on EVERY stored label, valued person-or-None: attempt,
    # not outcome. Keying it only on named clusters cannot work -- 27% of
    # clusters legitimately match nobody, so a label with no name would be
    # indistinguishable from a label that has gone missing.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice"), _stored("SPEAKER_01")]
    chunks = [_chunk("SPEAKER_01")]

    stamped, refused, unmapped = carry_identity(stored, chunks, identity_version="1@1")

    assert (stamped, refused, unmapped) == (0, [], [])
    assert "speaker_id" not in chunks[0]["metadata"]
    assert chunks[0]["metadata"]["identity_version"] == "1@1"


def test_present_beats_absent_within_one_label():
    # The section 1.1 row-3 shape: a torn episode's repair GROWS the record
    # set, so old positions carry a name and the new ones carry nothing. The
    # absent records are simply the ones written before identification ran.
    from corpus.identify import carry_identity

    stored = [
        _stored("SPEAKER_00", "Alice"),
        _stored("SPEAKER_00"),
        _stored("SPEAKER_00"),
    ]
    chunks = [_chunk("SPEAKER_00")]

    stamped, refused, _ = carry_identity(stored, chunks, identity_version="1@1")

    assert (stamped, refused) == (1, [])
    assert chunks[0]["metadata"]["speaker_id"] == "Alice"


def test_a_label_naming_two_people_is_refused_and_carries_nothing():
    # Its stored records disagree, so the map is self-contradictory. Carrying
    # it would propagate whichever name happened to be read first.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice"), _stored("SPEAKER_00", "Bob")]
    chunks = [_chunk("SPEAKER_00")]

    stamped, refused, _ = carry_identity(stored, chunks, identity_version="1@1")

    assert stamped == 0
    assert refused == ["SPEAKER_00"]
    assert "speaker_id" not in chunks[0]["metadata"]


def test_a_refusal_withholds_the_version_so_the_episode_re_earns_from_audio():
    # Fails safe toward re-earning from audio, never toward certifying a gap.
    # Writing the version here would send the episode to SKIP with a speaker
    # we explicitly refused to trust, permanently, and nothing signalling it.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice"), _stored("SPEAKER_00", "Bob")]
    chunks = [_chunk("SPEAKER_00")]

    carry_identity(stored, chunks, identity_version="1@1")

    assert "identity_version" not in chunks[0]["metadata"]


def test_a_conflict_is_reported_as_conflicting_not_as_corruption():
    # Two things a disagreement can mean, and only one of them is a data
    # fault. A same-day unnumbered TRIPLE COLLISION makes two distinct
    # episodes share episode_id_prefix and interleave, at which point one
    # label legitimately carries two people. Refusing is right either way;
    # calling it stamp corruption is not.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice"), _stored("SPEAKER_00", "Bob")]

    _, refused, _ = carry_identity(
        stored, [_chunk("SPEAKER_00")], identity_version="1@1"
    )

    assert refused == ["SPEAKER_00"]


def test_an_unknown_bucket_is_never_named_even_when_stored_records_name_it():
    # UNKNOWN is chunking.py's literal substitute for a segment carrying no
    # speaker -- 53 of the 439 transcripts hold such a bucket. It is audio
    # belonging to nobody in particular, and naming it is the silent
    # over-merge this whole line of work treats as the error that matters
    # most. The fixture MUST supply a speaker_id, or the test passes against
    # an implementation with no such rule and becomes a check that cannot
    # fail.
    from corpus.identify import carry_identity

    stored = [_stored("UNKNOWN", "Alice"), _stored("SPEAKER_00", "Bob")]
    chunks = [_chunk("UNKNOWN"), _chunk("SPEAKER_00")]

    carry_identity(stored, chunks, identity_version="1@1")

    assert "speaker_id" not in chunks[0]["metadata"]
    assert chunks[1]["metadata"]["speaker_id"] == "Bob"


def test_an_unknown_bucket_still_receives_the_version():
    # Uniformity: episode_state reads one arbitrary record, so an UNKNOWN
    # chunk left unstamped could be the record it reads.
    from corpus.identify import carry_identity

    stored = [_stored("UNKNOWN"), _stored("SPEAKER_00", "Bob")]
    chunks = [_chunk("UNKNOWN"), _chunk("SPEAKER_00")]

    carry_identity(stored, chunks, identity_version="1@1")

    assert chunks[0]["metadata"]["identity_version"] == "1@1"


def test_an_unknown_bucket_absent_from_the_stored_map_does_not_fail_the_subset():
    # UNKNOWN is excluded from BOTH sides of the subset test. Counting it
    # would fail all 53 UNKNOWN-bucket transcripts on every run and
    # re-download their audio nightly forever.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice")]
    chunks = [_chunk("SPEAKER_00"), _chunk("UNKNOWN")]

    stamped, refused, unmapped = carry_identity(stored, chunks, identity_version="1@1")

    assert (refused, unmapped) == ([], [])
    assert stamped == 1
    assert chunks[1]["metadata"]["identity_version"] == "1@1"


# --- the subset rule --------------------------------------------------------
#
# Two failure modes, one remedy. Carry only when the incoming label set is a
# subset of the stored one; otherwise carry NOTHING and withhold the version,
# sending the episode to IDENTIFY_ONLY to re-earn from audio.


def test_a_renamed_label_abandons_the_whole_carry():
    # Case 1: the invariance claim is false on exactly the path RECHUNK
    # exists for. RULES_VERSION is bumped whenever anything deriving chunk
    # CONTENT from a transcript changes -- "today the speaker labels" -- so a
    # rules bump is precisely the event that may change how `speaker` is
    # derived. The transcript's labels are invariant; the DERIVATION is not.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice"), _stored("SPEAKER_01", "Bob")]
    chunks = [_chunk("SPEAKER_00"), _chunk("SPEAKER_02")]

    stamped, refused, unmapped = carry_identity(stored, chunks, identity_version="1@1")

    assert stamped == 0
    assert unmapped == ["SPEAKER_02"]
    assert refused == []


def test_a_label_only_in_a_torn_episodes_missing_tail_abandons_the_carry():
    # Case 2, and this one is LIVE in phase 1 with no rules bump at all. A
    # tear removes a contiguous tail of positions, so a cluster appearing
    # only in that tail has no stored record and no map entry. Writing the
    # version anyway would send the episode to SKIP with that speaker
    # permanently unnamed and nothing signalling it.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice")]
    chunks = [_chunk("SPEAKER_00"), _chunk("SPEAKER_00"), _chunk("SPEAKER_01")]

    stamped, _, unmapped = carry_identity(stored, chunks, identity_version="1@1")

    assert stamped == 0
    assert unmapped == ["SPEAKER_01"]


def test_an_abandoned_carry_leaves_no_name_and_no_version_anywhere():
    # "Carry NOTHING" is literal: a partial carry would stamp the labels it
    # could and certify an episode with a hole in it.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice")]
    chunks = [_chunk("SPEAKER_00"), _chunk("SPEAKER_01")]

    carry_identity(stored, chunks, identity_version="1@1")

    for chunk in chunks:
        assert "speaker_id" not in chunk["metadata"]
        assert "identity_version" not in chunk["metadata"]


def test_a_strict_subset_carries_fine():
    # A re-chunk that drops a label entirely -- the stored set is a superset,
    # which the rule permits. Only the reverse is unsafe.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice"), _stored("SPEAKER_01", "Bob")]
    chunks = [_chunk("SPEAKER_00")]

    stamped, refused, unmapped = carry_identity(stored, chunks, identity_version="1@1")

    assert (stamped, refused, unmapped) == (1, [], [])


def test_no_stored_records_abandons_rather_than_certifies():
    # Nothing to carry is not the same as nothing to do. An episode with no
    # stored records has no map, so every incoming label is unmapped and the
    # episode must go to audio, not to SKIP.
    from corpus.identify import carry_identity

    chunks = [_chunk("SPEAKER_00")]

    stamped, _, unmapped = carry_identity([], chunks, identity_version="1@1")

    assert stamped == 0
    assert unmapped == ["SPEAKER_00"]
    assert "identity_version" not in chunks[0]["metadata"]


def test_the_reports_are_ordered_so_two_runs_read_the_same():
    # Set iteration order would make an otherwise identical nightly line
    # differ between runs, which is exactly what makes a log unreadable
    # across nights.
    from corpus.identify import carry_identity

    stored = [_stored("SPEAKER_00", "Alice")]
    chunks = [_chunk("SPEAKER_09"), _chunk("SPEAKER_02"), _chunk("SPEAKER_05")]

    _, _, unmapped = carry_identity(stored, chunks, identity_version="1@1")

    assert unmapped == ["SPEAKER_02", "SPEAKER_05", "SPEAKER_09"]


# --- the phase-1 guard ------------------------------------------------------
#
# Phase 1 is NOT purely additive. `stamp_chunks` runs on the TRANSCRIBE path,
# which in phase 1 has no delete, so a re-transcribe would write
# `identity_version` onto records still carrying positionally-stale
# `speaker_id`s -- CERTIFYING them. Today such an episode is merely wrong;
# certified it is wrong AND retired from IDENTIFY_ONLY permanently, and phase
# 2 ships no rescan.


def test_an_episode_with_no_stored_records_is_not_guarded():
    from corpus.identify import has_stored_identity

    assert has_stored_identity([]) is False


def test_stored_records_with_no_names_are_not_guarded():
    # The ordinary first transcription, and the ordinary unmatched episode.
    # Guarding here would withhold the version from every new episode and
    # queue the whole corpus for audio repair.
    from corpus.identify import has_stored_identity

    assert has_stored_identity([_stored("SPEAKER_00"), _stored("SPEAKER_01")]) is False


def test_one_stored_name_is_enough_to_guard():
    # ANY surviving name means the merge can carry a positional stamp across
    # the new boundaries. One is enough, because the guard's job is to refuse
    # to certify, not to measure how wrong the episode is.
    from corpus.identify import has_stored_identity

    assert has_stored_identity([_stored("SPEAKER_00"), _stored("SPEAKER_01", "Bob")])


def test_an_empty_string_name_is_not_a_name():
    # Chroma cannot store None, so an emptied field comes back as "". Treating
    # it as a name would guard every episode a repair had ever cleared.
    from corpus.identify import has_stored_identity

    assert has_stored_identity([_stored("SPEAKER_00", "")]) is False


def test_a_withheld_version_still_writes_this_runs_correct_names():
    # The guarded TRANSCRIBE path. The names computed THIS run are correct for
    # THIS run's chunks -- it is the surviving OLD records that carry stale
    # stamps, and only the certification is unsafe. Refusing to write the
    # names too would throw away work that is right.
    from corpus.identify import stamp_chunks

    chunks = [_chunk("SPEAKER_00"), _chunk("SPEAKER_01")]

    stamped = stamp_chunks(chunks, {"SPEAKER_00": "Alice"}, identity_version=None)

    assert stamped == 1
    assert chunks[0]["metadata"]["speaker_id"] == "Alice"
    for chunk in chunks:
        assert "identity_version" not in chunk["metadata"]
