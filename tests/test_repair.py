"""The identity repair path: a metadata-only update over stored records.

WHY THIS IS NOT A RE-TRANSCRIBE. Re-transcribing is a full download plus a
GPU pass, whisper is non-deterministic so the chunk boundaries move, and it
would walk straight into the stale-stamp exposure the same design is trying to
close. The repair instead re-reads the SAVED transcript, identifies from
audio, and writes metadata only -- it never touches a document, so it cannot
move a boundary.

WHY IT MUST BE ABLE TO REFUSE. A metadata-only `update()` can ADD a name and
can never REMOVE one. IDENTIFY_ONLY never deletes -- only TRANSCRIBE and
RECHUNK do -- so an episode arriving here with positionally-stale stamps would
have its matched clusters corrected, its declined clusters left wrong, and the
run would report success. That is the exact failure the whole design exists to
prevent, arriving through the door built to fix it.
"""

from __future__ import annotations

from corpus.identify import plan_repair_updates

VERSION = "1@1"


def _rec(chunk_id: str, speaker: str, person: str | None = None, **extra) -> dict:
    meta = {"speaker": speaker}
    if person is not None:
        meta["speaker_id"] = person
    meta.update(extra)
    return {"id": chunk_id, "metadata": meta}


def _plan(records, identified, *, identity_version=VERSION):
    return plan_repair_updates(
        [r["id"] for r in records],
        [r["metadata"] for r in records],
        identified,
        identity_version=identity_version,
    )


def test_a_matched_cluster_gets_its_name_on_every_one_of_its_records():
    plan = _plan(
        [
            _rec("e-0", "SPEAKER_00"),
            _rec("e-1", "SPEAKER_01"),
            _rec("e-2", "SPEAKER_00"),
        ],
        {"SPEAKER_00": "Alice"},
    )

    assert plan.refusal is None
    assert plan.ids == ["e-0", "e-1", "e-2"]
    assert [m.get("speaker_id") for m in plan.metadatas] == ["Alice", None, "Alice"]


def test_the_version_lands_on_every_record_including_unmatched_clusters():
    # Uniformity again: episode_state reads ONE arbitrary record, so a record
    # left unstamped could be the one it reads, and the episode would be
    # re-queued for audio forever.
    plan = _plan(
        [_rec("e-0", "SPEAKER_00"), _rec("e-1", "SPEAKER_01")],
        {"SPEAKER_00": "Alice"},
    )

    assert [m["identity_version"] for m in plan.metadatas] == [VERSION, VERSION]


def test_nobody_matching_is_a_completed_identification_not_a_failure():
    # 27% of archive clusters match nobody. This episode must reach SKIP and
    # STAY there, or the repair queue never terminates.
    plan = _plan([_rec("e-0", "SPEAKER_00")], {})

    assert plan.refusal is None
    assert plan.metadatas == [{"identity_version": VERSION}]


def test_a_book_record_sharing_the_triple_is_not_updated():
    # upload_book.py's records can share an episode's triple by construction.
    # Stamping one would claim a book chunk was identified as an episode.
    plan = _plan(
        [_rec("e-0", "SPEAKER_00"), _rec("book-7", "SPEAKER_00", source="book")],
        {"SPEAKER_00": "Alice"},
    )

    assert plan.ids == ["e-0"]


def test_a_broken_join_writes_nothing_at_all_not_even_the_version():
    # Clusters identified and NO stored record carrying a matching `speaker`.
    # Its own copy of the predicate, because this one joins against records in
    # Chroma rather than chunk dicts on their way to a write. Stamping the
    # version here would retire from repair an episode that stored no
    # identity -- a loud nightly failure converted into a silent permanent
    # one.
    plan = _plan([_rec("e-0", "SPEAKER_07")], {"SPEAKER_00": "Alice"})

    assert plan.ids == []
    assert plan.metadatas == []
    assert plan.refusal is not None
    assert "no stored record" in plan.refusal


def test_a_stored_name_contradicting_the_fresh_one_refuses_the_whole_episode():
    # THE HOLE PART 2 WRONGLY ASSUMED WAS CLOSED. Part 2 argues a stale name
    # cannot survive into a repair "because the delete clears the record
    # first" -- but IDENTIFY_ONLY never deletes. Correcting SPEAKER_00 while
    # leaving a declined cluster's stale name in place, and reporting success,
    # is verbatim the failure the design is about.
    plan = _plan([_rec("e-0", "SPEAKER_00", "Alice")], {"SPEAKER_00": "Bob"})

    assert plan.ids == []
    assert plan.refusal is not None
    assert "SPEAKER_00" in plan.refusal


def test_a_stored_name_on_a_cluster_the_fresh_run_declined_also_refuses():
    # The declined case is the one a "correct what you can" implementation
    # silently gets wrong: there is no fresh name to overwrite it with, and a
    # metadata update cannot remove it, so proceeding leaves a name the
    # current rules would not assign and marks the episode identified.
    plan = _plan(
        [_rec("e-0", "SPEAKER_00", "Alice"), _rec("e-1", "SPEAKER_01")],
        {"SPEAKER_01": "Bob"},
    )

    assert plan.ids == []
    assert plan.refusal is not None
    assert "SPEAKER_00" in plan.refusal


def test_a_stored_name_the_fresh_run_agrees_with_is_not_a_contradiction():
    # The legitimate live population. Live-ingest identity landed after the
    # archive pass, so episodes published in between carry a CORRECT
    # speaker_id and no identity_version -- they are repaired, and the fresh
    # identification simply confirms what is already there. Refusing these
    # would stall the entire first-night drain.
    plan = _plan([_rec("e-0", "SPEAKER_00", "Alice")], {"SPEAKER_00": "Alice"})

    assert plan.refusal is None
    assert plan.ids == ["e-0"]


def test_an_empty_stored_name_is_not_a_contradiction():
    # Chroma cannot store None, so a cleared field reads back as "".
    plan = _plan([_rec("e-0", "SPEAKER_00", "")], {"SPEAKER_00": "Alice"})

    assert plan.refusal is None
    assert plan.metadatas[0]["speaker_id"] == "Alice"


def test_an_unknown_bucket_is_never_named_by_the_repair():
    # Same rule as the carry path, applied where a reader of this function
    # meets it rather than inherited from cluster_crop_plan three modules away.
    plan = _plan(
        [_rec("e-0", "UNKNOWN"), _rec("e-1", "SPEAKER_00")],
        {"UNKNOWN": "Alice", "SPEAKER_00": "Bob"},
    )

    assert plan.refusal is None
    assert "speaker_id" not in plan.metadatas[0]
    assert plan.metadatas[0]["identity_version"] == VERSION
    assert plan.metadatas[1]["speaker_id"] == "Bob"


def test_an_episode_with_no_stored_records_refuses_rather_than_reporting_done():
    # Nothing to update is not the same as nothing to do. Reporting success
    # here would mark an episode repaired that has no records to carry the
    # mark, and the next plan would queue it again -- a silent nightly loop.
    plan = _plan([], {"SPEAKER_00": "Alice"})

    assert plan.ids == []
    assert plan.refusal is not None


def test_a_refusal_names_the_clusters_so_two_runs_read_the_same():
    plan = _plan(
        [_rec("e-0", "SPEAKER_02", "Alice"), _rec("e-1", "SPEAKER_01", "Carol")],
        {"SPEAKER_02": "Bob", "SPEAKER_01": "Dave"},
    )

    assert plan.refusal is not None
    assert plan.refusal.index("SPEAKER_01") < plan.refusal.index("SPEAKER_02")
