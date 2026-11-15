"""The renumber migration's decisions, without Modal, a volume or Chroma.

2026-11-15: the JSP feed shifted every <itunes:episode> by +72 (three by +73).
The planner keys transcripts on (show, episode, date), so every archive
episode looked untranscribed. These tests pin how the migration maps a feed
entry back to the transcript and records it already has.
"""

import pytest

from corpus.renumber import plan_renumber, rekey, rewrite_header

SHOW = "The Jacob Shapiro Podcast"


def _feed(*rows):
    return [{"episode_number": n, "date": d, "guid": g} for n, d, g in rows]


def _plan(feed, volume, stored):
    """`stored` maps (episode_number, date) -> guid or None."""
    stored_by_guid: dict[str, set[tuple[str, str]]] = {}
    for triple, guid in stored.items():
        if guid:
            stored_by_guid.setdefault(guid, set()).add(triple)
    return plan_renumber(
        SHOW,
        feed,
        volume_triples={(SHOW, n, d) for n, d in volume},
        stored_triples={(SHOW, n, d) for n, d in stored},
        stored_by_guid=stored_by_guid,
    )


def test_a_renumbered_episode_is_renamed_from_its_old_number():
    plan = _plan(
        _feed(("389", "2025-12-22", "g1")),
        volume=[("317", "2025-12-22")],
        stored={("317", "2025-12-22"): "g1"},
    )
    assert [(r.old_number, r.new_number, r.date, r.guid) for r in plan.renames] == [
        ("317", "389", "2025-12-22", "g1")
    ]
    assert plan.unresolved == []


def test_the_match_is_by_guid_not_by_the_shift():
    # Three episodes moved by +73, not +72. A shift-based match would miss
    # them; the guid does not care.
    plan = _plan(
        _feed(("230", "2024-01-01", "g1")),
        volume=[("157", "2024-01-01")],
        stored={("157", "2024-01-01"): "g1"},
    )
    assert [(r.old_number, r.new_number) for r in plan.renames] == [("157", "230")]


def test_same_day_episodes_each_follow_their_own_guid():
    plan = _plan(
        _feed(("228", "2023-11-20", "ga"), ("229", "2023-11-20", "gb")),
        volume=[("156", "2023-11-20"), ("157", "2023-11-20")],
        stored={("156", "2023-11-20"): "ga", ("157", "2023-11-20"): "gb"},
    )
    assert sorted((r.old_number, r.new_number) for r in plan.renames) == [
        ("156", "228"),
        ("157", "229"),
    ]


def test_an_episode_already_on_the_volume_under_its_feed_number_is_left_alone():
    plan = _plan(
        _feed(("389", "2025-12-22", "g1")),
        volume=[("389", "2025-12-22")],
        stored={("389", "2025-12-22"): "g1"},
    )
    assert plan.renames == []
    assert plan.unresolved == []


def test_a_genuinely_new_episode_is_counted_new_not_renamed():
    plan = _plan(_feed(("437", "2026-11-16", "gnew")), volume=[], stored={})
    assert plan.renames == []
    assert plan.new == 1
    assert plan.unresolved == []


def test_an_excluded_episode_is_never_renamed():
    # 271 is a real exclusion (corpus/exclusions.py), matched by guid there.
    plan = _plan(
        _feed(("343", "2025-04-04", "c4af95bf-cfbc-4c0a-b4d7-2c2df77d1fe6")),
        volume=[("271", "2025-04-04")],
        stored={},
    )
    assert plan.renames == []
    assert plan.unresolved == []
    assert plan.orphans == []


def test_a_guid_stored_under_two_old_triples_is_unresolved():
    plan = _plan(
        _feed(("389", "2025-12-22", "g1")),
        volume=[("317", "2025-12-22"), ("318", "2025-12-22")],
        stored={("317", "2025-12-22"): "g1", ("318", "2025-12-22"): "g1"},
    )
    assert plan.renames == []
    assert len(plan.unresolved) == 1
    assert "389" in plan.unresolved[0]


def test_a_guid_whose_date_moved_is_unresolved():
    plan = _plan(
        _feed(("389", "2025-12-23", "g1")),
        volume=[("317", "2025-12-22")],
        stored={("317", "2025-12-22"): "g1"},
    )
    assert plan.renames == []
    assert len(plan.unresolved) == 1


def test_an_old_triple_with_no_transcript_file_is_unresolved():
    plan = _plan(
        _feed(("389", "2025-12-22", "g1")),
        volume=[],
        stored={("317", "2025-12-22"): "g1"},
    )
    assert plan.renames == []
    assert len(plan.unresolved) == 1


def test_an_old_triple_still_in_the_feed_is_unresolved():
    # Renaming it would take the transcript away from the entry that owns it.
    plan = _plan(
        _feed(("389", "2025-12-22", "g1"), ("317", "2025-12-22", "g2")),
        volume=[("317", "2025-12-22")],
        stored={("317", "2025-12-22"): "g1"},
    )
    assert plan.renames == []
    assert len(plan.unresolved) == 1


def test_a_feed_entry_with_no_guid_is_unresolved_when_it_has_no_file():
    plan = _plan(_feed(("389", "2025-12-22", None)), volume=[], stored={})
    assert plan.renames == []
    assert len(plan.unresolved) == 1


def test_a_file_in_neither_feed_nor_store_is_an_orphan():
    # The old copy of an episode already re-transcribed under its new number:
    # the guid prune removed its records, and bulk_embed would re-add them.
    plan = _plan(
        _feed(("389", "2025-12-22", "g1")),
        volume=[("389", "2025-12-22"), ("317", "2025-12-22")],
        stored={("389", "2025-12-22"): "g1"},
    )
    assert plan.orphans == [(SHOW, "317", "2025-12-22")]


def test_an_aged_off_episode_still_in_the_store_is_not_an_orphan():
    plan = _plan(
        _feed(), volume=[("7", "2022-05-27")], stored={("7", "2022-05-27"): None}
    )
    assert plan.orphans == []


def test_an_old_triple_being_renamed_is_not_an_orphan():
    plan = _plan(
        _feed(("389", "2025-12-22", "g1")),
        volume=[("317", "2025-12-22")],
        stored={("317", "2025-12-22"): "g1"},
    )
    assert plan.orphans == []


# --- rekey -------------------------------------------------------------------


def _old_records():
    prefix = "The_Jacob_Shapiro_Podcast-ep317-2025-12-22"
    ids = [f"{prefix}-1", f"{prefix}-0"]
    metas = [
        {"show": SHOW, "episode_number": "317", "date": "2025-12-22", "speaker": "A"},
        {"show": SHOW, "episode_number": "317", "date": "2025-12-22", "speaker": "B"},
    ]
    return ids, metas


def test_rekey_moves_each_chunk_to_the_same_index_under_the_new_number():
    ids, metas = _old_records()
    new_ids, new_metas = rekey(
        ids, metas, show=SHOW, old_number="317", new_number="389", date="2025-12-22"
    )
    assert new_ids == [
        "The_Jacob_Shapiro_Podcast-ep389-2025-12-22-1",
        "The_Jacob_Shapiro_Podcast-ep389-2025-12-22-0",
    ]
    assert [m["episode_number"] for m in new_metas] == ["389", "389"]
    assert [m["speaker"] for m in new_metas] == ["A", "B"]


def test_rekey_does_not_mutate_the_records_it_was_given():
    ids, metas = _old_records()
    rekey(ids, metas, show=SHOW, old_number="317", new_number="389", date="2025-12-22")
    assert metas[0]["episode_number"] == "317"


def test_rekey_refuses_an_id_outside_the_old_prefix():
    ids, metas = _old_records()
    ids[0] = "The_Jacob_Shapiro_Podcast-ep318-2025-12-22-1"
    with pytest.raises(ValueError):
        rekey(
            ids, metas, show=SHOW, old_number="317", new_number="389", date="2025-12-22"
        )


def test_rekey_refuses_a_record_whose_metadata_names_another_episode():
    ids, metas = _old_records()
    metas[1]["episode_number"] = "318"
    with pytest.raises(ValueError):
        rekey(
            ids, metas, show=SHOW, old_number="317", new_number="389", date="2025-12-22"
        )


# --- rewrite_header ----------------------------------------------------------


def test_rewrite_header_changes_only_the_first_line():
    text = f"# {SHOW} - Episode 317\n# Right Wing Ascendant\n# Published: 2025-12-22\n"
    out = rewrite_header(text, SHOW, "317", "389")
    assert out.splitlines()[0] == f"# {SHOW} - Episode 389"
    assert out.splitlines()[1:] == text.splitlines()[1:]


def test_rewrite_header_leaves_an_unexpected_header_untouched():
    text = "# something else\n[SPEAKER_00] 1.0s - hi\n"
    assert rewrite_header(text, SHOW, "317", "389") == text
