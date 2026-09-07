import pytest

from corpus.exclusions import EXCLUDED_EPISODES, is_excluded
from corpus.identity import IDENTITY_VERSION
from corpus.planning import Action, cap_work, decide_action


def _decide(**overrides):
    kwargs = {
        "transcript_exists": True,
        "complete_in_chroma": True,
        "stored_rules_version": "1",
        "stored_identity_version": IDENTITY_VERSION,
        "excluded": False,
        "parses_to_chunks": True,
    }
    kwargs.update(overrides)
    return decide_action(**kwargs)


def test_healthy_episode_is_skipped():
    assert _decide() is Action.SKIP


def test_missing_transcript_is_transcribed():
    assert _decide(transcript_exists=False) is Action.TRANSCRIBE


def test_transcript_present_but_incomplete_is_embed_only():
    assert _decide(complete_in_chroma=False) is Action.EMBED_ONLY


def test_stale_rules_version_forces_re_embed():
    # Completeness alone returns SKIP for every episode after migration, so
    # without this a rules-change re-embed is a silent no-op.
    assert _decide(stored_rules_version="0") is Action.EMBED_ONLY


def test_absent_rules_version_forces_re_embed():
    assert _decide(stored_rules_version=None) is Action.EMBED_ONLY


def test_exclusion_beats_embed_only():
    # Both excluded episodes have transcripts on the volume, so without this
    # the cron re-embeds them nightly and reverts the approved deletion.
    assert _decide(excluded=True, complete_in_chroma=False) is Action.EXCLUDE


def test_exclusion_beats_transcribe():
    assert _decide(excluded=True, transcript_exists=False) is Action.EXCLUDE


def test_unparseable_transcript_is_terminal_not_a_re_embed_loop():
    assert (
        _decide(parses_to_chunks=False, complete_in_chroma=False) is Action.UNPARSEABLE
    )


def test_missing_transcript_is_transcribed_even_if_it_would_not_parse():
    # UNPARSEABLE is terminal and means "a transcript exists on disk that will
    # never produce chunks". With no transcript at all the answer is TRANSCRIBE
    # -- go fetch it. Ordering these the other way round would permanently
    # foreclose transcription for an episode that merely needs transcribing.
    assert _decide(transcript_exists=False, parses_to_chunks=False) is Action.TRANSCRIBE


def test_exclusion_beats_unparseable():
    assert _decide(excluded=True, parses_to_chunks=False) is Action.EXCLUDE


@pytest.mark.parametrize("episode", sorted(EXCLUDED_EPISODES))
def test_every_excluded_episode_is_recognised(episode):
    assert is_excluded(*episode)


def test_the_geopolitical_cousins_originals_are_not_excluded():
    # Only the Jacob Shapiro re-posts are excluded; the GC originals stay.
    assert not is_excluded("Geopolitical Cousins", "73", "2026-07-29")
    assert not is_excluded("Geopolitical Cousins", "74", "2026-07-31")


def test_exclusions_are_exactly_the_four_cross_posts():
    # The 2026 pair fall back to "Unknown" because Captivate publishes no
    # itunes_episode for them; the 2025 pair carry real numbers because it
    # does. Both shapes have to survive in one list, so the literal set is
    # asserted rather than a count -- a count would not notice a triple
    # silently changing shape.
    assert EXCLUDED_EPISODES == frozenset(
        {
            ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29"),
            ("The Jacob Shapiro Podcast", "Unknown", "2026-07-31"),
            ("The Jacob Shapiro Podcast", "271", "2025-04-04"),
            ("The Jacob Shapiro Podcast", "273", "2025-04-08"),
        }
    )


def test_derived_views_cannot_drift_from_the_record_list():
    from corpus.exclusions import EXCLUDED, EXCLUDED_GUIDS

    assert EXCLUDED_EPISODES == frozenset(e.triple for e in EXCLUDED)
    assert EXCLUDED_GUIDS == frozenset(e.guid for e in EXCLUDED if e.guid)
    assert all(e.reason for e in EXCLUDED)


def test_guid_arm_survives_an_episode_number_backfill():
    # Both excluded episodes fall back to "Unknown" because Captivate
    # publishes no itunes_episode for them. If it ever backfills one, the
    # triple changes and the triple arm silently stops matching.
    assert is_excluded(
        "The Jacob Shapiro Podcast",
        "352",  # a backfilled number -- the triple no longer matches
        "2026-07-29",
        episode_guid="1c45dbd9-0dc3-4d07-b2d1-758fe78405fe",
    )


def test_guid_arm_does_not_over_match():
    assert not is_excluded(
        "Geopolitical Cousins", "73", "2026-07-29", episode_guid="some-other-guid"
    )


def test_the_guid_arm_never_excludes_the_original_it_was_cross_posted_from():
    # THE BUG THIS PINS, found by reading a real repair run's plan output.
    # Every excluded episode is a CROSS-POST, and a cross-post syndicated
    # through Captivate carries THE SAME GUID as the episode it copies. An
    # unscoped guid arm therefore matched Geopolitical Cousins 73 itself --
    # the 431-chunk episode this whole project exists to repair -- and the
    # repair run planned EXCLUDE for it and for 74.
    #
    # test_guid_arm_does_not_over_match cannot catch this: it passes a guid
    # that is not in EXCLUDED_GUIDS at all, so it never exercises the arm.
    shared_guid = "1c45dbd9-0dc3-4d07-b2d1-758fe78405fe"
    assert not is_excluded(
        "Geopolitical Cousins", "73", "2026-07-29", episode_guid=shared_guid
    )
    # ...while the cross-post itself is still excluded by that same guid,
    # so the arm is scoped, not disabled.
    assert is_excluded(
        "The Jacob Shapiro Podcast", "Unknown", "2026-07-29", episode_guid=shared_guid
    )


# --- IDENTIFY_ONLY ----------------------------------------------------------


def test_stale_identity_version_queues_an_identity_repair():
    # The signal records that identification RAN, not that it MATCHED. 27% of
    # archive clusters legitimately match nobody, so a planner keyed on
    # `speaker_id` absence would re-download their audio every night forever.
    assert _decide(stored_identity_version="0@1") is Action.IDENTIFY_ONLY


def test_absent_identity_version_queues_an_identity_repair():
    # The state left by an EMBED_ONLY re-embed, an enrolment failure, and an
    # identify_clusters exception -- the three producers that never recover
    # today because decide_action never branches on identity.
    assert _decide(stored_identity_version=None) is Action.IDENTIFY_ONLY


def test_identity_is_not_repaired_while_the_episode_is_torn():
    # Identity is derived from chunks, so chunks settle first. Repairing
    # identity over a torn episode stamps a chunk set that is about to be
    # rewritten.
    assert (
        _decide(complete_in_chroma=False, stored_identity_version=None)
        is Action.EMBED_ONLY
    )


def test_identity_is_not_repaired_while_the_rules_version_is_stale():
    assert (
        _decide(stored_rules_version="0", stored_identity_version=None)
        is Action.EMBED_ONLY
    )


def test_skip_requires_identity_current_as_well_as_complete_and_current():
    assert _decide() is Action.SKIP


def test_exclusion_beats_identify_only():
    assert _decide(excluded=True, stored_identity_version=None) is Action.EXCLUDE


def test_a_missing_transcript_is_transcribed_not_identity_repaired():
    assert (
        _decide(transcript_exists=False, stored_identity_version=None)
        is Action.TRANSCRIBE
    )


def test_an_unparseable_transcript_is_not_identity_repaired():
    assert (
        _decide(parses_to_chunks=False, stored_identity_version=None)
        is Action.UNPARSEABLE
    )


# --- cap_work ---------------------------------------------------------------


def _eps(n):
    return [{"episode_number": str(i)} for i in range(n)]


def test_cap_work_under_the_cap_acts_on_everything_and_leaves_none():
    todo, left = cap_work(_eps(3), 25)
    assert [e["episode_number"] for e in todo] == ["0", "1", "2"]
    assert left == 0


def test_cap_work_at_the_cap_acts_on_everything_and_leaves_none():
    todo, left = cap_work(_eps(25), 25)
    assert len(todo) == 25
    assert left == 0


def test_cap_work_over_the_cap_acts_on_the_cap_and_reports_the_remainder():
    todo, left = cap_work(_eps(30), 25)
    assert len(todo) == 25
    assert left == 5


def test_cap_work_preserves_the_feeds_ordering():
    # The caps drain a backlog nightly. Re-ordering here would let an episode
    # at the tail starve while the head is repaired repeatedly.
    todo, _ = cap_work(_eps(30), 3)
    assert [e["episode_number"] for e in todo] == ["0", "1", "2"]


def test_cap_work_at_zero_acts_on_nothing_and_reports_the_whole_backlog():
    # Part 5's landing order: ship the code with the caps at 0 so the planner
    # classifies and reports against production for one night with no
    # bandwidth spent. A cap of 0 that acted on everything would spend it.
    todo, left = cap_work(_eps(30), 0)
    assert todo == []
    assert left == 30


def test_cap_work_does_not_alias_the_caller_list():
    episodes = _eps(30)
    todo, _ = cap_work(episodes, 25)
    todo.append({"episode_number": "injected"})
    assert len(episodes) == 30
