import pytest

from corpus.exclusions import EXCLUDED_EPISODES, is_excluded
from corpus.planning import Action, decide_action


def _decide(**overrides):
    kwargs = {
        "transcript_exists": True,
        "complete_in_chroma": True,
        "stored_rules_version": "1",
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


def test_exclusion_beats_unparseable():
    assert _decide(excluded=True, parses_to_chunks=False) is Action.EXCLUDE


@pytest.mark.parametrize("episode", sorted(EXCLUDED_EPISODES))
def test_every_excluded_episode_is_recognised(episode):
    assert is_excluded(*episode)


def test_the_geopolitical_cousins_originals_are_not_excluded():
    # Only the Jacob Shapiro re-posts are excluded; the GC originals stay.
    assert not is_excluded("Geopolitical Cousins", "73", "2026-07-29")
    assert not is_excluded("Geopolitical Cousins", "74", "2026-07-31")


def test_exclusions_are_exactly_the_two_cross_posts():
    assert EXCLUDED_EPISODES == frozenset(
        {
            ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29"),
            ("The Jacob Shapiro Podcast", "Unknown", "2026-07-31"),
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
