import pytest

from corpus.identity import (
    chunk_id,
    episode_id_prefix,
    parse_transcript_filename,
    transcript_filename,
)


def test_spaces_become_underscores():
    assert (
        episode_id_prefix("Geopolitical Cousins", "1", "2025-03-14")
        == "Geopolitical_Cousins-ep1-2025-03-14"
    )


def test_prefix_is_stable_across_calls():
    a = episode_id_prefix("The Jacob Shapiro Podcast", "243", "2024-11-07")
    b = episode_id_prefix("The Jacob Shapiro Podcast", "243", "2024-11-07")
    assert a == b


@pytest.mark.parametrize(
    "show,ep,date_a,date_b",
    [
        # the duplicate episode number that silently lost an episode
        ("The Jacob Shapiro Podcast", "243", "2024-11-07", "2024-11-08"),
        # the two 'Unknown' groups
        ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29", "2026-07-31"),
        ("The Observing Japan Podcast", "Unknown", "2026-05-12", "2026-06-05"),
    ],
)
def test_same_episode_number_different_dates_do_not_collide(show, ep, date_a, date_b):
    assert episode_id_prefix(show, ep, date_a) != episode_id_prefix(show, ep, date_b)


@pytest.mark.parametrize(
    "show,date,ep_a,ep_b",
    [
        # the four (show, date) pairs that made date-only keying wrong
        ("Geopolitical Cousins", "2026-05-22", "10", "11"),
        ("The Jacob Shapiro Podcast", "2023-11-20", "150", "151"),
        ("The Jacob Shapiro Podcast", "2025-03-28", "270", "271"),
        ("The Jacob Shapiro Podcast", "2025-06-13", "300", "301"),
    ],
)
def test_same_date_different_episode_numbers_do_not_collide(show, date, ep_a, ep_b):
    assert episode_id_prefix(show, ep_a, date) != episode_id_prefix(show, ep_b, date)


def test_chunk_id_appends_index():
    prefix = episode_id_prefix("Geopolitical Cousins", "73", "2026-07-29")
    assert chunk_id(prefix, 0) == "Geopolitical_Cousins-ep73-2026-07-29-0"
    assert chunk_id(prefix, 430) == "Geopolitical_Cousins-ep73-2026-07-29-430"


def test_filename_round_trip():
    name = transcript_filename("Geopolitical Cousins", "73", "2026-07-29")
    assert name == "Geopolitical Cousins - Episode 73 - 2026-07-29.txt"
    assert parse_transcript_filename(name) == (
        "Geopolitical Cousins",
        "73",
        "2026-07-29",
    )


def test_filename_parses_unknown_episode_number():
    assert parse_transcript_filename(
        "The Observing Japan Podcast - Episode Unknown - 2026-05-12.txt"
    ) == ("The Observing Japan Podcast", "Unknown", "2026-05-12")


def test_filename_rejects_non_transcript():
    assert parse_transcript_filename("notes.md") is None
    assert parse_transcript_filename("Show - Episode 1 - 2025-13-99.txt") is None
