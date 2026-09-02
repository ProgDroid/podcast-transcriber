"""The MCP render path.

`speaker` was carried in the result dict and never rendered, so no consumer
ever saw an attribution -- while the README advertised one. These pin the
field into the string a caller actually reads.
"""

from corpus.rendering import format_result, format_results


def _result(**overrides):
    r = {
        "text": "[SPEAKER_00] the thing about Taiwan is that it is an island.",
        "show": "Geopolitical Cousins",
        "episode_number": "73",
        "episode_title": "Riding the Hog",
        "date": "2026-07-29",
        "speaker": "SPEAKER_00",
        "start_time": 412.0,
        "n_chunks": 431,
        "episode_guid": "abc",
        "relevance_score": 0.82,
    }
    r.update(overrides)
    return r


def test_speaker_appears_in_the_rendered_result():
    """The bug. The dict carried it; the string handed to the caller did not."""
    assert "SPEAKER_00" in format_result(_result())


def test_named_speaker_appears_in_the_rendered_result():
    """A metadata-only rename must change what the consumer sees."""
    out = format_result(
        _result(speaker="Jacob Shapiro", text="[Jacob Shapiro] Taiwan is an island.")
    )
    assert "Jacob Shapiro" in out


def test_matching_speaker_prefix_is_stripped_from_the_text():
    """Rendered separately, so leaving it inline says it twice."""
    out = format_result(_result())
    assert "[SPEAKER_00]" not in out
    assert "the thing about Taiwan is that it is an island." in out


def test_text_without_a_prefix_is_left_alone():
    out = format_result(_result(text="no prefix here at all"))
    assert "no prefix here at all" in out


def test_a_prefix_that_disagrees_with_the_metadata_is_not_stripped():
    """A disagreement is information. Hiding it would make a desynced
    metadata-only rename look clean."""
    out = format_result(_result(speaker="SPEAKER_01"))
    assert "[SPEAKER_00]" in out


def test_a_missing_speaker_renders_unknown_and_does_not_raise():
    """Pre-migration records carry no speaker key at all."""
    r = _result()
    del r["speaker"]
    assert "UNKNOWN" in format_result(r)


def test_relevance_is_omitted_when_not_requested():
    """latest_on_topic renders no relevance score."""
    out = format_result(_result(), include_relevance=False)
    assert "relevance" not in out
    assert "SPEAKER_00" in out


def test_relevance_is_included_by_default():
    assert "relevance: 0.82" in format_result(_result())


def test_the_existing_header_fields_survive():
    out = format_result(_result())
    for expected in ("2026-07-29", "Geopolitical Cousins", "73", "Riding the Hog"):
        assert expected in out
    assert "412.0s" in out


def test_format_results_joins_every_result():
    out = format_results([_result(), _result(speaker="SPEAKER_01")])
    assert out.count("Geopolitical Cousins") == 2
    assert "SPEAKER_01" in out
