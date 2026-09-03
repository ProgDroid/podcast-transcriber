"""The MCP render path.

`speaker` was carried in the result dict and never rendered, so no consumer
ever saw an attribution -- while the README advertised one. These pin the
field into the string a caller actually reads.

Speaker identity added a second field, `speaker_id`, resolved through a names
map into four render states (see `display_speaker`'s docstring in
corpus/rendering.py). Three tests below were written before `speaker_id`
existed and pinned the pre-identity behaviour, where the raw `speaker` field
was read as if it were already a display name. They are updated in place to
the four-state contract rather than preserved, since the contract now
supersedes what they pinned.
"""

from corpus.rendering import display_speaker, format_result, format_results


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
    """Updated: with no speaker_id, an unresolved cluster label rides along
    as a parenthetical rather than appearing bare -- a bare SPEAKER_00 is a
    token an LLM consumer would invent meaning for."""
    out = format_result(_result())
    assert "unidentified voice (SPEAKER_00)" in out


def test_named_speaker_appears_in_the_rendered_result():
    """Updated: a name is now resolved through speaker_id + the names map,
    not read directly off the `speaker` cluster-label field. A metadata-only
    rename of the NAMES entry must change what the consumer sees."""
    out = format_result(
        _result(speaker_id="Jacob Shapiro"),
        names={"Jacob Shapiro": "Jacob Shapiro"},
    )
    assert "speaker: Jacob Shapiro" in out


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


def test_a_missing_speaker_renders_unidentified_and_does_not_raise():
    """Updated: pre-migration records carry no speaker key at all, and with
    no speaker_id either, that is the fourth state -- a bare `unidentified`,
    not the raw UNKNOWN cluster token."""
    r = _result()
    del r["speaker"]
    out = format_result(r)
    assert "speaker: unidentified" in out
    assert "UNKNOWN" not in out


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


# --- The four-state contract -------------------------------------------

BASE = {
    "text": "[SPEAKER_01] the thing he said",
    "speaker": "SPEAKER_01",
    "date": "2026-01-01",
    "show": "The Jacob Shapiro Podcast",
    "episode_number": "300",
    "episode_title": "A Title",
    "start_time": 12.0,
    "relevance_score": 0.9,
}
NAMES = {"Rob Larity": "Rob Larity"}


def test_named_identity_renders_the_name():
    result = {**BASE, "speaker_id": "Rob Larity"}
    assert display_speaker(result, NAMES) == "Rob Larity"


def test_unnamed_group_never_renders_a_bare_token():
    # An opaque grp_007 alone is a token an LLM consumer will invent meaning
    # for. It appears only as a parenthetical after words that say what it is.
    result = {**BASE, "speaker_id": "grp_007"}
    rendered = display_speaker(result, {})
    assert "grp_007" in rendered
    assert rendered != "grp_007"
    assert "unidentified voice" in rendered


def test_absent_speaker_id_with_a_real_cluster_label_carries_the_label():
    # speaker_id absent but `speaker` is a real cluster label: `speaker` is
    # the only within-episode discriminator once identity has not resolved,
    # so it rides along rather than collapsing to a bare "unidentified".
    result = {**BASE, "speaker": "SPEAKER_01"}
    assert "speaker_id" not in result
    assert display_speaker(result, NAMES) == "unidentified voice (SPEAKER_01)"


def test_absent_speaker_id_and_missing_speaker_renders_unidentified():
    result = {k: v for k, v in BASE.items() if k != "speaker"}
    assert display_speaker(result, NAMES) == "unidentified"


def test_unknown_speaker_renders_unidentified():
    result = {**BASE, "speaker": "UNKNOWN", "text": "[UNKNOWN] words"}
    assert display_speaker(result, NAMES) == "unidentified"


def test_prefix_is_stripped_with_the_cluster_label_not_the_display_name():
    # THE trap: chunking.py bakes [SPEAKER_01] into the document and the strip
    # must key off the raw cluster label. Stripping with "Rob Larity" would
    # leave the cluster label visible in the text it was meant to replace.
    result = {**BASE, "speaker_id": "Rob Larity"}
    rendered = format_result(result, names=NAMES)
    assert "[SPEAKER_01]" not in rendered
    assert "speaker: Rob Larity" in rendered
    assert "the thing he said" in rendered


def test_rendering_is_unchanged_when_no_speaker_id_exists_anywhere():
    # Every record in Chroma is in this state until the backfill runs, so this
    # is the compatibility test: the change must be inert before it is useful.
    assert format_result(BASE) == format_result(BASE, names={})
