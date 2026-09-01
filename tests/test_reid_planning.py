from corpus.reid_planning import build_episode_facts, enrich_metadata

TRANSCRIPT = """# Show - Episode 1
# Title
# Published: 2025-01-01

[SPEAKER_00] 1.0s - Hello there.
[SPEAKER_01] 5.0s - And hello to you.
"""

KEY = ("Show", "1", "2025-01-01")


def test_facts_carry_expected_chunk_count_and_guid():
    facts = build_episode_facts({KEY: TRANSCRIPT}, {KEY: "guid-abc"})
    assert facts[KEY]["n_chunks"] == 2
    assert facts[KEY]["episode_guid"] == "guid-abc"


def test_absent_guid_does_not_affect_the_chunk_count():
    # build_episode_facts has no stored-records input at all -- the
    # transcript-not-stored-records property that matters is enforced one
    # layer down, in count_chunks_from_text (which only ever sees the
    # transcript text) and in migration/reid.py's run() (which reads
    # transcripts fresh off the volume rather than counting anything in
    # Chroma). This test only pins that n_chunks is unaffected by whether a
    # guid was found.
    facts = build_episode_facts({KEY: TRANSCRIPT}, {})
    assert facts[KEY]["n_chunks"] == 2


def test_missing_guid_is_omitted_not_null():
    facts = build_episode_facts({KEY: TRANSCRIPT}, {})
    assert "episode_guid" not in facts[KEY]


def test_enrich_adds_the_declared_keys_and_preserves_the_rest():
    meta = {
        "show": "Show",
        "episode_number": "1",
        "date": "2025-01-01",
        "speaker": "SPEAKER_00",
    }
    facts = build_episode_facts({KEY: TRANSCRIPT}, {KEY: "guid-abc"})
    out = enrich_metadata(meta, facts[KEY])
    assert out["speaker"] == "SPEAKER_00"
    assert out["n_chunks"] == 2
    assert out["episode_guid"] == "guid-abc"
    assert out["rules_version"] == "1"


def test_enrich_without_facts_leaves_metadata_untouched():
    # Book records and any episode with no transcript on the volume.
    meta = {"show": "Geopolitical Alpha", "episode_number": "N/A"}
    assert enrich_metadata(meta, None) == meta
