from corpus.chunking import (
    build_chunks,
    build_chunks_from_text,
    count_chunks_from_text,
    parse_transcript_segments,
)

TRANSCRIPT = """# Geopolitical Cousins - Episode 1
# F*cking Around and Finding Out
# Published: 2025-03-14

[SPEAKER_00] 3.1s - Hello listeners, welcome to the inaugural episode.
[SPEAKER_00] 6.9s - I assume that Marco and I will do our own separate intro.
[SPEAKER_01] 52.8s - All right, the two cousins officially doing business.
[UNKNOWN] 61.0s - A stray segment.
this line is malformed and must be skipped
"""


def test_parses_segments_and_skips_headers_and_junk():
    segs = parse_transcript_segments(TRANSCRIPT)
    assert len(segs) == 4
    assert segs[0] == {
        "speaker": "SPEAKER_00",
        "start": 3.1,
        "text": "Hello listeners, welcome to the inaugural episode.",
    }
    assert segs[3]["speaker"] == "UNKNOWN"


def test_chunks_split_on_speaker_change():
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "F*cking Around", "2025-03-14"
    )
    speakers = [c["metadata"]["speaker"] for c in chunks]
    assert speakers == ["SPEAKER_00", "SPEAKER_01", "UNKNOWN"]


def test_every_chunk_carries_n_chunks_equal_to_the_total():
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "F*cking Around", "2025-03-14"
    )
    assert all(c["metadata"]["n_chunks"] == len(chunks) for c in chunks)


def test_metadata_carries_identity_and_versions():
    chunks = build_chunks_from_text(
        TRANSCRIPT,
        "Geopolitical Cousins",
        "1",
        "F*cking Around",
        "2025-03-14",
        episode_guid="b4a9c88b-9dbf-46b7-9dc1-a7812a9bde65",
    )
    meta = chunks[0]["metadata"]
    assert meta["show"] == "Geopolitical Cousins"
    assert meta["episode_number"] == "1"
    assert meta["date"] == "2025-03-14"
    assert meta["date_ts"] == 20250314
    assert meta["episode_guid"] == "b4a9c88b-9dbf-46b7-9dc1-a7812a9bde65"
    assert meta["rules_version"] == "1"


def test_guid_is_omitted_rather_than_null_when_absent():
    # Chroma rejects a None metadata value; the key must simply not be there.
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "t", "2025-03-14"
    )
    assert "episode_guid" not in chunks[0]["metadata"]


def test_document_text_embeds_the_speaker_label():
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "t", "2025-03-14"
    )
    assert chunks[0]["text"].startswith("[SPEAKER_00] ")


def test_long_single_speaker_run_splits_at_max_words():
    segments = [{"speaker": "SPEAKER_00", "start": 0.0, "text": "word " * 1000}]
    chunks = build_chunks(segments, "Show", "1", "t", "2025-01-01")
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 401 for c in chunks)


def test_count_matches_build():
    n = count_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "F*cking Around", "2025-03-14"
    )
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "F*cking Around", "2025-03-14"
    )
    assert n == len(chunks)


def test_empty_transcript_yields_no_chunks():
    assert build_chunks_from_text("", "Show", "1", "t", "2025-01-01") == []
    assert count_chunks_from_text("", "Show", "1", "t", "2025-01-01") == 0


def test_unknown_date_gets_zero_timestamp():
    chunks = build_chunks_from_text(TRANSCRIPT, "Show", "1", "t", "Unknown Date")
    assert chunks[0]["metadata"]["date_ts"] == 0
