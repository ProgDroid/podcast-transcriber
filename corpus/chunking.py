"""Transcript parsing and chunking.

Lifted verbatim from transcribe.py so that chunk counts can be derived without
importing modal. `n_chunks` MUST come from re-parsing the transcript rather
than from counting stored records: counting stored records would stamp a torn
episode's truncated count as its expected count and freeze it as permanently
complete, so the completeness check would certify the exact damage it exists
to detect.
"""

from __future__ import annotations

import re

from corpus.identity import RULES_VERSION

# BGE-large handles 512 tokens; ~400 words is a safe proxy.
MAX_CHUNK_WORDS = 400
CHUNK_OVERLAP_WORDS = 50

_SEGMENT_RE = re.compile(r"\[([^\]]+)\]\s+([\d.]+)s\s+-\s+(.*)")


def parse_transcript_segments(text: str) -> list[dict]:
    """Parse the `[SPEAKER_XX] 12.3s - text` format the pipeline writes."""
    segments: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SEGMENT_RE.match(line)
        if m:
            segments.append(
                {
                    "speaker": m.group(1),
                    "start": float(m.group(2)),
                    "text": m.group(3),
                }
            )
    return segments


def build_chunks(
    segments: list[dict],
    show_name: str,
    episode_number: str,
    episode_title: str,
    date_str: str,
    *,
    episode_guid: str | None = None,
    rules_version: str = RULES_VERSION,
) -> list[dict]:
    """Group consecutive same-speaker segments into chunks.

    Splits at MAX_CHUNK_WORDS and overlaps CHUNK_OVERLAP_WORDS into the next
    chunk for context continuity.
    """
    chunks: list[dict] = []
    current_speaker: str | None = None
    current_words: list[str] = []
    current_start = 0.0

    def flush(speaker: str | None, words: list[str], start: float) -> None:
        if not words:
            return
        metadata = {
            "show": show_name,
            "episode_number": episode_number,
            "episode_title": episode_title,
            "date": date_str,
            "speaker": speaker if speaker is not None else "UNKNOWN",
            "start_time": start,
            "date_ts": (
                int(date_str.replace("-", "")) if date_str != "Unknown Date" else 0
            ),
            "rules_version": rules_version,
        }
        # Chroma rejects a None metadata value, so an absent guid means an
        # absent KEY, never a null. 6 of 438 episodes have aged off their feed
        # and can never be assigned one.
        if episode_guid is not None:
            metadata["episode_guid"] = episode_guid
        chunks.append(
            {"text": f"[{metadata['speaker']}] {' '.join(words)}", "metadata": metadata}
        )

    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        start = segment.get("start", 0.0)
        words = text.split()

        if speaker != current_speaker and current_words:
            flush(current_speaker, current_words, current_start)
            current_words = current_words[-CHUNK_OVERLAP_WORDS:]
            current_start = start

        if not current_words:
            current_start = start

        current_speaker = speaker
        current_words.extend(words)

        while len(current_words) >= MAX_CHUNK_WORDS:
            flush(current_speaker, current_words[:MAX_CHUNK_WORDS], current_start)
            current_words = current_words[-CHUNK_OVERLAP_WORDS:]
            current_start = start

    flush(current_speaker, current_words, current_start)

    # n_chunks is only knowable once the whole episode is chunked.
    for chunk in chunks:
        chunk["metadata"]["n_chunks"] = len(chunks)
    return chunks


def build_chunks_from_text(
    text: str,
    show_name: str,
    episode_number: str,
    episode_title: str,
    date_str: str,
    *,
    episode_guid: str | None = None,
    rules_version: str = RULES_VERSION,
) -> list[dict]:
    """Chunk a saved transcript file's contents."""
    return build_chunks(
        parse_transcript_segments(text),
        show_name,
        episode_number,
        episode_title,
        date_str,
        episode_guid=episode_guid,
        rules_version=rules_version,
    )


def count_chunks_from_text(
    text: str,
    show_name: str,
    episode_number: str,
    episode_title: str,
    date_str: str,
) -> int:
    """How many chunks this transcript SHOULD produce. The expected count."""
    return len(
        build_chunks_from_text(text, show_name, episode_number, episode_title, date_str)
    )
