"""Load a directory of transcripts into episode records.

Shared by `speaker_stats.py` and `speaker_tool.py`. It lives here rather than
in either of them because the moment two callers derive turns separately is
the moment they can disagree about them, and this whole line of work exists
because a figure derived by an unrecorded method could not be checked.

Reading files is I/O, but it is local, offline and deterministic, so this
still runs in the test suite without a network or a GPU.
"""

from __future__ import annotations

from pathlib import Path

from corpus.chunking import parse_transcript_segments
from corpus.identity import parse_transcript_filename
from corpus.speakers import MIN_TURN_S, count_non_monotonic, merge_turns, speech_shares


def show_of(filename: str) -> str:
    return filename.split(" - Episode ")[0]


def load_episodes(directory: Path, *, cap: float) -> list[dict]:
    """Parse every transcript in `directory` into an episode record.

    `cap` is `merge_turns`' gap ceiling and is required for the same reason it
    is required there: two runs are only comparable if they agree on it.
    """
    episodes: list[dict] = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        segments = parse_transcript_segments(text)
        if not segments:
            continue
        turns = merge_turns(segments, gap_cap_s=cap)
        shares = speech_shares(turns, min_turn_s=MIN_TURN_S)
        parsed = parse_transcript_filename(path.name)
        episodes.append(
            {
                "show": show_of(path.name),
                "name": path.name,
                "path": path,
                "text": text,
                "date": parsed[2] if parsed else "",
                "year": parsed[2][:4] if parsed else "unparsed",
                "segments": segments,
                "n_segments": len(segments),
                "turns": turns,
                "shares": shares,
                "non_monotonic": count_non_monotonic(segments),
            }
        )
    return episodes


def dominant_speaker(episode: dict) -> str | None:
    """The cluster holding the most speech time, or None if there is none.

    This reports which cluster is dominant. It does NOT assert that cluster is
    the host -- that assertion is Tier 1's hypothesis and the thing the eval
    exists to test, so it stays out of this module.
    """
    by_speaker = episode["shares"]["by_speaker_s"]
    if not by_speaker:
        return None
    return max(by_speaker, key=lambda k: by_speaker[k])
