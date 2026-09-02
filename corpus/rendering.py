"""Rendering a search result into the string a consumer actually reads.

WHY THIS IS NOT IN mcp_server.py. The searcher put `speaker` into its result
dict and both tools then rendered a string that never mentioned it, so no
caller ever saw an attribution -- while `search_podcasts`'s own docstring and
the README both advertised one. The dict was right; the render was the layer
that mattered, and it had no test because it was nested inside the Modal ASGI
factory and could not be imported.

Living here rather than in the Modal file makes the render reachable from the
test suite, which is the same argument corpus/showplan.py makes for planning.
"""

from __future__ import annotations

# chunking.py substitutes this when a segment carries no speaker, so a record
# with no `speaker` key at all and one written from an unlabelled segment
# render identically. That is deliberate: both mean "we do not know".
UNKNOWN_SPEAKER = "UNKNOWN"


def _strip_speaker_prefix(text: str, speaker: str) -> str:
    """Drop the leading `[speaker] ` that chunking.py bakes into the document.

    Only when it MATCHES the metadata. A prefix that disagrees is left in
    place: the disagreement is information, and hiding it would make a
    metadata-only rename that failed to update the document text look clean.
    """
    prefix = f"[{speaker}] "
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def format_result(result: dict, *, include_relevance: bool = True) -> str:
    """One search hit, as the caller reads it.

    `include_relevance` is False for latest_on_topic, which orders by recency
    and so has no score worth showing.
    """
    speaker = result.get("speaker") or UNKNOWN_SPEAKER
    text = _strip_speaker_prefix(result.get("text", ""), speaker)

    header = (
        f"[{result['date']}] {result['show']} - "
        f"Episode {result['episode_number']} ({result['episode_title']}) "
        f"@ {result['start_time']:.1f}s | speaker: {speaker}"
    )
    if include_relevance:
        header += f" [relevance: {result['relevance_score']}]"

    return f"{header}\n{text}\n"


def format_results(results: list[dict], *, include_relevance: bool = True) -> str:
    """The whole response body."""
    return "\n---\n".join(
        format_result(r, include_relevance=include_relevance) for r in results
    )
