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

# What a consumer reads when identity has not resolved to a name. Used alone
# only when `speaker` itself discriminates nothing (UNKNOWN or missing) --
# otherwise the cluster label rides along as a parenthetical, since `speaker`
# is the only within-episode discriminator once identity is unresolved and
# collapsing it away would make two different voices in one episode
# indistinguishable.
UNIDENTIFIED = "unidentified"


def _strip_speaker_prefix(text: str, speaker: str) -> str:
    """Drop the leading `[speaker] ` that chunking.py bakes into the document.

    Only when it MATCHES the metadata. A prefix that disagrees is left in
    place: the disagreement is information, and hiding it would make a
    metadata-only rename that failed to update the document text look clean.

    Always keyed off the RAW CLUSTER LABEL (`speaker`), never a resolved
    display name. `chunking.py:84` bakes `[SPEAKER_01] ` into the document
    text, so keying the strip off a resolved name would leave the cluster
    label visible in the very text the name was meant to replace.
    """
    prefix = f"[{speaker}] "
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def display_speaker(result: dict, names: dict[str, str] | None) -> str:
    """The identity a caller reads, across four states:

    - speaker_id present, resolved in names -> the resolved name.
    - speaker_id present, not in names -> "unidentified voice (grp_007)".
    - speaker_id absent, speaker is a real cluster label ->
      "unidentified voice (SPEAKER_01)".
    - speaker_id absent, speaker is UNKNOWN or missing -> "unidentified".

    Never returns a bare group id or cluster label alone. An opaque token
    like `grp_007` or `SPEAKER_01` is one an LLM consumer will invent meaning
    for, so it appears only as a parenthetical after words that say what it
    is.

    The last two states render differently on purpose. `speaker` is the only
    within-episode discriminator once identity has not resolved to a name, so
    carrying it forward keeps two different voices in one episode
    distinguishable even while neither is named.
    """
    speaker_id = result.get("speaker_id")
    if speaker_id:
        resolved = (names or {}).get(speaker_id)
        if resolved:
            return resolved
        return f"{UNIDENTIFIED} voice ({speaker_id})"

    cluster = result.get("speaker")
    if cluster and cluster != UNKNOWN_SPEAKER:
        return f"{UNIDENTIFIED} voice ({cluster})"
    return UNIDENTIFIED


def format_result(
    result: dict,
    *,
    include_relevance: bool = True,
    names: dict[str, str] | None = None,
) -> str:
    """One search hit, as the caller reads it.

    `include_relevance` is False for latest_on_topic, which orders by recency
    and so has no score worth showing.
    """
    cluster = result.get("speaker") or UNKNOWN_SPEAKER
    text = _strip_speaker_prefix(result.get("text", ""), cluster)
    speaker = display_speaker(result, names)

    header = (
        f"[{result['date']}] {result['show']} - "
        f"Episode {result['episode_number']} ({result['episode_title']}) "
        f"@ {result['start_time']:.1f}s | speaker: {speaker}"
    )
    if include_relevance:
        header += f" [relevance: {result['relevance_score']}]"

    return f"{header}\n{text}\n"


def format_results(
    results: list[dict],
    *,
    include_relevance: bool = True,
    names: dict[str, str] | None = None,
) -> str:
    """The whole response body."""
    return "\n---\n".join(
        format_result(r, include_relevance=include_relevance, names=names)
        for r in results
    )
