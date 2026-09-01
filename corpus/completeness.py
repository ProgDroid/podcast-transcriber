"""Is this episode fully and currently stored?"""

from __future__ import annotations

from corpus.chunking import count_chunks_from_text
from corpus.exclusions import is_excluded
from corpus.planning import Action, decide_action
from corpus.store import episode_where, is_complete, paged_get_ids


def episode_state(
    collection, show: str, episode_number: str, date_str: str
) -> tuple[list[str], int | None, str | None]:
    """Stored ids, the expected chunk count, and the rules version.

    The id list is PAGED. An unlimited get() silently returns 300, so an
    unpaged count against a 431-chunk episode reports short, judges a healthy
    episode torn, and re-embeds it on every run forever.
    """
    where = episode_where(show, episode_number, date_str)
    ids = paged_get_ids(collection, where)
    if not ids:
        return [], None, None
    head = collection.get(ids=ids[:1], include=["metadatas"])
    meta = (head.get("metadatas") or [{}])[0]
    n_chunks = meta.get("n_chunks")
    return (
        ids,
        n_chunks if isinstance(n_chunks, int) else None,
        meta.get("rules_version"),
    )


def plan_episode(
    collection,
    *,
    show: str,
    episode_number: str,
    date_str: str,
    transcript_text: str | None,
    episode_guid: str | None = None,
    expected_n_chunks: int | None = None,
) -> Action:
    """What this episode needs.

    `expected_n_chunks` may be supplied by a caller that has already counted;
    otherwise it is derived from the transcript. It is NEVER taken from stored
    records, which would freeze a torn episode as permanently complete.

    `episode_guid` MUST be passed by the cron path. Without it the exclusion
    check falls back to the triple alone, and the guid arm -- which exists
    precisely because the triple is not durable -- is unreachable from the
    only path that runs nightly.
    """
    excluded = is_excluded(show, episode_number, date_str, episode_guid)
    if transcript_text is None:
        return decide_action(
            transcript_exists=False,
            complete_in_chroma=False,
            stored_rules_version=None,
            excluded=excluded,
            parses_to_chunks=True,
        )

    if expected_n_chunks is None:
        expected_n_chunks = count_chunks_from_text(
            transcript_text, show, episode_number, "", date_str
        )
    stored_ids, _stored_n, stored_rules = episode_state(
        collection, show, episode_number, date_str
    )
    return decide_action(
        transcript_exists=True,
        complete_in_chroma=is_complete(stored_ids, expected_n_chunks),
        stored_rules_version=stored_rules,
        excluded=excluded,
        parses_to_chunks=expected_n_chunks > 0,
    )
