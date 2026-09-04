"""Decide who is speaking, from segments and the audio file, at ingest time.

This is the piece that ends the backfill. `backfill_speaker_id.py` exists
because identity was computed months after the chunks were written; the
nightly job already holds both inputs this needs -- the diarised segments and
the audio file, still on disk -- so identity can be settled before the chunks
are written at all.

It composes parts that already exist and are tested, and adds no new rule:

    merge_turns          (corpus.speakers)   segments  -> turns
    cluster_crop_plan    (corpus.embedding)  turns     -> crops to embed
    Embedder.embed       (corpus.voice)      crop      -> vector
    weighted_centroid    (corpus.embedding)  vectors   -> one per cluster
    assign               (corpus.enrolment)  centroid  -> a person, or None

**The centroid must be weighted by TRUE TURN DURATION**, matching
`assign_speakers.py:centroids_by_cluster`. The thresholds this scores against
(0.516 / 0.074) were measured against centroids built that way, so any other
weighting produces vectors that are not comparable to the measurement -- and
it fails silently, because the assignments still look reasonable.

Failures are NOT swallowed. A broken embedder and a genuinely unrecognised
voice must not both render as "unidentified": the first is a fault to repair,
the second is the expected state for 27% of clusters. The caller decides.
"""

from __future__ import annotations

from corpus.embedding import cluster_crop_plan, weighted_centroid
from corpus.enrolment import assign
from corpus.speakers import CLIP_MIN_TURN_S, GAP_CAP_S, merge_turns


def identify_clusters(
    segments: list[dict],
    audio_path: str,
    embedder,
    centroids: dict[str, list[float]],
    *,
    threshold: float,
    margin: float,
    gap_cap_s: float = GAP_CAP_S,
    min_turn_s: float = CLIP_MIN_TURN_S,
    episode_name: str = "",
) -> dict[str, str]:
    """Map each diarisation cluster to an enrolled person, where one matches.

    Returns only the clusters that matched. **An absent key is the
    unidentified state** -- the same contract the archive pass used, and the
    one `display_speaker` already renders. Writing a placeholder instead would
    put a value into `speaker_id` that means "no value", which the MCP filter
    would then match on.
    """
    turns = merge_turns(segments, gap_cap_s=gap_cap_s)
    if not turns:
        return {}

    crops = cluster_crop_plan(
        {"name": episode_name, "turns": turns}, min_turn_s=min_turn_s
    )
    if not crops:
        return {}

    # Weight by the turn's real duration, not the crop's length. Every crop
    # drawn from one turn carries that turn's weight, so a cluster whose
    # evidence is one long turn and one short one leans toward the long one.
    grouped: dict[str, list[tuple[list[float], float]]] = {}
    for crop in crops:
        vector = embedder.embed(audio_path, crop["crop_start_s"], crop["crop_length_s"])
        grouped.setdefault(crop["cluster"], []).append(
            (vector, crop["turn_duration_s"])
        )

    identified: dict[str, str] = {}
    for cluster, items in sorted(grouped.items()):
        centroid = weighted_centroid([v for v, _ in items], [w for _, w in items])
        person = assign(centroid, centroids, threshold=threshold, margin=margin)
        if person is not None:
            identified[cluster] = person
    return identified


def stamp_chunks(chunks: list[dict], identified: dict[str, str]) -> int:
    """Write `speaker_id` into each chunk whose cluster was identified.

    Returns how many chunks were stamped. Separate from `identify_clusters`
    and from the ingest function because this is the join that silently
    regresses: it depends on `chunking.py` calling the metadata key
    `speaker`, and if that ever drifts the stamp would land nowhere while
    every log line still reported clusters identified.

    A cluster with no match is skipped rather than given a placeholder. An
    absent `speaker_id` IS the unidentified state -- writing a sentinel would
    put a value meaning "no value" into the field the MCP filter matches on.
    """
    stamped = 0
    for chunk in chunks:
        person = identified.get(chunk["metadata"].get("speaker"))
        if person is not None:
            chunk["metadata"]["speaker_id"] = person
            stamped += 1
    return stamped
