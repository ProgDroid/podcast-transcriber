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


def usable_segments(segments: list[dict]) -> tuple[list[dict], int]:
    """Segments `merge_turns` can safely consume, and how many were dropped.

    THIS EXISTS BECAUSE OF WHERE ITS CALLER SITS. Every other user of
    `merge_turns` feeds it segments parsed from a saved transcript, where
    `start` is guaranteed by the line format. `identify_clusters` is the only
    one handed RAW `result["segments"]` straight out of
    `whisperx.assign_word_speakers`, and `corpus/speakers.py` indexes
    `segments[i + 1]["start"]` unguarded -- so one segment without a numeric
    `start` raises KeyError and kills identity for the entire episode.

    Note the asymmetry that made this easy to miss: the same module already
    defends `speaker` with `.get(..., "UNKNOWN")` two lines later. The
    optional-field question was asked about one key and not its neighbour.

    Dropping rather than aborting: losing a stray segment shifts one turn
    duration slightly, while aborting loses identity for the whole episode --
    and the caller policy is that identity failure must cost as little as
    possible. The count comes back so the caller can say so out loud instead
    of absorbing it.

    `bool` is excluded deliberately. `True == 1` in Python, so a bool would
    pass an `isinstance(x, int)` check and be read as a timestamp of 1
    second.
    """
    kept = [
        seg
        for seg in segments
        if isinstance(seg.get("start"), (int, float))
        and not isinstance(seg.get("start"), bool)
    ]
    return kept, len(segments) - len(kept)


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
    segments, dropped = usable_segments(segments)
    if dropped:
        print(
            f"WARNING: {dropped} segment(s) had no usable `start` and were "
            f"dropped before identification."
        )
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

    KNOWN EXPOSURE, confirmed by
    `tests/test_write_path.py::test_a_stale_speaker_id_survives_a_re_transcribe`.
    This function can ADD a `speaker_id` and can never REMOVE one. Chroma
    merges metadata and `corpus/writing.py` writes POSITIONAL chunk ids, so
    re-transcribing an episode that already carries identities -- transcript
    lost from the volume, or a forced re-run -- renumbers the diarisation
    clusters while chunk i keeps the PREVIOUS run's `speaker_id`. A chunk
    labelled `SPEAKER_02` can then carry a name assigned when cluster 2 was
    somebody else, and `display_speaker` prefers `speaker_id`, so a reader
    sees a confident wrong name.

    NOT fixed here, because the only cure is deleting the episode's records
    before writing, and `upsert_then_prune` deliberately upserts first so a
    crash mid-write leaves MORE data rather than less. Reversing that is a
    write-path decision, not an identity one. Routine ingest never hits it:
    an episode is transcribed once.
    """
    stamped = 0
    for chunk in chunks:
        person = identified.get(chunk["metadata"].get("speaker"))
        if person is not None:
            chunk["metadata"]["speaker_id"] = person
            stamped += 1
    return stamped
