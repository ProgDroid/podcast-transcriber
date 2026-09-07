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


def stamp_chunks(
    chunks: list[dict],
    identified: dict[str, str],
    *,
    identity_version: str | None,
) -> int:
    """Write `speaker_id` onto identified chunks, and the version onto all.

    Returns how many chunks were stamped with a name. Separate from
    `identify_clusters` and from the ingest function because this is the join
    that silently regresses: it depends on `chunking.py` calling the metadata
    key `speaker`, and if that ever drifts the stamp would land nowhere while
    every log line still reported clusters identified.

    A cluster with no match is skipped rather than given a placeholder. An
    absent `speaker_id` IS the unidentified state -- writing a sentinel would
    put a value meaning "no value" into the field the MCP filter matches on.

    `identity_version` goes on EVERY chunk, including chunks of clusters that
    matched nobody. That is what `episode_state`'s head-record read depends
    on: it takes one arbitrary record and would otherwise report an episode's
    identity from whichever chunk the paging happened to return first. The key
    answers "did identification RUN", never "did it MATCH" -- 27% of archive
    clusters legitimately match nobody, and an episode of theirs must reach
    SKIP rather than be re-queued for audio every night forever.

    THE ONE EXCLUSION: if `identified` is non-empty and zero chunks were
    stamped, the version is withheld. Clusters named and not one chunk taking
    a value can only mean the `speaker` key drifted between `chunking.py` and
    here. Stamping the version there would record "identification ran" for an
    episode that stored no identity, retiring it from repair forever and
    converting a loud nightly failure into a silent permanent one. This is the
    same `STAMP JOIN BROKEN` predicate `transcribe.py` alarms on -- one
    predicate, two consumers.

    Withholding is also how two of the unstamped-episode producers self-heal
    with no code of their own: an enrolment failure and an `identify_clusters`
    exception both skip this call entirely, so the episode is stored unstamped
    and the planner queues it for IDENTIFY_ONLY.

    KNOWN EXPOSURE, confirmed by
    `tests/test_write_path.py::test_a_stale_speaker_id_survives_a_re_transcribe`,
    and fixed in PHASE 2 of
    `docs/superpowers/specs/2026-09-05-identity-lifecycle-design.md`, not here.
    This function can ADD a `speaker_id` and can never REMOVE one. Chroma
    merges metadata and `corpus/writing.py` writes POSITIONAL chunk ids, so a
    write that moves a chunk boundary leaves chunk i carrying the PREVIOUS
    run's `speaker_id`. The cure is deleting the episode's records before
    writing, which is a write-path decision (`upsert_then_prune` upserts first
    on purpose, so a crash leaves MORE data rather than less).

    Until that lands, the caller on the TRANSCRIBE path must not stamp the
    version over records that already carry a `speaker_id` -- doing so would
    CERTIFY positionally-stale names and retire the episode from repair. See
    the phase-1 guard in `transcribe.py`.
    """
    stamped = 0
    for chunk in chunks:
        person = identified.get(chunk["metadata"].get("speaker"))
        if person is not None:
            chunk["metadata"]["speaker_id"] = person
            stamped += 1

    # None WITHHOLDS the version while still writing this run's names --
    # the phase-1 TRANSCRIBE guard. The names computed this run are
    # correct for this run's chunks; it is the surviving OLD records that
    # carry stale stamps, so only the certification is unsafe.
    if identity_version is None:
        return stamped

    if identified and stamped == 0:
        return stamped

    for chunk in chunks:
        chunk["metadata"]["identity_version"] = identity_version
    return stamped


def _is_cluster(label) -> bool:
    """Is this label a diarisation cluster, as opposed to `UNKNOWN`?

    `corpus/chunking.py` substitutes the literal `UNKNOWN` for a segment
    carrying no speaker, and 53 of the 439 transcripts hold such a bucket
    (measured 2026-09-05). That bucket is audio belonging to nobody in
    particular, and naming it would be the silent over-merge this line of work
    treats as the error that matters most.

    `corpus/embedding.py::cluster_crop_plan` already filters to this same
    prefix, so `UNKNOWN` never reaches `assign` and never acquires a
    `speaker_id` in the first place. THE GUARANTEE IS REAL BUT INHERITED FROM
    ANOTHER MODULE: `carry_identity` reads stored records rather than
    `identify_clusters`' output, so it must not depend on a filter three
    modules away staying where it is. It applies the rule itself, here, where
    a reader of this file meets it.
    """
    return isinstance(label, str) and label.startswith("SPEAKER_")


def carry_identity(
    stored: list[dict], chunks: list[dict], *, identity_version: str
) -> tuple[int, list[str], list[str]]:
    """Re-apply an episode's existing cluster->person map to re-chunked text.

    Returns the number of chunks stamped, the clusters REFUSED as
    self-contradictory, and the incoming labels found UNMAPPED.

    WHY THIS EXISTS. The cluster->person mapping is invariant under
    re-chunking: an `EMBED_ONLY` re-embed re-chunks the SAME transcript, so
    its `[SPEAKER_XX]` labels are unchanged and `build_chunks_from_text`
    derives the same `speaker` values. Only the positional layout moves, and
    the existing records already hold the mapping. Re-earning it from audio
    would mean a full download plus a GPU pass to recompute names that are
    already correct -- 56 GB across the archive.

    The join is on `speaker`, NEVER on position. Position is the whole defect
    this design closes.

    The map is keyed on every stored cluster label and valued person-or-None:
    ATTEMPT, not outcome. Keying it only on named clusters cannot work,
    because 27% of clusters legitimately match nobody and a label with no name
    would then be indistinguishable from a label that has gone missing.

    Within one label, PRESENT BEATS ABSENT. That is the ordinary shape of a
    torn episode's repair, where the records written before identification ran
    carry no name and the ones written after do.

    TWO REFUSALS, both of which carry NOTHING and withhold the version, so the
    episode falls back to re-earning identity from audio rather than being
    certified with a hole in it:

    1.  A cluster whose stored records name two different people. This is the
        design's only self-checking read of its own past output. Two things it
        can mean, and the caller must not claim the first: usually the stamps
        are corrupt, but a same-day unnumbered TRIPLE COLLISION (still open in
        `docs/corpus-integrity.md`) makes two distinct episodes share
        `episode_id_prefix` and interleave, at which point one label
        legitimately carries two people. Refusing is right either way; calling
        it corruption is not, so the refusal is reported as a conflict and the
        caller names the triple.

    2.  An incoming label absent from the stored map -- the SUBSET RULE. One
        rule, two causes. `RULES_VERSION` is bumped "whenever anything that
        derives chunk CONTENT from a transcript changes -- today the speaker
        labels", so a rules bump is precisely the event that may change how
        `speaker` is derived: the transcript's labels are invariant, the
        DERIVATION is not. And with no rules bump at all, a tear removes a
        contiguous tail of positions, so a cluster appearing only in that tail
        has no stored record and no map entry. Certifying either would send
        the episode to SKIP with a speaker permanently unnamed and nothing
        signalling it.

    `UNKNOWN` is excluded from BOTH sides of the subset test. Counting it
    would fail all 53 UNKNOWN-bucket transcripts on every run and re-download
    their audio nightly forever. Those chunks still receive
    `identity_version`, because uniformity is what `episode_state`'s
    single-record read depends on.

    Reports come back sorted so that two runs over the same data produce the
    same nightly line; set iteration order would make an unchanged backlog
    look like a changing one.
    """
    mapping: dict[str, str | None] = {}
    conflicting: set[str] = set()

    for meta in stored:
        label = meta.get("speaker")
        if not _is_cluster(label):
            continue
        person = meta.get("speaker_id")
        if label not in mapping:
            mapping[label] = person
            continue
        current = mapping[label]
        if person is None or person == current:
            continue
        if current is None:
            mapping[label] = person  # present beats absent
            continue
        conflicting.add(label)

    incoming = {
        chunk["metadata"].get("speaker")
        for chunk in chunks
        if _is_cluster(chunk["metadata"].get("speaker"))
    }
    unmapped = sorted(incoming - set(mapping))
    refused = sorted(conflicting)

    if refused or unmapped:
        return 0, refused, unmapped

    stamped = 0
    for chunk in chunks:
        meta = chunk["metadata"]
        label = meta.get("speaker")
        person = mapping.get(label) if _is_cluster(label) else None
        if person is not None:
            meta["speaker_id"] = person
            stamped += 1
        meta["identity_version"] = identity_version
    return stamped, refused, unmapped


def has_stored_identity(stored: list[dict]) -> bool:
    """Does this episode already carry a name that a merge could preserve?

    THE PHASE-1 GUARD'S PREDICATE. `stamp_chunks` runs on the TRANSCRIBE path,
    which until phase 2 lands has no delete before its write. Chroma merges
    metadata and the chunk ids are positional, so a re-transcribe leaves
    chunk i holding the PREVIOUS run's `speaker_id` while the new diarisation
    has renumbered the clusters. Writing `identity_version` over that state
    would certify the stale names and retire the episode from IDENTIFY_ONLY
    permanently -- and phase 2 ships no rescan, so the only recovery would be
    an archive-wide IDENTITY_RULES_VERSION bump.

    So: any surviving name at all means withhold the version and report.
    Loud and unresolved beats silent and certified. ONE is enough -- the job
    is to refuse to certify, not to measure how wrong the episode is.

    An empty string is not a name. Chroma cannot store None, so a field a
    repair has cleared comes back as `""`, and reading that as a name would
    guard every episode that was ever cleared.
    """
    return any(meta.get("speaker_id") for meta in stored)
