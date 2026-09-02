"""Matching a cluster to an enrolled voice, or to nobody.

Centroids are DERIVED from labels and never stored. `speakers/labels.json`
holds provenance only. This follows `corpus/exclusions.py`, where the lists
are derived for the same reason: two hand-maintained copies drift
asymmetrically, and the stale one works right up until the moment it matters.

`None` IS THE ANSWER, not a missing one. Under-merging is an annoyance a
person can repair; over-merging silently attributes one person's views to
another and corrupts the "has the view moved" query this exists to serve, with
nothing in the output to show it happened.
"""

from __future__ import annotations

import math

from corpus.embedding import weighted_centroid

# Only voice-verified labels enrol. `speaker_tool.py:705` offers "heard it (v)
# or inferred from context (c)" and warns in its own comment that a context
# label scores a matcher correct for the wrong reason.
ENROLMENT_BASIS = "voice"


def build_centroids(
    labels: dict[str, dict],
    vectors: dict[str, list[float]],
    weights: dict[str, float] | None = None,
) -> dict[str, list[float]]:
    """One centroid per named person, from voice-verified labels only.

    A label with no vector is skipped rather than raising: the caller may hold
    labels for clips it has not embedded, and refusing the whole enrolment
    over one absent clip would be brittle for no gain.
    """
    grouped: dict[str, list[tuple[list[float], float]]] = {}
    for clip_id, label in labels.items():
        person = label.get("person")
        if person is None or label.get("basis") != ENROLMENT_BASIS:
            continue
        vector = vectors.get(clip_id)
        if vector is None:
            continue
        weight = 1.0 if weights is None else weights.get(clip_id, 1.0)
        grouped.setdefault(person, []).append((vector, weight))

    return {
        person: weighted_centroid([v for v, _ in items], [w for _, w in items])
        for person, items in sorted(grouped.items())
    }


def score(
    vector: list[float], centroids: dict[str, list[float]]
) -> list[tuple[str, float]]:
    """Every candidate and its similarity, best first.

    A true cosine. The CENTROID side is unit-length by construction
    (`weighted_centroid` normalises), but the query side is a raw model
    output -- `_Embedder.embed` returns pyannote's vector untouched -- so
    dividing by its norm is required rather than cosmetic. Assuming otherwise
    makes every threshold scale-dependent and silently unstable.

    Ties break on name so the order is reproducible.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        raise ValueError("cannot score a zero vector")
    ranked = [
        (
            person,
            sum(a * b for a, b in zip(vector, centroid, strict=True)) / norm,
        )
        for person, centroid in centroids.items()
    ]
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked


def assign(
    vector: list[float],
    centroids: dict[str, list[float]],
    *,
    threshold: float,
    margin: float,
) -> str | None:
    """The enrolled person this vector belongs to, or None.

    BOTH conditions must hold. A threshold alone resolves the dangerous case
    -- two high scores a hair apart -- confidently and wrongly, so the runner
    up must also be beaten by `margin`. With a single enrolled voice there is
    no runner-up and the margin cannot apply.
    """
    ranked = score(vector, centroids)
    if not ranked:
        return None
    best_person, best_score = ranked[0]
    if best_score < threshold:
        return None
    if len(ranked) > 1 and best_score - ranked[1][1] < margin:
        return None
    return best_person


def co_membership_conflicts(assignments: list[dict]) -> list[dict]:
    """Episodes where two clusters were assigned the SAME person.

    A free over-merge detector needing no labels, so unlike the seed conflict
    check it covers every cluster rather than the 38% that carry a label.

    It is a FLAG, not a gate. Diarisation can over-segment one person into two
    clusters, and reuniting those is correct -- arguably the feature working.
    Two genuinely different people collapsing into one identity is the error.
    This cannot tell them apart; it partitions the archive into the cases
    worth looking at, which is all it claims to do.
    """
    seen: dict[tuple[str, str], list[str]] = {}
    for record in assignments:
        person = record.get("person")
        if person is None:
            continue
        seen.setdefault((record["episode"], person), []).append(record["cluster"])

    return [
        {"episode": episode, "person": person, "clusters": sorted(clusters)}
        for (episode, person), clusters in sorted(seen.items())
        if len(clusters) > 1
    ]
