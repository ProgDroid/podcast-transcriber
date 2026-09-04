"""THROWAWAY PROBE -- are the residual 45 conflicts BENIGN or MALIGNANT?

The sample-rate fix took co-membership conflicts from 30.8% to 6.4%. That is
still above the 5% guardrail, so the rate alone says "do not backfill". But
the rate counts two opposite failures as if they were one thing:

  BENIGN    the diariser split ONE speaker into two clusters. Both really
            are that person. The backfill would label both correctly and
            the conflict is an artifact of counting, not an error.

  MALIGNANT two DIFFERENT people collapsed onto one identity. This is what
            the corrupt run was full of (Marko's cluster scoring 0.879
            against Jacob's centroid) and it is the error that would put
            one person's words under another person's name.

The discriminator: for a conflicting pair of clusters, how similar are they
to EACH OTHER? One split voice is two views of the same person and should
score high between themselves. Two collapsed people should not.

PRE-REGISTERED, before running:

  * cross-person baseline median  < 0.55  (positive control: clusters
                                           assigned to DIFFERENT people in
                                           the same episode. If this is not
                                           clearly low, the metric does not
                                           discriminate and nothing below
                                           can be read either way.)
  * conflict-pair median          > 0.75  if the residual is over-segmentation
  * fraction of conflict pairs
    scoring below the baseline
    median                        ~ 0     any such pair is a candidate
                                          MALIGNANT case and is listed

If conflict pairs come back looking like the cross-person baseline, the
residual is real misattribution and Task 8 stays blocked.

Also re-checks the corrupt run's smoking gun: The Jacob Shapiro Podcast
Episode 100, where SPEAKER_01 is Marko Papic by transcript text and was
assigned to Jacob at 0.879.

Run:  py probe_conflict_shape.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ASSIGNMENTS = Path("speakers/assignments.json")
VECTORS = Path("speakers/archive_vectors.json")
SMOKING_GUN = "The Jacob Shapiro Podcast - Episode 100"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def centroids() -> dict[tuple[str, str], list[float]]:
    """Mean vector per (episode, cluster), matching the assignment's own view."""
    buckets: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for record in json.loads(VECTORS.read_text(encoding="utf-8")):
        if record.get("vector") is not None:
            buckets[(record["episode"], record["cluster"])].append(record["vector"])
    out = {}
    for key, vectors in buckets.items():
        dim = len(vectors[0])
        out[key] = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    return out


def main() -> int:
    data = json.loads(ASSIGNMENTS.read_text(encoding="utf-8"))
    cents = centroids()
    print(f"centroids: {len(cents)}")

    # Who each cluster was assigned to, keyed the same way as the centroids.
    assigned: dict[tuple[str, str], str] = {}
    for row in data["assignments"]:
        person = row.get("person") or row.get("name") or row.get("assigned")
        if person:
            assigned[(row["episode"], row["cluster"])] = person
    print(f"assigned clusters: {len(assigned)}")
    if not assigned:
        print("REFUSING: parsed zero assignments; the key names are wrong.")
        return 2

    # -- positive control: DIFFERENT people, same episode -----------------
    by_episode: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (episode, cluster), person in assigned.items():
        by_episode[episode].append((cluster, person))
    cross = [
        cosine(cents[(ep, ca)], cents[(ep, cb)])
        for ep, items in by_episode.items()
        for (ca, pa), (cb, pb) in combinations(sorted(items), 2)
        if pa != pb and (ep, ca) in cents and (ep, cb) in cents
    ]
    if not cross:
        print("REFUSING: no cross-person pairs; the control cannot be computed.")
        return 2
    baseline = statistics.median(cross)
    print(
        f"CONTROL cross-person same-episode pairs: n={len(cross)} "
        f"median={baseline:.4f}  (expect < 0.55)"
    )

    # -- the question: conflicting pairs ---------------------------------
    pairs: list[tuple[float, str, str, str, str]] = []
    for conflict in data["conflicts"]:
        episode = conflict["episode"]
        person = conflict.get("person") or conflict.get("name")
        clusters = conflict["clusters"]
        for ca, cb in combinations(sorted(clusters), 2):
            if (episode, ca) in cents and (episode, cb) in cents:
                score = cosine(cents[(episode, ca)], cents[(episode, cb)])
                pairs.append((score, episode, person, ca, cb))
    if not pairs:
        print("REFUSING: no conflict pairs resolved to centroids.")
        return 2

    scores = [p[0] for p in pairs]
    print(
        f"CONFLICT pairs: n={len(pairs)} median={statistics.median(scores):.4f} "
        f"min={min(scores):.4f} max={max(scores):.4f}"
    )

    suspect = sorted(p for p in pairs if p[0] < baseline)
    print(f"conflict pairs scoring BELOW the cross-person baseline: {len(suspect)}")
    for score, episode, person, ca, cb in suspect:
        print(f"  MALIGNANT? {score:.4f}  {episode}  {person}  {ca}/{cb}")

    # -- the corrupt run's smoking gun ------------------------------------
    print(f"\n{SMOKING_GUN}:")
    hits = {
        cluster: person
        for (episode, cluster), person in sorted(assigned.items())
        if episode.startswith(SMOKING_GUN)
    }
    print(f"  assignments now: {hits or 'none assigned'}")

    benign = len(pairs) - len(suspect)
    print(
        f"\nVERDICT: {benign} of {len(pairs)} conflict pairs look like one "
        f"voice split in two; {len(suspect)} look like two people merged."
    )
    print("PROBE COMPLETE")  # terminal marker; absence means it did not finish
    return 0


if __name__ == "__main__":
    sys.exit(main())
