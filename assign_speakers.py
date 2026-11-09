"""Assign each cluster to an enrolled voice, then count what is left over.

Two outputs. The assignments Ship 1 backfills, and the histogram that decides
whether Ship 2 is worth building -- which is the question the second half of
the design assumes and nobody has measured.

THRESHOLD AND MARGIN default to the values Task 4's bake-off measured
(`speakers/bakeoff.json`, both embedders agreeing to six decimal places):
threshold=0.516, margin=0.074. These are measured, not invented -- see
spec SS3.3, "Suggested threshold and margin for Task 6".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from corpus.embedding import weighted_centroid
from corpus.enrolment import (
    MARGIN,
    THRESHOLD,
    assign,
    build_centroids,
    co_membership_conflicts,
)
from corpus.identity import parse_transcript_filename


def centroids_by_cluster(records: list[dict]) -> dict[tuple[str, str], list[float]]:
    """One centroid per (episode, cluster), from that cluster's crops.

    Weighted by TRUE turn duration, not by crop length -- the distinction
    corpus/embedding.py exists to keep.
    """
    grouped: dict[tuple[str, str], list[tuple[list[float], float]]] = {}
    for record in records:
        if record.get("vector") is None:
            continue
        key = (record["episode"], record["cluster"])
        grouped.setdefault(key, []).append(
            (record["vector"], record["turn_duration_s"])
        )
    return {
        key: weighted_centroid([v for v, _ in items], [w for _, w in items])
        for key, items in grouped.items()
    }


class _UnionFind:
    """Plain disjoint-set over the unmatched cluster keys.

    Not `corpus/grouping.py` -- the spec is explicit that this measurement
    runs "before a line of corpus/grouping.py exists" (SS1.3). It exists only
    to size Ship 2, not to ship it.
    """

    def __init__(self, items: list[tuple[str, str]]) -> None:
        self.parent = {item: item for item in items}

    def find(self, x: tuple[str, str]) -> tuple[str, str]:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_unmatched(
    unmatched_keys: list[tuple[str, str]],
    centroids: dict[tuple[str, str], list[float]],
    *,
    threshold: float,
) -> list[list[tuple[str, str]]]:
    """Single-linkage grouping of unmatched centroids at the enrolment bar.

    Same similarity test `corpus.enrolment.score` uses for a named voice --
    reused here, not re-derived, because "the same voice" is one question
    whether the other side is an enrolled centroid or another unmatched one.
    No margin: a pairwise merge test has no runner-up to be confident against.
    O(n^2) over the unmatched population, which is small enough that this
    stays a local, seconds-long measurement rather than a reason to build
    real clustering.
    """
    uf = _UnionFind(unmatched_keys)
    for i, key_a in enumerate(unmatched_keys):
        vec_a = centroids[key_a]
        for key_b in unmatched_keys[i + 1 :]:
            vec_b = centroids[key_b]
            similarity = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
            if similarity >= threshold:
                uf.union(key_a, key_b)

    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key in unmatched_keys:
        groups.setdefault(uf.find(key), []).append(key)
    return list(groups.values())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign clusters to enrolled voices and size Ship 2."
    )
    parser.add_argument(
        "clip_vectors",
        help="the winning speakers/clip_vectors_<embedder>.json from Task 4",
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--margin", type=float, default=MARGIN)
    args = parser.parse_args()

    labels = json.loads(Path("speakers/labels.json").read_text(encoding="utf-8"))
    records = json.loads(
        Path("speakers/archive_vectors.json").read_text(encoding="utf-8")
    )
    clip_vectors = json.loads(Path(args.clip_vectors).read_text(encoding="utf-8"))

    voice_labels = {
        clip_id
        for clip_id, label in labels.items()
        if label.get("basis") == "voice" and label.get("person") is not None
    }
    overlap = voice_labels & clip_vectors.keys()
    print(
        f"labels: {len(labels)} total, {len(voice_labels)} basis=voice; "
        f"clip_vectors: {len(clip_vectors)}; overlap: {len(overlap)}"
    )
    if not overlap:
        print(
            "REFUSING: zero overlap between voice-verified label keys and "
            "clip vector keys. This is a keying bug, not a model result."
        )
        return 2

    centroids = build_centroids(labels, clip_vectors)
    print(f"enrolled={sorted(centroids)}")
    if not centroids:
        print("REFUSING: no enrolment centroids. Check the clip vector keys.")
        return 2

    clusters = centroids_by_cluster(records)
    print(f"clusters with a centroid: {len(clusters)}")

    assignments = []
    for (episode, cluster), vector in sorted(clusters.items()):
        parsed = parse_transcript_filename(episode)
        if parsed is None:
            print(f"REFUSING: cannot parse episode name {episode!r}")
            return 2
        show, episode_number, date_str = parsed
        assignments.append(
            {
                "episode": episode,
                # The triple the backfill filters on. Chunk metadata has NO
                # `episode` key (corpus/chunking.py:66-80); an episode is
                # (show, episode_number, date), and carrying it here means
                # Task 8 never re-derives it.
                "show": show,
                "episode_number": episode_number,
                "date": date_str,
                "cluster": cluster,
                "person": assign(
                    vector,
                    centroids,
                    threshold=args.threshold,
                    margin=args.margin,
                ),
            }
        )

    matched = [a for a in assignments if a["person"] is not None]
    print(f"assigned={len(matched)} of {len(assignments)}")
    for person, count in Counter(a["person"] for a in matched).most_common():
        print(f"  {person}: {count} clusters")

    conflicts = co_membership_conflicts(assignments)
    print(f"\nco-membership conflicts: {len(conflicts)}")
    for conflict in conflicts[:20]:
        print(f"  {conflict['episode']}: {conflict['person']} {conflict['clusters']}")
    if matched and len(conflicts) > 0.05 * len(matched):
        print(
            f"WARNING: conflicts ({len(conflicts)}) exceed 5% of matched "
            f"clusters ({len(matched)}). Read them before backfilling."
        )

    # --- the checkpoint -------------------------------------------------
    unmatched = [a for a in assignments if a["person"] is None]
    print(f"\nunmatched clusters: {len(unmatched)}")

    unmatched_keys = [(a["episode"], a["cluster"]) for a in unmatched]
    groups = group_unmatched(unmatched_keys, clusters, threshold=args.threshold)
    episode_counts = sorted(
        (len({episode for episode, _ in group}) for group in groups), reverse=True
    )
    histogram = Counter(episode_counts)
    print(f"\nunmatched centroids grouped into {len(groups)} local groups")
    print("group-size histogram (distinct episodes per group -> group count):")
    for episode_count, group_count in sorted(histogram.items(), reverse=True):
        print(f"  {episode_count} episodes: {group_count} groups")

    qualifying = sum(1 for c in episode_counts if c >= 5)
    clears = qualifying >= 15
    print(
        f"\nqualifying groups (>=5 distinct episodes): {qualifying}\n"
        f"pre-registered Ship 2 threshold: >=15 such groups\n"
        f"CHECKPOINT: {'CLEARS' if clears else 'DOES NOT CLEAR'} the threshold"
    )

    Path("speakers/assignments.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "margin": args.margin,
                "assignments": assignments,
                "conflicts": conflicts,
                "checkpoint": {
                    "unmatched_groups": len(groups),
                    "qualifying_groups": qualifying,
                    "ship2_threshold": 15,
                    "clears_threshold": clears,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("ASSIGN COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
