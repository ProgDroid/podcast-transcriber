"""Which embedder separates these five voices better?

Leave-one-out over the 208 voice-verified labels: hold one clip out, build
centroids from the rest, and ask whether the held-out clip is assigned to its
own labelled person. Cheap, local, and it uses only audio already on disk.

NOT A PRECISION CLAIM. Per the eval design's 3.4 these labels are a dev set --
learning from them is exactly what a dev set is for, and quoting a number off
them as precision is what 3.4 forbids. This picks a model; it does not measure
the system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corpus.enrolment import assign, build_centroids, score

LABELS_PATH = Path("speakers/labels.json")
THRESHOLD = 0.0  # scored as a pure nearest-centroid choice: no refusals here,
MARGIN = 0.0  # because the question is separability, not the operating point.

# `embed_clips` raises only if NOTHING embedded (Task 3 review), so a partial
# failure would yield fewer than 381 vectors and look like a normal, smaller
# file. Pre-registered alongside the decision rule: any vector file that does
# not hold exactly this many keys is refused rather than scored, because a
# silently short input would move every accuracy figure without saying so.
EXPECTED_VECTOR_COUNT = 381


def leave_one_out(labels: dict[str, dict], vectors: dict[str, list[float]]) -> dict:
    voice = {
        clip_id: label
        for clip_id, label in labels.items()
        if label.get("basis") == "voice" and label.get("person") is not None
    }
    scored = correct = skipped = 0
    confusions: dict[tuple[str, str], int] = {}
    same: list[float] = []
    different: list[float] = []

    for held_out, label in sorted(voice.items()):
        if held_out not in vectors:
            skipped += 1
            continue
        rest = {k: v for k, v in voice.items() if k != held_out}
        centroids = build_centroids(rest, vectors)
        if label["person"] not in centroids:
            # The only clip for this person; nothing to enrol from.
            skipped += 1
            continue
        ranked = score(vectors[held_out], centroids)
        for person, value in ranked:
            (same if person == label["person"] else different).append(value)
        predicted = assign(
            vectors[held_out], centroids, threshold=THRESHOLD, margin=MARGIN
        )
        scored += 1
        if predicted == label["person"]:
            correct += 1
        else:
            confusions[(label["person"], str(predicted))] = (
                confusions.get((label["person"], str(predicted)), 0) + 1
            )

    # The SCORE DISTRIBUTION, not just the accuracy. Task 6's THRESHOLD and
    # MARGIN are the only unmeasured numbers in this plan and the spec leaves
    # them unset on purpose (6.7); they are meant to come from here. A
    # bake-off that printed accuracy alone would leave the next task guessing.
    same.sort()
    different.sort()

    def pct(values: list[float], q: float) -> float:
        return values[min(int(q * len(values)), len(values) - 1)] if values else 0.0

    return {
        "scored": scored,
        "correct": correct,
        "skipped": skipped,
        "accuracy": correct / scored if scored else 0.0,
        "confusions": {f"{a} -> {b}": n for (a, b), n in sorted(confusions.items())},
        "same_person_score_p05": pct(same, 0.05),
        "same_person_score_median": pct(same, 0.5),
        "other_person_score_p95": pct(different, 0.95),
        "other_person_score_median": pct(different, 0.5),
        "suggested_threshold": (pct(same, 0.05) + pct(different, 0.95)) / 2,
        "suggested_margin": max(0.0, (pct(same, 0.05) - pct(different, 0.95)) / 4),
    }


def main() -> int:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    results = {}
    for path in sorted(Path("speakers").glob("clip_vectors_*.json")):
        name = path.stem.replace("clip_vectors_", "")
        vectors = json.loads(path.read_text(encoding="utf-8"))
        # A partial embed_clips run raises only if NOTHING embedded, so a
        # short file looks completely normal otherwise. Refuse rather than
        # silently score a shrunken input.
        if len(vectors) != EXPECTED_VECTOR_COUNT:
            print(
                f"\n== {name}: REFUSING -- expected "
                f"{EXPECTED_VECTOR_COUNT} vectors, found {len(vectors)}. "
                "This is a short/partial embed run, not a model result."
            )
            return 2
        # Prove the probe can return something before believing its verdict:
        # zero overlap between labels and vectors would score 0.0 and look
        # exactly like a model that cannot tell two people apart.
        overlap = len(set(labels) & set(vectors))
        print(
            f"\n== {name}: labels={len(labels)} "
            f"vectors={len(vectors)} overlap={overlap}"
        )
        if overlap == 0:
            print("  REFUSING: no clip_id in common. This is a key mismatch,")
            print("  not a model result.")
            return 2
        outcome = leave_one_out(labels, vectors)
        results[name] = outcome
        print(
            f"  accuracy={outcome['accuracy']:.3f} "
            f"({outcome['correct']}/{outcome['scored']}, skipped={outcome['skipped']})"
        )
        for pair, count in outcome["confusions"].items():
            print(f"    {pair}: {count}")
        print(
            f"  same-person p05={outcome['same_person_score_p05']:.3f} "
            f"median={outcome['same_person_score_median']:.3f}"
        )
        print(
            f"  other-person p95={outcome['other_person_score_p95']:.3f} "
            f"median={outcome['other_person_score_median']:.3f}"
        )
        print(
            f"  -> suggested threshold={outcome['suggested_threshold']:.3f} "
            f"margin={outcome['suggested_margin']:.3f}"
        )

    Path("speakers/bakeoff.json").write_text(json.dumps(results, indent=2), "utf-8")
    print("\nBAKEOFF COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
