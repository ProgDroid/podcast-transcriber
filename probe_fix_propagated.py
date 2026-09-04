"""THROWAWAY PROBE -- did the sample-rate fix actually reach the container?

`speaker_embed.py` is not named in `add_local_python_source`; it rides along
as the entrypoint module. If that mounting does not pick up the edit, the
full archive pass would spend 56 GB faithfully reproducing the bug and every
downstream number would look exactly as broken as before, with no error.

The discriminating question is not "are the new vectors plausible" -- the
corrupt ones were entirely plausible. It is "are they DIFFERENT from the
ones the buggy code produced for the same crop". Identical means the old
code ran.

PRE-REGISTERED, before running:

  * shared crop identities  > 0        (a join matching nothing would print
                                        "0 identical" and look like a pass)
  * self-control cosine     == 1.0     (old vs old, same identity -- proves
                                        the join and the metric work at all)
  * old vs new, median      < 0.9      (the fix moved the audio)
  * fraction at cos > 0.999 == 0.0     (nothing survived unchanged)
  * same-cluster agreement  > 0.75     (the corrupt run scored median 0.435
                                        against a bake-off same-person 0.842;
                                        this is the number the fix is for)

If the last one fails while the others pass, the fix propagated but did not
work, which is a different problem and must not be read as the same one.

Run:  py probe_fix_propagated.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

OLD = Path("speakers/corrupt-pre-68231ba/archive_vectors.jsonl")
NEW = Path("speakers/archive_vectors.jsonl")


def load(path: Path) -> dict[tuple, list[float]]:
    """Last occurrence per crop identity wins, matching `dedupe_records`."""
    out: dict[tuple, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("vector") is None:
                continue
            key = (
                record["episode"],
                record["cluster"],
                record["draw"],
                record["length_s"],
            )
            out[key] = record["vector"]
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def main() -> int:
    for path in (OLD, NEW):
        if not path.exists():
            print(f"MISSING: {path}")
            return 2

    old, new = load(OLD), load(NEW)
    print(f"old records: {len(old)}   new records: {len(new)}")

    shared = sorted(set(old) & set(new))
    print(f"shared crop identities: {len(shared)}")
    if not shared:
        # A join matching nothing produces a clean-looking "nothing identical"
        # result that means the opposite of what it appears to mean.
        print("REFUSING: no shared identities. The join is broken, not the fix.")
        return 2

    # Positive control. If this is not 1.0 the comparison itself is wrong and
    # nothing below can be believed in either direction.
    control = [cosine(old[k], old[k]) for k in shared[:50]]
    control_min = min(control)
    print(f"CONTROL old-vs-old min cosine: {control_min:.6f}  (expect 1.000000)")
    if control_min < 0.9999:
        print("REFUSING: the self-comparison control failed.")
        return 2

    changed = [cosine(old[k], new[k]) for k in shared]
    identical = sum(1 for c in changed if c > 0.999)
    print(
        f"old vs new: median={statistics.median(changed):.4f} "
        f"min={min(changed):.4f} max={max(changed):.4f}"
    )
    print(f"unchanged (cos > 0.999): {identical} of {len(changed)}")

    # The fix's actual purpose: two crops of the SAME diarisation cluster
    # should now agree. The corrupt run scored median 0.435 here.
    by_cluster: dict[tuple, list[list[float]]] = {}
    for (episode, cluster, _draw, length_s), vector in new.items():
        if length_s == 30.0:
            by_cluster.setdefault((episode, cluster), []).append(vector)
    within = [
        cosine(a, b)
        for vectors in by_cluster.values()
        for a, b in combinations(vectors, 2)
    ]
    if within:
        print(
            f"same-cluster 30s crop agreement: n={len(within)} "
            f"median={statistics.median(within):.4f}  "
            f"(corrupt run scored 0.435; bake-off same-person 0.842)"
        )
    else:
        print("same-cluster agreement: UNKNOWN -- no cluster had two 30s crops")

    verdict = "FIX PROPAGATED" if identical == 0 else "FIX DID NOT PROPAGATE"
    print(f"VERDICT: {verdict}")
    print("PROBE COMPLETE")  # terminal marker; absence means it did not finish
    return 0


if __name__ == "__main__":
    sys.exit(main())
