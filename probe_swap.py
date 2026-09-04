"""Is the archive crop vector attached to the cluster it claims?

Independent of enrolment centroids entirely. For each episode, compare every
archive cluster centroid against every CLIP vector from the SAME episode. If
keying is sound, an archive cluster's nearest clip-vector cluster is itself.
"""

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from corpus.embedding import weighted_centroid


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


rec = json.loads(Path("speakers/archive_vectors.json").read_text(encoding="utf-8"))
cv = json.loads(
    Path("speakers/clip_vectors_speaker-diarization-community-1.json").read_text(
        encoding="utf-8"
    )
)

# archive cluster centroids
g = defaultdict(list)
for r in rec:
    g[(r["episode"], r["cluster"])].append((r["vector"], r["turn_duration_s"]))
arch = {
    k: weighted_centroid([v for v, _ in it], [w for _, w in it]) for k, it in g.items()
}

# clip vectors grouped by (episode.txt, cluster) -> mean of that cluster's clips
cg = defaultdict(list)
for ck, vec in cv.items():
    stem, cl, _d = ck.split("--")
    cg[(stem + ".txt", cl)].append(vec)
clip = {k: weighted_centroid(v, [1.0] * len(v)) for k, v in cg.items()}

# episodes where BOTH sides know 2+ clusters
by_ep_arch = defaultdict(list)
for ep, cl in arch:
    by_ep_arch[ep].append(cl)
by_ep_clip = defaultdict(list)
for ep, cl in clip:
    by_ep_clip[ep].append(cl)

eps = [
    e for e in by_ep_clip if len(by_ep_clip[e]) >= 2 and len(by_ep_arch.get(e, [])) >= 2
]
print("episodes with >=2 clusters on BOTH sides:", len(eps))
if not eps:
    print("POSITIVE CONTROL FAILED: nothing to compare. Result is UNKNOWN, not clean.")
    raise SystemExit(1)

self_best = 0
other_best = 0
selfsims = []
bestsims = []
examples = []
for ep in sorted(eps):
    cls = sorted(set(by_ep_clip[ep]) & set(by_ep_arch[ep]))
    if len(cls) < 2:
        continue
    for a in cls:
        av = arch[(ep, a)]
        ranked = sorted(((dot(av, clip[(ep, b)]), b) for b in cls), reverse=True)
        best_s, best_c = ranked[0]
        own_s = next(s for s, c in ranked if c == a)
        selfsims.append(own_s)
        bestsims.append(best_s)
        if best_c == a:
            self_best += 1
        else:
            other_best += 1
            if len(examples) < 12:
                examples.append((ep, a, best_c, round(own_s, 3), round(best_s, 3)))

n = self_best + other_best
print("\nPROBE H: archive cluster centroid vs same-episode clip centroids")
print("  comparisons: %d" % n)
print(
    "  nearest clip cluster IS itself : %d (%.1f%%)" % (self_best, 100 * self_best / n)
)
print(
    "  nearest is a DIFFERENT cluster : %d (%.1f%%)"
    % (other_best, 100 * other_best / n)
)
print("  median cosine to OWN clip cluster : %.4f" % statistics.median(selfsims))
print("  median cosine to BEST clip cluster: %.4f" % statistics.median(bestsims))
print("\n  mismatches (archive cluster -> nearest clip cluster, own / best):")
for e in examples:
    print(
        "    %-58s %s -> %s  own=%.3f best=%.3f" % (e[0][:58], e[1], e[2], e[3], e[4])
    )

# how often is the mismatch a clean 2-cluster SWAP?
swaps = Counter()
for ep in sorted(eps):
    cls = sorted(set(by_ep_clip[ep]) & set(by_ep_arch[ep]))
    if len(cls) != 2:
        continue
    a, b = cls
    na = max(cls, key=lambda x: dot(arch[(ep, a)], clip[(ep, x)]))
    nb = max(cls, key=lambda x: dot(arch[(ep, b)], clip[(ep, x)]))
    if na == a and nb == b:
        swaps["identity"] += 1
    elif na == b and nb == a:
        swaps["SWAPPED"] += 1
    else:
        swaps["collapsed(both->same)"] += 1
print("\n  two-cluster episodes, mapping shape: %s" % dict(swaps))
