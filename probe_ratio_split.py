"""Confirmatory: split the ratio scan by each episode's MEASURED sample rate.

PRE-REGISTERED PREDICTION (before running): if the diagnosis is right, the
48 kHz episodes peak at r = 0.3333 and the 44.1 kHz episodes peak at
r = 0.3628 -- each at its own 16000/native -- rather than both peaking at the
same place. If both peak identically, something other than the file rate is
driving the offset and the diagnosis is incomplete.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from corpus.embedding import weighted_centroid
from corpus.speakers import GAP_CAP_S
from corpus.transcripts import load_episodes
from probe_rate import first_frame_rate


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

# measured rate per episode, from the local clip files
ep_rate = {}
for p in sorted(Path("clips").glob("*.mp3")):
    r = first_frame_rate(p)
    if r:
        ep_rate.setdefault(p.stem.split("--")[0] + ".txt", r)

cg = defaultdict(list)
for ck, vec in cv.items():
    stem, cl, _d = ck.split("--")
    cg[(stem + ".txt", cl)].append(vec)
clipc = {k: weighted_centroid(v, [1.0] * len(v)) for k, v in cg.items()}

turns_by_ep = {
    e["name"]: e["turns"] for e in load_episodes(Path("downloaded"), cap=GAP_CAP_S)
}
usable = {
    ep
    for ep in {k[0] for k in clipc}
    if len({c for (e, c) in clipc if e == ep}) >= 2
    and ep in turns_by_ep
    and ep in ep_rate
}
print("usable episodes with a MEASURED rate: %d" % len(usable))
byrate = defaultdict(set)
for ep in usable:
    byrate[ep_rate[ep]].add(ep)
print("  by rate: %s" % {k: len(v) for k, v in sorted(byrate.items())})


def speaker_at(turns, t):
    for tn in turns:
        if tn["start"] <= t < tn["start"] + tn["duration_s"]:
            return tn["speaker"]
    return None


def scan(eps, r):
    hit = miss = 0
    for c in rec:
        if c["episode"] not in eps or not c.get("vector") or c["length_s"] != 30.0:
            continue
        spk = speaker_at(turns_by_ep[c["episode"]], c["crop_start_s"] * r)
        cands = {cl: v for (e, cl), v in clipc.items() if e == c["episode"]}
        if spk is None or spk not in cands:
            continue
        u = unit(c["vector"])
        if max(cands, key=lambda cl: dot(u, cands[cl])) == spk:
            hit += 1
        else:
            miss += 1
    return hit, miss


grid = [
    1.0,
    0.5,
    0.40,
    0.3810,
    0.3700,
    0.3628,
    0.3556,
    0.3478,
    0.3404,
    0.3333,
    0.3200,
    0.30,
]
for rate in sorted(byrate):
    eps = byrate[rate]
    print(
        "\n=== episodes measured at %d Hz (n=%d)  expected peak r=%.4f ==="
        % (rate, len(eps), 16000 / rate)
    )
    best = (-1, None)
    for r in grid:
        h, m = scan(eps, r)
        acc = h / (h + m) if h + m else 0.0
        if acc > best[0]:
            best = (acc, r)
        print("   r=%.4f  hit=%3d miss=%3d  acc=%.3f" % (r, h, m, acc))
    print(
        "   PEAK r=%.4f acc=%.3f   (16000/%d = %.4f)"
        % (best[1], best[0], rate, 16000 / rate)
    )
