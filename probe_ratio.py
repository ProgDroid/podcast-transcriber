"""Scan the offset ratio r: at what fraction of the nominal time did the crop land?

PRE-REGISTERED PREDICTION (written before running):
  If `_Embedder.embed` passes a 16 kHz frame count to torchaudio.load on a
  44.1/48 kHz file, the crop lands at nominal_time * 16000/native. Agreement
  between "who the crop sounds like" and "who the transcript says is speaking"
  should PEAK at r ~= 0.3628 (44.1 kHz) or r ~= 0.3333 (48 kHz), and be poor
  at r = 1.0.
  If it peaks at r = 1.0 instead, this diagnosis is WRONG and I will say so.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from corpus.embedding import weighted_centroid
from corpus.speakers import CLIP_MIN_TURN_S, GAP_CAP_S
from corpus.transcripts import load_episodes


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

# reference per (episode, cluster) from the CLIP pipeline (whole-file loads,
# immune to the offset path under test)
cg = defaultdict(list)
for ck, vec in cv.items():
    stem, cl, _d = ck.split("--")
    cg[(stem + ".txt", cl)].append(vec)
clipc = {k: weighted_centroid(v, [1.0] * len(v)) for k, v in cg.items()}

eps = load_episodes(Path("downloaded"), cap=GAP_CAP_S)
turns_by_ep = {e["name"]: e["turns"] for e in eps}
print("episodes loaded: %d ; clip reference clusters: %d" % (len(eps), len(clipc)))

# only episodes where the clip pipeline knows >=2 clusters (so "who is it"
# is a real question with a checkable answer)
usable_eps = {
    ep for ep in {k[0] for k in clipc} if len({c for (e, c) in clipc if e == ep}) >= 2
}
usable_eps &= set(turns_by_ep)
print(
    "usable episodes (>=2 clip clusters and a local transcript): %d" % len(usable_eps)
)
if not usable_eps:
    print("POSITIVE CONTROL FAILED -- nothing to scan. UNKNOWN, not clean.")
    raise SystemExit(1)

crops = [
    r
    for r in rec
    if r["episode"] in usable_eps and r.get("vector") and r["length_s"] == 30.0
]
print("crops under test: %d" % len(crops))


def speaker_at(turns, t):
    for tn in turns:
        if tn["start"] <= t < tn["start"] + tn["duration_s"]:
            return tn["speaker"]
    return None


def agreement(r):
    hit = miss = nocover = noref = 0
    for c in crops:
        ep = c["episode"]
        t = c["crop_start_s"] * r
        spk = speaker_at(turns_by_ep[ep], t)
        if spk is None:
            nocover += 1
            continue
        cands = {cl: v for (e, cl), v in clipc.items() if e == ep}
        if spk not in cands:
            noref += 1
            continue
        u = unit(c["vector"])
        best = max(cands, key=lambda cl: dot(u, cands[cl]))
        if best == spk:
            hit += 1
        else:
            miss += 1
    return hit, miss, nocover, noref


grid = [
    1.0,
    0.75,
    0.5,
    0.45,
    0.40,
    0.3810,
    0.3628,
    0.3556,
    0.3478,
    0.3404,
    0.3333,
    0.30,
    0.25,
    0.1814,
    0.1667,
]
print("\n     r      hit   miss   acc     (uncovered / no-ref)")
best_r = None
best_acc = -1.0
for r in grid:
    h, m, nc, nr = agreement(r)
    acc = h / (h + m) if h + m else 0.0
    flag = ""
    if acc > best_acc:
        best_acc, best_r = acc, r
    print("  %7.4f  %5d  %5d  %.3f   (%d / %d)%s" % (r, h, m, acc, nc, nr, flag))
print("\nPEAK at r=%.4f with accuracy %.3f" % (best_r, best_acc))
print(
    "  16000/44100 = %.5f   16000/48000 = %.5f   no bug = 1.0"
    % (16000 / 44100, 16000 / 48000)
)
