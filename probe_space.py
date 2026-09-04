import json, math, statistics
from pathlib import Path
from collections import defaultdict

def unit(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v]
def cos(a,b):
    return sum(x*y for x,y in zip(unit(a),unit(b)))

rec = json.loads(Path("speakers/archive_vectors.json").read_text(encoding="utf-8"))
cv  = json.loads(Path("speakers/clip_vectors_speaker-diarization-community-1.json").read_text(encoding="utf-8"))
plan= json.loads(Path("speakers/clip_plan.json").read_text(encoding="utf-8"))
print("clip_plan type", type(plan).__name__, "len", len(plan))
p0 = plan[0] if isinstance(plan, list) else plan[next(iter(plan))]
print("clip_plan sample:", json.dumps(p0)[:400])

# --- PROBE A: same turn, 10s vs 30s ---
by_turn = defaultdict(dict)
for r in rec:
    by_turn[(r["episode"], r["cluster"], r["draw"])][r["length_s"]] = r["vector"]
pairs = [(d[10.0], d[30.0]) for d in by_turn.values() if 10.0 in d and 30.0 in d]
sims = [cos(a,b) for a,b in pairs]
sims.sort()
print("\nPROBE A: 10s vs 30s cosine, same turn. n=%d" % len(sims))
print("  min %.4f p05 %.4f median %.4f p95 %.4f max %.4f" % (
    sims[0], sims[int(.05*len(sims))], statistics.median(sims), sims[int(.95*len(sims))], sims[-1]))
print("  fraction below 0.516 threshold: %.3f" % (sum(1 for s in sims if s < 0.516)/len(sims)))

# --- PROBE B: clip vector vs archive vector, SAME (episode,cluster,draw) ---
# clip key = episode stem + '--' + cluster + '--d' + draw
arch_by_key = {}
for r in rec:
    stem = r["episode"][:-4] if r["episode"].endswith(".txt") else r["episode"]
    arch_by_key[(stem, r["cluster"], r["draw"], r["length_s"])] = r["vector"]
matched = 0; simsB10=[]; simsB30=[]
for ck, cvec in cv.items():
    parts = ck.split("--")
    if len(parts) != 3: continue
    stem, cl, d = parts[0], parts[1], int(parts[2][1:])
    a10 = arch_by_key.get((stem, cl, d, 10.0)); a30 = arch_by_key.get((stem, cl, d, 30.0))
    if a10 is not None:
        matched += 1; simsB10.append(cos(cvec,a10))
    if a30 is not None:
        simsB30.append(cos(cvec,a30))
print("\nPROBE B: clip vector vs archive vector, same (episode,cluster,draw)")
print("  clip keys parsed: %d ; matched to a 10s archive crop: %d" % (len(cv), matched))
for nm, s in (("10s", simsB10), ("30s", simsB30)):
    if s:
        s = sorted(s)
        print("  %s: n=%d min %.4f median %.4f max %.4f  frac>0.9: %.3f" % (
            nm, len(s), s[0], statistics.median(s), s[-1], sum(1 for x in s if x>0.9)/len(s)))
    else:
        print("  %s: NO OVERLAP (n=0)" % nm)
