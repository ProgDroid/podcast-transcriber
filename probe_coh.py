import json, math, statistics
from collections import defaultdict
from pathlib import Path
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def unit(v):
    n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v]
rec = json.loads(Path("speakers/archive_vectors.json").read_text(encoding="utf-8"))
# within-cluster coherence at a FIXED length (removes the 10/30 factor)
by = defaultdict(dict)
for r in rec:
    if r["length_s"]==30.0:
        by[(r["episode"],r["cluster"])][r["draw"]] = r["vector"]
sims=[]
for k,d in by.items():
    ds=sorted(d)
    for i in range(len(ds)):
        for j in range(i+1,len(ds)):
            sims.append(dot(unit(d[ds[i]]),unit(d[ds[j]])))
sims.sort()
print("PROBE I: within-ARCHIVE-cluster, draw-vs-draw cosine at 30s. n=%d" % len(sims))
print("  min %.4f p05 %.4f median %.4f p95 %.4f max %.4f" % (sims[0],sims[int(.05*len(sims))],statistics.median(sims),sims[int(.95*len(sims))],sims[-1]))
print("  frac below 0.516: %.3f" % (sum(1 for s in sims if s<0.516)/len(sims)))

# cross-cluster, same episode, 30s draw 0 -- the contrast
cross=[]
epc=defaultdict(dict)
for r in rec:
    if r["length_s"]==30.0 and r["draw"]==0:
        epc[r["episode"]][r["cluster"]]=r["vector"]
for ep,cd in epc.items():
    cs=sorted(cd)
    for i in range(len(cs)):
        for j in range(i+1,len(cs)):
            cross.append(dot(unit(cd[cs[i]]),unit(cd[cs[j]])))
cross.sort()
print("\n  contrast: DIFFERENT clusters, same episode, 30s d0. n=%d median %.4f p95 %.4f" % (len(cross),statistics.median(cross),cross[int(.95*len(cross))]))
