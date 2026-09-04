import json, math, statistics
from pathlib import Path
from collections import defaultdict, Counter
from corpus.enrolment import build_centroids, score, assign
from corpus.embedding import weighted_centroid

TH, MG = 0.516, 0.074
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def unit(v):
    n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v]

lab = json.loads(Path("speakers/labels.json").read_text(encoding="utf-8"))
cv  = json.loads(Path("speakers/clip_vectors_speaker-diarization-community-1.json").read_text(encoding="utf-8"))
rec = json.loads(Path("speakers/archive_vectors.json").read_text(encoding="utf-8"))
C   = build_centroids(lab, cv)

# archive cluster centroids, recomputed INDEPENDENTLY of assign_speakers.py
g = defaultdict(list)
for r in rec:
    g[(r["episode"], r["cluster"])].append((r["vector"], r["turn_duration_s"]))
clusters = {k: weighted_centroid([v for v,_ in it],[w for _,w in it]) for k,it in g.items()}
print("clusters with a centroid (independent recompute):", len(clusters))

# ground truth per (episode, cluster) from voice labels
truth = defaultdict(set)
for cid, l in lab.items():
    if l.get("basis")!="voice" or not l.get("person"): continue
    stem, cl, _ = cid.split("--")
    truth[(stem+".txt", cl)].add(l["person"])
single = {k: next(iter(v)) for k,v in truth.items() if len(v)==1}
print("labelled (episode,cluster) pairs:", len(truth), " unambiguous:", len(single))
present = [k for k in single if k in clusters]
print("of those, present in archive clusters:", len(present), " ABSENT:", len(single)-len(present))

ok=refuse=wrong=0; conf=Counter(); wrong_ex=[]
for k in sorted(present):
    p = assign(clusters[k], C, threshold=TH, margin=MG)
    t = single[k]
    if p is None: refuse+=1
    elif p==t: ok+=1
    else:
        wrong+=1; conf[(t,p)]+=1
        if len(wrong_ex)<10: wrong_ex.append((k,t,p,score(clusters[k],C)[:2]))
print("\nPROBE F: assignment of LABELLED archive clusters (ground truth = voice labels)")
print("  correct=%d  refused(None)=%d  WRONG=%d  of %d" % (ok,refuse,wrong,len(present)))
print("  precision on decided = %.3f" % (ok/(ok+wrong) if ok+wrong else 0))
for (t,p),n in conf.most_common(): print("    %s -> %s : %d" % (t,p,n))
for k,t,p,r in wrong_ex: print("    e.g. %s %s: true=%s pred=%s top2=%s" % (k[0],k[1],t,p,[(a,round(b,3)) for a,b in r]))

# --- Episode 100 reproduction ---
print("\nPROBE G: The Jacob Shapiro Podcast - Episode 100")
ep = [k for k in clusters if k[0].startswith("The Jacob Shapiro Podcast - Episode 100 -")]
print("  matching episode keys:", sorted({k[0] for k in ep}))
for k in sorted(ep):
    r = score(clusters[k], C)
    a = assign(clusters[k], C, threshold=TH, margin=MG)
    print("  %s -> assigned=%s  scores=%s" % (k[1], a, [(p,round(s,3)) for p,s in r]))
    print("     label truth:", single.get(k, "(unlabelled)"))
