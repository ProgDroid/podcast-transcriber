import json
from pathlib import Path
from collections import defaultdict, Counter
from corpus.enrolment import build_centroids, assign
from corpus.embedding import weighted_centroid
TH,MG=0.516,0.074
lab=json.loads(Path("speakers/labels.json").read_text(encoding="utf-8"))
cv=json.loads(Path("speakers/clip_vectors_speaker-diarization-community-1.json").read_text(encoding="utf-8"))
rec=json.loads(Path("speakers/archive_vectors.json").read_text(encoding="utf-8"))
C=build_centroids(lab,cv)
g=defaultdict(list)
for r in rec: g[(r["episode"],r["cluster"])].append((r["vector"],r["turn_duration_s"]))
cl={k:weighted_centroid([v for v,_ in it],[w for _,w in it]) for k,it in g.items()}
un=[k for k,v in sorted(cl.items()) if assign(v,C,threshold=TH,margin=MG) is None]
print("unmatched:",len(un))
# independent single-linkage via BFS, not union-find
adj=defaultdict(set)
for i,a in enumerate(un):
    va=cl[a]
    for b in un[i+1:]:
        if sum(x*y for x,y in zip(va,cl[b]))>=TH:
            adj[a].add(b); adj[b].add(a)
seen=set(); groups=[]
for k in un:
    if k in seen: continue
    comp=[]; stack=[k]; seen.add(k)
    while stack:
        n=stack.pop(); comp.append(n)
        for m in adj[n]:
            if m not in seen: seen.add(m); stack.append(m)
    groups.append(comp)
eps=sorted((len({e for e,_ in gr}) for gr in groups), reverse=True)
print("groups:",len(groups))
print("episode-span histogram:", dict(Counter(eps)))
print("groups with >=5 distinct episodes:", sum(1 for c in eps if c>=5))
print("largest span:", eps[0] if eps else 0)
