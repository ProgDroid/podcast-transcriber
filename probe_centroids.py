import json, math, statistics
from pathlib import Path
from collections import defaultdict, Counter
from corpus.enrolment import build_centroids, score

def unit(v):
    n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b): return dot(unit(a),unit(b))

lab = json.loads(Path("speakers/labels.json").read_text(encoding="utf-8"))
cv  = json.loads(Path("speakers/clip_vectors_speaker-diarization-community-1.json").read_text(encoding="utf-8"))
rec = json.loads(Path("speakers/archive_vectors.json").read_text(encoding="utf-8"))

voice = {k:v for k,v in lab.items() if v.get("basis")=="voice" and v.get("person")}
print("voice labels:", len(voice), " overlap with clip_vectors:", len(set(voice)&set(cv)))
per = Counter(v["person"] for k,v in voice.items() if k in cv)
print("enrolment clips per person:", dict(per))

C = build_centroids(lab, cv)
print("centroids:", sorted(C))
print("centroid norms:", {p: round(math.sqrt(dot(v,v)),6) for p,v in C.items()})

# --- own-clip vs own-centroid and cross, LEAVE-ONE-OUT ---
print("\nPROBE C: per-person clip scores vs each centroid (leave-one-out on own centroid)")
people = sorted(C)
rows = defaultdict(lambda: defaultdict(list))
for cid, l in sorted(voice.items()):
    if cid not in cv: continue
    p = l["person"]
    rest = {k:v for k,v in voice.items() if k != cid}
    Cl = build_centroids(rest, cv)
    for person, s in score(cv[cid], Cl):
        rows[p][person].append(s)
hdr = "  %-16s " % r"clips of \ vs" + " ".join("%-16s" % q for q in people)
print(hdr)
for p in people:
    cells = []
    for q in people:
        vals = rows[p][q]
        cells.append("%-16s" % ("%.3f (n=%d)" % (statistics.median(vals), len(vals)) if vals else "-"))
    print("  %-16s " % p + " ".join(cells))

# --- centroid vs centroid ---
print("\nPROBE D: centroid-to-centroid cosine")
for i,p in enumerate(people):
    for q in people[i+1:]:
        print("  %-16s %-16s %.4f" % (p,q,dot(C[p],C[q])))

# --- global mean attractor test ---
gm = unit([sum(c)/len(rec) for c in zip(*[unit(r["vector"]) for r in rec])])
print("\nPROBE E: cosine of each centroid to the GLOBAL MEAN direction of all 5424 archive vectors")
for p in people:
    print("  %-16s %.4f   (enrolment clips: %d)" % (p, dot(C[p], gm), per[p]))
print("  median cosine of an archive vector to global mean: %.4f" %
      statistics.median([dot(unit(r["vector"]), gm) for r in rec]))
