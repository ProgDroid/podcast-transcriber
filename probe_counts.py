import json, math
from pathlib import Path
from collections import Counter, defaultdict
from corpus.enrolment import build_centroids, assign, co_membership_conflicts
from corpus.embedding import weighted_centroid
from corpus.identity import parse_transcript_filename
TH,MG=0.516,0.074
lab=json.loads(Path("speakers/labels.json").read_text(encoding="utf-8"))
cv=json.loads(Path("speakers/clip_vectors_speaker-diarization-community-1.json").read_text(encoding="utf-8"))
rec=json.loads(Path("speakers/archive_vectors.json").read_text(encoding="utf-8"))
C=build_centroids(lab,cv)
g=defaultdict(list)
for r in rec: g[(r["episode"],r["cluster"])].append((r["vector"],r["turn_duration_s"]))
cl={k:weighted_centroid([v for v,_ in it],[w for _,w in it]) for k,it in g.items()}
A=[{"episode":e,"cluster":c,"person":assign(v,C,threshold=TH,margin=MG)} for (e,c),v in sorted(cl.items())]
m=[a for a in A if a["person"]]
print("INDEPENDENT RECOMPUTE")
print("  clusters: %d ; assigned: %d ; unassigned: %d" % (len(A),len(m),len(A)-len(m)))
print("  per person:", dict(Counter(a["person"] for a in m).most_common()))
print("  conflicts: %d (%.1f%% of matched)" % (len(co_membership_conflicts(A)), 100*len(co_membership_conflicts(A))/len(m)))
S=json.loads(Path("speakers/assignments.json").read_text(encoding="utf-8"))
print("\nFROM speakers/assignments.json")
print("  threshold/margin:", S["threshold"], S["margin"])
sa=S["assignments"]; sm=[a for a in sa if a["person"]]
print("  clusters: %d ; assigned: %d ; unassigned: %d" % (len(sa),len(sm),len(sa)-len(sm)))
print("  per person:", dict(Counter(a["person"] for a in sm).most_common()))
print("  conflicts stored: %d" % len(S["conflicts"]))
print("  checkpoint:", S["checkpoint"])
print("\n  MATCH:" , len(sa)==len(A) and len(sm)==len(m))
