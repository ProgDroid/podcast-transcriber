import json, math
from pathlib import Path

rec = json.loads(Path("speakers/archive_vectors.json").read_text(encoding="utf-8"))
print("archive_vectors: type", type(rec).__name__, "len", len(rec))
print("first record keys:", sorted(rec[0].keys()) if isinstance(rec, list) else "n/a")
r0 = rec[0]
for k, v in r0.items():
    if isinstance(v, list):
        print(f"  {k}: list len={len(v)} first3={v[:3]}")
    else:
        print(f"  {k}: {v!r}")

nvec = sum(1 for r in rec if r.get("vector") is not None)
print("records with vector:", nvec, "of", len(rec))
dims = {len(r["vector"]) for r in rec if r.get("vector")}
print("vector dims present:", dims)
lens = {}
for r in rec:
    lens[r.get("length_s")] = lens.get(r.get("length_s"), 0) + 1
print("length_s histogram:", lens)

norms = [math.sqrt(sum(x*x for x in r["vector"])) for r in rec[:200] if r.get("vector")]
print("archive raw vector norms sample: min %.4f max %.4f mean %.4f" % (min(norms), max(norms), sum(norms)/len(norms)))

cv = json.loads(Path("speakers/clip_vectors_speaker-diarization-community-1.json").read_text(encoding="utf-8"))
print("clip_vectors: len", len(cv))
k0 = next(iter(cv))
print("sample clip key:", repr(k0), "dim", len(cv[k0]))
cn = [math.sqrt(sum(x*x for x in v)) for v in list(cv.values())[:200]]
print("clip raw norms: min %.4f max %.4f mean %.4f" % (min(cn), max(cn), sum(cn)/len(cn)))

lab = json.loads(Path("speakers/labels.json").read_text(encoding="utf-8"))
print("labels:", len(lab))
lk = next(iter(lab))
print("sample label:", repr(lk), lab[lk])
