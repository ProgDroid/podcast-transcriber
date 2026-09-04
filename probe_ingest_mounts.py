"""THROWAWAY PROBE -- do the enrolment mounts reach the transcribe container?

The nightly ingest now derives its five enrolment centroids from two files
mounted into the container at startup. If either mount is missing, the
`except` around that block does exactly what it was written to do: the batch
transcribes and stores WITHOUT `speaker_id`, loudly in the log but with the
corpus still growing. That is the right failure, and it is also the failure
nobody would notice for a week.

So it is worth one CPU-only run to find out now.

This imports `image` FROM transcribe.py rather than restating the chain, so a
drift between what is probed and what runs is impossible. It does not need a
GPU, the model volume, or the HF token -- the embedder load path is already
exercised by the archive pass; the mounts are the new thing.

PRE-REGISTERED, before running:

  * both mounted paths exist                    True
  * bogus-path control                          False  (proves the existence
                                                 check discriminates rather
                                                 than returning True)
  * build_centroids yields exactly              5 people
  * and they are                                Jacob Shapiro, Marko Papic,
                                                Matt Gertken, Rob Larity,
                                                Tobias Harris
  * every centroid's dimension                  256
  * image built without rebuild lines           (add_local_file(copy=False)
                                                mounts at startup and must
                                                not change the content
                                                address)

Run:  modal run probe_ingest_mounts.py
"""

from __future__ import annotations

import modal

from transcribe import ENROLMENT_LABELS, ENROLMENT_VECTORS, image

# The probe imports transcribe for the REAL image object, so the container
# needs that module too -- transcribe's own image mounts only `corpus`.
probe_image = image.add_local_python_source("transcribe")

app = modal.App("probe-ingest-mounts")

EXPECTED = [
    "Jacob Shapiro",
    "Marko Papic",
    "Matt Gertken",
    "Rob Larity",
    "Tobias Harris",
]


@app.function(image=probe_image, timeout=600)
def check() -> dict:
    import json
    import os

    from corpus.enrolment import MARGIN, THRESHOLD, build_centroids

    present = {p: os.path.exists(p) for p in (ENROLMENT_LABELS, ENROLMENT_VECTORS)}
    # Control: the same check on a path nobody mounted must come back False.
    present["/enrolment/__nothing__.json"] = os.path.exists(
        "/enrolment/__nothing__.json"
    )
    print(f"paths: {present}")

    if not all(present[p] for p in (ENROLMENT_LABELS, ENROLMENT_VECTORS)):
        return {"present": present, "people": [], "dims": []}

    with open(ENROLMENT_LABELS) as handle:
        labels = json.load(handle)
    with open(ENROLMENT_VECTORS) as handle:
        clip_vectors = json.load(handle)

    centroids = build_centroids(labels, clip_vectors)
    people = sorted(centroids)
    dims = sorted({len(v) for v in centroids.values()})
    print(f"labels={len(labels)} clip_vectors={len(clip_vectors)}")
    print(f"enrolled={people}")
    print(f"centroid dims={dims}")
    print(f"tuning: threshold={THRESHOLD} margin={MARGIN}")
    print("CHECK COMPLETE")  # terminal marker
    return {"present": present, "people": people, "dims": dims}


@app.local_entrypoint()
def main() -> None:
    result = check.remote()
    present = result["present"]

    checks = [
        ("labels mounted", present.get(ENROLMENT_LABELS) is True),
        ("clip vectors mounted", present.get(ENROLMENT_VECTORS) is True),
        (
            "bogus-path control is False",
            present.get("/enrolment/__nothing__.json") is False,
        ),
        ("five people enrolled", result["people"] == EXPECTED),
        ("centroids are 256-dim", result["dims"] == [256]),
    ]
    print()
    for label, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
    print("ALL PASS" if all(ok for _, ok in checks) else "SOME CHECKS FAILED")
