---
name: new-episodes-lack-speaker-id
description: CLOSED 2026-09-04 — ingest now assigns speaker_id itself; the archive is complete up to then and new episodes are covered from deploy 1f9bb83
metadata: 
  node_type: memory
  type: project
  originSessionId: 27309bd2-5b49-424f-acba-3c36276e31a7
  modified: 2026-09-04T21:20:54.260Z
---

**This gap is CLOSED as of 2026-09-04 (commit `1f9bb83`, deployed).** Kept
rather than deleted because the *shape* of the bug is worth recognising
again, and because the boundary date still matters when reading old results.

**What it was.** `speaker_id` was written only by `backfill_speaker_id.py`, a
manual one-off. The nightly ingest never wrote it. The MCP speaker filter is
`{"speaker_id": {"$eq": name}}`, and **Chroma's `$eq` does not match a record
missing the key** — so every episode after the backfill would have dropped
out of speaker-filtered queries silently, with no error, returning older
results that looked correct. At ~2.3 episodes/week that is roughly a quarter
of the corpus per year.

**What fixed it.** `corpus/identify.py` (`identify_clusters`, `stamp_chunks`)
wired into `transcribe.py`, deciding identity while the audio is still on
disk and stamping chunks before the single upsert. Far cheaper than the batch
pass because the expensive part — fetching and decoding audio — is already
paid for by transcription. The five enrolment centroids are derived per
container from `speakers/labels.json` + `speakers/clip_vectors_*.json`,
mounted with `add_local_file(copy=False)` so the image content address (and
therefore the pinned CUDA/whisperx versions) is unchanged.

**Boundary that still matters:** chunks written on or before 2026-09-04 got
`speaker_id` from the backfill; chunks after it get it at ingest. Both use
threshold 0.516 / margin 0.074 from `corpus/enrolment.py`, so they mean the
same thing.

**Why:** the failure mode here is the one this project keeps hitting — a
wrong answer that looks exactly like a right one. Worth re-reading before
adding any new metadata key that a query filters on: **a filter over a key
some records lack silently narrows the corpus instead of erroring.**

**How to apply / still unverified:** identity failure never loses an episode
— it is stored without `speaker_id` and named in the run's failure list.
**Nobody has yet confirmed a real nightly run stamps correctly**; the next
check is to read Chroma back for an episode dated after 2026-09-04 and
confirm it carries `speaker_id` (`probe_verify_backfill.py` is the pattern).
One unmeasured risk: the embedder now loads alongside the whisperx models on
a T4, so watch the first runs for GPU memory pressure.

Related: [[ship-2-speaker-grouping-rejected]], [[chroma-v2-collection-cutover]]
