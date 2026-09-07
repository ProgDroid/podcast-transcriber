---
name: ship-2-speaker-grouping-rejected
description: Ship 2 (archive-wide grouping of unnamed voices) was measured and REJECTED 2026-09-04; do not re-propose without a new histogram
metadata: 
  node_type: memory
  type: project
  originSessionId: 27309bd2-5b49-424f-acba-3c36276e31a7
  modified: 2026-09-04T18:40:41.794Z
---

**Ship 2 — archive-wide clustering of unnamed voices — is REJECTED as of
2026-09-04, on a pre-registered measurement. It is not parked. Do not
re-propose it.**

The checkpoint required **≥15 groups spanning ≥5 distinct episodes**; the
measurement returned **4**. Of 83 local groups over the 266 unmatched
centroids, **62 appear in exactly one episode**. A one-off guest has no "has
the view moved" question, which was the whole case for grouping. The archive
is a small known cast plus single-appearance guests.

The 127-episode group is not a counterexample — it survived the sample-rate
fix unchanged (112 before), the signature of non-speech content (music bed,
intro sting, room tone).

**Re-open on exactly one trigger:** a measured histogram, on correct vectors,
clearing ≥15 groups at ≥5 episodes — which would need feed composition to
change materially, e.g. a new show with a rotating cast. Do **not** re-open
on a better clustering algorithm, a new embedding model, the archive merely
growing, or the fact that 266 clusters stay unidentified (that residual is
measured and expected).

**Why:** the two-ship structure existed to buy this answer for the price of
Ship 1 rather than the price of both, and it worked. Re-litigating it throws
away what the split was for. Note also that the *first* run of this
checkpoint returned "1 qualifying group" from vectors corrupted by
[[speaker-embed-sample-rate-bug]]; that figure was void and was retracted
rather than banked, even though it pointed the same way as the real one.

**How to apply:** if grouping comes up again, point at spec §1.3 in
`docs/superpowers/specs/2026-09-02-speaker-grouping-design.md`, which holds
the full histogram and the re-open trigger. What survives and should be kept:
`corpus/names.py`'s `grp_NNN`-shaped indirection and `display_speaker`'s four
states, so the render path needs no reshaping if this is ever revisited.

Related: [[speaker-id-tier1-failed]], [[new-episodes-lack-speaker-id]]
