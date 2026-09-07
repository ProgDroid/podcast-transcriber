---
name: downloaded-cache-goes-stale
description: downloaded/ is a gitignored cache of the Modal volume that silently goes stale; refresh it before computing anything from it
metadata: 
  node_type: memory
  type: project
  originSessionId: b1d639af-de60-496e-986e-aae48020777f
  modified: 2026-09-02T08:55:44.479Z
---

`downloaded/` is a gitignored local copy of the `podcast-transcripts` Modal volume, not a fixed dataset. On 2026-09-02 it turned out to be a snapshot from ~2026-05-06 — 39 episodes behind the volume's 439 — and figures derived from it had been quoted in a design spec as current for months.

Refresh with `MSYS_NO_PATHCONV=1 modal volume get podcast-transcripts / downloaded --force`. The env var is not optional under Git Bash: without it the bare `/` is rewritten to a Windows path and the command fails with a bare "No such file or directory" naming neither the path nor the cause.

Show composition as of 2026-09-02: The Jacob Shapiro Podcast 356, Geopolitical Cousins 76, The Observing Japan Podcast 7. **Observing Japan is dormant, not broken** — its feed returns exactly 7 items, newest 2026-06-19, matching the transcripts one-for-one, and one of the 7 is a trailer. A small OJ count is expected and is not a pipeline fault; do not go looking for dropped episodes.

**Why:** the staleness was invisible because nothing recorded when the cache was taken, and every figure derived from it looked entirely plausible. It also concealed a real change — the corpus has drifted toward more speakers and shorter turns since May 2026.

**How to apply:** refresh before any measurement, and make whatever you produce state its own inputs. `speaker_stats.py` prints `dir=`, `files=` and `gap_cap=` on every run for exactly this reason. See [[chroma-v2-collection-cutover]] for the other place this corpus's state lives.
