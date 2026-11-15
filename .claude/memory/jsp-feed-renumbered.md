---
name: jsp-feed-renumbered
description: JSP feed shifted itunes:episode +72 (~2026-11-12); planner re-transcribed old episodes nightly until the 2h timeout killed the run; guard + GUID rename migration chosen 2026-11-15
metadata:
  node_type: memory
  type: project
  originSessionId: db957515-d2dc-4e31-a5e0-811e99d1b017
  modified: 2026-11-15T12:38:57.223Z
---

**Measured 2026-11-15.** The Jacob Shapiro feed renumbered its whole back catalogue: `<itunes:episode>` +72 (298 entries; 3 showed +73, same-day pairs where the probe picked the wrong sibling). Feed entries 350; volume had 414 JSP files; 301 feed (ep, date) triples had no transcript file, 49 did.

**Failure chain:** filename-keyed `transcript_exists` is False → `TRANSCRIBE` → `process_episodes` re-transcribes ~13/night → `scheduled_job`/`transcribe`/`process_episodes` all hit `timeout=7200` at 11:00 UTC (the three cancellations in the email). Cousins runs first (fine), Observing Japan last (starved). Nov 14 logs had already expired from Modal by Nov 15 (retention ~1 day) — UNKNOWN, inferred from volume state.

**Corpus NOT duplicated:** the GUID survived the renumber, and `upsert_then_prune`'s guid arm (`corpus/writing.py`) deleted the old-number records when the new copy landed. Chroma holds one numbering per episode; the volume holds both files for the ~49 re-done ones.

**RESOLVED 2026-11-15 (he chose guard, then migrate).** Guard `guard_transcribe` / `TRANSCRIBE_ANOMALY_THRESHOLD = 10` deployed as `podcast-transcriber` **v9** (12:16 UTC); verified live: JSP refused (299), no GPU container. Then `migration/renumber.py` applied after a fresh backup (`chroma-backups` `2026-11-15T1227Z`, 30,550 records): every pre-registered number held — 299 renames (15,313 records), 0 new, 0 unresolved, 46 orphans (old 317–363 + `Unknown 2026-11-11`) moved to `/renumbered/`; count 30,550 before and after; volume 368 top-level JSP / 345 archived; deployed planner re-plan `0 to transcribe, 0 to re-embed`, backlog 1. **Follow-up the agent owns:** confirm the 2026-11-16 09:00 UTC nightly succeeds on all three shows (no email). Runbook: `docs/operations.md` § `TRANSCRIBE REFUSED`. Code was uncommitted at the end of the session unless a later commit says otherwise.

**How to apply:** a large `N to transcribe` on an established show means a feed-shape change, not new episodes. Check the feed's numbering against the volume before anything else. Probe used: `modal volume ls` (needs `MSYS_NO_PATHCONV=1`) vs `corpus.feed.episode_number_of` over the live feed.

Related: [[speaker-identity-next-session]], [[downloaded-cache-goes-stale]]
