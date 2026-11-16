---
name: jsp-feed-renumbered
description: JSP feed shifted itunes:episode +72 (~2026-11-12); planner re-transcribed old episodes nightly until the 2h timeout killed the run; guard (v10) + GUID rename migration + assignments remap all shipped 2026-11-15; Nov 16 nightly CONFIRMED clean from logs
metadata:
  node_type: memory
  type: project
  originSessionId: db957515-d2dc-4e31-a5e0-811e99d1b017
  modified: 2026-11-15T14:05:39.427Z
---

**Measured 2026-11-15.** The Jacob Shapiro feed renumbered its whole back catalogue: `<itunes:episode>` +72 (298 entries; 3 showed +73, same-day pairs where the probe picked the wrong sibling). Feed entries 350; volume had 414 JSP files; 301 feed (ep, date) triples had no transcript file, 49 did.

**Failure chain:** filename-keyed `transcript_exists` is False → `TRANSCRIBE` → `process_episodes` re-transcribes ~13/night → `scheduled_job`/`transcribe`/`process_episodes` all hit `timeout=7200` at 11:00 UTC (the three cancellations in the email). Cousins runs first (fine), Observing Japan last (starved). Nov 14 logs had already expired from Modal by Nov 15 (retention ~1 day) — UNKNOWN, inferred from volume state.

**Corpus NOT duplicated:** the GUID survived the renumber, and `upsert_then_prune`'s guid arm (`corpus/writing.py`) deleted the old-number records when the new copy landed. Chroma holds one numbering per episode; the volume holds both files for the ~49 re-done ones.

**RESOLVED 2026-11-15 (the user chose guard, then migrate).** Guard `guard_transcribe` / `TRANSCRIBE_ANOMALY_THRESHOLD = 10` deployed as `podcast-transcriber` **v9** (12:16 UTC), redeployed **v10** (12:42) from clean commit 5aef228; verified live: JSP refused (299), no GPU container. Then `migration/renumber.py` applied after a fresh backup (`chroma-backups` `2026-11-15T1227Z`, 30,550 records): every pre-registered number held — 299 renames (15,313 records), 0 new, 0 unresolved, 46 orphans (old 317–363 + `Unknown 2026-11-11`) moved to `/renumbered/`; count 30,550 before and after; volume 368 top-level JSP / 345 archived; deployed planner re-plan `0 to transcribe, 0 to re-embed`, backlog 1. **Follow-up the agent owns:** confirm the 2026-11-16 09:00 UTC nightly succeeds on all three shows (no email). Runbook: `docs/operations.md` § `TRANSCRIBE REFUSED`. Committed and pushed: 24ee12b (guard), 727108a (migration), bf6b3e7 (assignments remap); CI green. The guard also refuses hand-runs (it lives inside `transcribe`): adding a show means raising the threshold in that commit and lowering it after.

**Nov 16 nightly CONFIRMED from same-day logs (v10):** all three shows planned `0 to transcribe, 0 to re-embed`, no GPU container, each printed `All done.`; 09:00:02 to 09:02:23 UTC. JSP: backlog 1 (`IDENTIFY_ONLY` ep 403), EXCLUDE 345/343 + two Unknowns. The user reported 'it failed again' that day, but Modal logs for transcriber, MCP server (only cold-start 400-then-200 handshakes) and backup (Sunday-only, nothing expected) showed no failure: the alert was a Modal email: `scheduled_job` timed out. Modal's call graph says today's run did NOT: root call `fc-01M7HSBC4QTJ120GKNVA7W2YZQ` (scheduled_job) status SUCCESS, `.get()` returned None, all three transcribe children SUCCESS. So the email names an earlier run (most likely Nov 15 09:00 on v8, before the 12:16 guard), unconfirmed until its timestamp is read.

**`speakers/assignments.json` remapped too (2026-11-15, `migration/remap_assignments.py`):** it still keyed 338/350 JSP episodes by old numbers. Remapped 334 (each proven by archived body == new body, never by +72), dropped 0; afterwards 346 join the store + 4 excluded = 350. `labels.json` and the clip vectors deliberately keep OLD names: they join only each other by clip id (the live enrolment path), never the store.

**Measured, and it contradicts "diarisation is not deterministic":** all 46 episodes re-transcribed on 2026-11-14/15 reproduced the old transcript BYTE FOR BYTE, SPEAKER_XX labels included (same image, T4, audio; pyannote disables TF32). I had predicted their labels would be stale. Holds for same image + GPU; do not assume across versions or hardware, and keep text checks rather than trusting it.

**How to apply:** a large `N to transcribe` on an established show means a feed-shape change, not new episodes. Check the feed's numbering against the volume before anything else. Probe used: `modal volume ls` (needs `MSYS_NO_PATHCONV=1`) vs `corpus.feed.episode_number_of` over the live feed.

Related: [[speaker-identity-next-session]], [[downloaded-cache-goes-stale]]
