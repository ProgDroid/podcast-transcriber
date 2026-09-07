---
name: speaker-identity-next-session
description: "Identity lifecycle: phase 1 BUILT and undeployed, phase 2 designed and not built, plus two untouched follow-ups"
metadata:
  node_type: memory
  type: project
  originSessionId: 27309bd2-5b49-424f-acba-3c36276e31a7
  modified: 2026-09-07T11:26:25.346Z
---

Four follow-ups agreed 2026-09-05. **Two of them are now built.** Design of
record: `docs/superpowers/specs/2026-09-05-identity-lifecycle-design.md`
(committed 2026-09-07). Runbook: `docs/operations.md` § "Identity repair, and
the phase gate".

**1 + 2 were designed together and land in two phases.**

**PHASE 1 — BUILT 2026-09-07, NOT DEPLOYED.** Closes item 2 (the two paths
that stored records with no identity and never recovered; review found a
third, `identify_clusters` raising). `identity_version` records that
identification RAN, never what it matched — the distinction that matters,
because 27% of clusters legitimately match nobody and a planner keyed on
`speaker_id` absence never terminates. Ships: the composite key,
`carry_identity` (label-joined, with the subset rule and the conflict
refusal), `stamp_chunks` writing the version, `IDENTIFY_ONLY`,
`repair_identity` (metadata-only, never re-diarises), `plan_repair_updates`'
three refusals, `cap_work`, the absolute backlog plus a **re-measured**
verified count, `stamp_identity_version.py`, and the phase-1 guard in
`transcribe.py`. 384 tests pass.

  **DO NOT DEPLOY WITHOUT THE LANDING ORDER.** `IDENTITY_REPAIR_CAP` ships at
  `0` on purpose. Deploy → one night of classification only → run
  `stamp_identity_version.py` (dry run, then `--apply --expect N`) → check the
  counts → only then raise the cap to 25. Deploying the signal before the
  stamp queues all 433 episodes on night one. **Verified 2026-09-07 against
  the real `speakers/assignments.json`: 970 cluster records, 433 distinct
  episodes, 3 shows (350/76/7), 704 named / 266 not, 0 episodes with zero
  named clusters.**

**PHASE 2 — DESIGNED, NOT BUILT.** Closes item 1 (a stale `speaker_id`
survives a re-write). `replace_identity` deleting before the upsert, `RECHUNK`
plus the `decide_action` reorder, `RECHUNK_CAP`, **the map persisted to the
volume before any delete**, the invariant-2 amendment, and inverting
`test_a_stale_speaker_id_survives_a_re_transcribe`. Two review corrections
that must survive: the exposure is **not** confined to re-transcription (the
mechanism is positional ids + metadata merge, so a `RULES_VERSION` bump and a
torn episode's repair fire it too), and invariant 3 recovers CHUNKS but says
nothing about IDENTITY — hence persisting the map.

  **Two constraints hold until phase 2 lands**, both in `docs/operations.md`:
  do not bump `RULES_VERSION`, and do not force a re-transcribe of an
  already-identified episode.

**3. The ingest path has never executed on real data. RESOLVED 2026-09-07.**
`probe_live_identity.py` (committed, with its result in the docstring) ran the
real chain on JSP Episode 261 (2025-02-28, **44.1 kHz** -- the minority rate
the bug under-exercised). **PASS on every pre-registered number**: 44100 Hz,
14 crops (6/2/6, matching a local pre-computation exactly), `load_embedder`
works in the transcribe image on a T4, identified exactly Jacob Shapiro +
Rob Larity with the third cluster absent as `assignments.json` records, and
**14 of 14 `torchaudio.load` calls observed with 14/14 offsets equal to
`int(start_s * 44100)` and 0 matching the old 16000 arithmetic**. 0 image
rebuild lines, so the probed chain is the deployed one. Covers
`repair_identity` too, which reaches `embedder.embed` through the same
`identify_clusters`. **Does NOT cover** the 48 kHz majority path (arithmetic
is rate-generic; the archive pass exercised it), whisper, diarisation, or the
write path.

  **The reusable lesson:** comparing the NAMES that fall out of the vectors
  proves only that two computations agree — the original bug produced "a
  perfectly plausible vector" from the wrong moment and survived a
  433-episode run that way. Wrapping `torchaudio.load` to record the actual
  `frame_offset` is what made the check able to fail. Confirm no crop starts
  at 0, where correct and buggy arithmetic agree.

**4. `HF_HOME` leaks to the pre-existing transcription path. UNTOUCHED.**
`corpus/voice.py:load_embedder` calls `os.environ.setdefault("HF_HOME", ...)`
before the episode loop, so `DiarizationPipeline` resolves its cache to
`/models`. **Agreed fix: set it explicitly and early in `process_episodes`.**
Note `load_embedder` is now also called from `repair_identity` via
`_load_enrolment`, so the leak has a second caller.

**Why:** every one of these was found by a review that RAN code rather than
read it, and the approaches were argued out once already. Re-deriving them
costs more than the work.

Related: [[ship-2-speaker-grouping-rejected]], [[new-episodes-lack-speaker-id]]
