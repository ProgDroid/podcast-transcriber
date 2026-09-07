---
name: speaker-identity-next-session
description: "Identity lifecycle: phase 1 BUILT and undeployed, phase 2 designed and not built, plus two untouched follow-ups"
metadata:
  node_type: memory
  type: project
  originSessionId: 27309bd2-5b49-424f-acba-3c36276e31a7
  modified: 2026-09-07T19:02:45.741Z
---

Four follow-ups agreed 2026-09-05. **Two of them are now built.** Design of
record: `docs/superpowers/specs/2026-09-05-identity-lifecycle-design.md`
(committed 2026-09-07). Runbook: `docs/operations.md` § "Identity repair, and
the phase gate".

**1 + 2 were designed together and land in two phases.**

**PHASE 1 — BUILT AND DEPLOYED 2026-09-07 (`podcast-transcriber` v7, 18:21
UTC), landing-order step 1 of 4 done.** Closes item 2 (the two paths
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

  **THE LANDING ORDER IS HALF-RUN — DO NOT SKIP AHEAD.**
  `IDENTITY_REPAIR_CAP` shipped at `0` as intended, so the deployed run
  classifies and reports and acts on nothing.

  **PRE-REGISTERED 2026-09-07, BEFORE THE RUN — the numbers to check the
  2026-09-08 09:00 UTC output against.** `scheduled_job` calls
  `transcribe.remote` once per show, so expect THREE backlog lines, not one
  aggregate:

  | Show | `IDENTIFY_ONLY` | `TRANSCRIBE` |
  |---|---|---|
  | The Jacob Shapiro Podcast | **349** | 1 (ep 354) |
  | Geopolitical Cousins | **76** | 1 (ep 77) |
  | The Observing Japan Podcast | **7** | 0 |
  | total | **432** | 2 |

  **432, NOT 433**, for two reasons that a re-derivation must reproduce:
  (a) **JSP episode 7 has rolled off the feed** — measured from the live
  feed, whose oldest entry is now ep 8 (2022-06-03). A stamped-but-
  unreachable episode can never be classified. (b) The two episodes
  published **2026-09-07 12:30 UTC** arrived AFTER that day's run, so they
  are `TRANSCRIBE`, and v7's ingest path stamps `identity_version`
  directly — they never enter the backlog. Tells: `433` = ep 7 is being
  counted; `435` = both corrections wrong; a SMALL number = the planner is
  not reading the key (the runbook's main warning); non-zero `EMBED_ONLY`
  pulls `IDENTIFY_ONLY` down by that amount and still reconciles.
  UNVERIFIED assumption, self-checking in the same output: that all 432 are
  complete and rules-current.

  Then → `stamp_identity_version.py` (dry run, then `--apply --expect N`),
  **still `--expect 433`**: selection is by triple against the STORE, not
  the feed, so the roll does not touch it and ep 7 is still stamped.
  Then → **the night after the stamp should be `0`, not "the post-boundary
  episodes"** as the runbook says: that wording predates this deploy, and
  because v7 went live BEFORE those two episodes were ingested they get
  stamped at ingest, while nothing was published between the archive
  boundaries and 2026-09-07. `432` there means the stamp did not land.
  → only then raise the cap to 25. Raising the cap before the stamp queues
  all 433 episodes on night one.

  To re-derive the feed side (the runbook's "enumerated from the feeds
  BEFORE the run"): the `uv` env has NO `feedparser`; use system `py` with
  `PYTHONPATH=G:/audio-transcription/via-modal`, importing the planner's own
  `parse_all_episodes` and the stamp script's own `episode_triples` so
  neither number is a lookalike. The control that caught the ep-7 roll:
  assert feed-episodes-at-or-before-boundary EQUALS the archive count per
  show (350/76/7), not merely that the totals look right. **Verified 2026-09-07 against
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

**4. `HF_HOME` leaked to the transcription path. RESOLVED 2026-09-07.**
Both GPU entrypoints (`process_episodes` AND `repair_identity`, which became a
second caller the same day) now set `os.environ["HF_HOME"] = MODEL_PATH`
explicitly and early. **The real defect was non-determinism, not the value:**
`load_embedder`'s `setdefault` runs before the pyannote import and before
anything that can raise, so a missing enrolment mount raised BEFORE it and
left `HF_HOME` unset, while a failed model download raised AFTER it had run --
both printing the identical `ENROLMENT UNAVAILABLE` line. Diarisation's cache
root depended on *where inside an unrelated subsystem* a failure happened.
`setdefault` is deliberately KEPT: it is what stops the call overwriting the
entrypoint's choice, and `speaker_embed.py` still relies on it. Pinned by
`tests/test_load_embedder_env.py`.

**Why:** every one of these was found by a review that RAN code rather than
read it, and the approaches were argued out once already. Re-deriving them
costs more than the work.

Related: [[ship-2-speaker-grouping-rejected]], [[new-episodes-lack-speaker-id]]
