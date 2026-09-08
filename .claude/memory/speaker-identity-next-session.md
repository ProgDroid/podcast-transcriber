---
name: speaker-identity-next-session
description: "Identity lifecycle: phase 1 deployed and step 1 verified 2026-09-08 (backlog 428, not the pre-registered 432); stamp dry run done, apply is --expect 28644; phase 2 designed not built"
metadata:
  node_type: memory
  type: project
  originSessionId: 27309bd2-5b49-424f-acba-3c36276e31a7
  modified: 2026-09-08T15:04:59.766Z
---

Four follow-ups agreed 2026-09-05. **Two of them are now built.** Design of
record: `docs/superpowers/specs/2026-09-05-identity-lifecycle-design.md`
(committed 2026-09-07). Runbook: `docs/operations.md` § "Identity repair, and
the phase gate".

**1 + 2 were designed together and land in two phases.**

**PHASE 1 — BUILT AND DEPLOYED 2026-09-07 (`podcast-transcriber` v7, 18:21
UTC). Landing-order steps 1 AND 2 of 4 done (2026-09-08); NEXT IS STEP 3,
confirm the next night's plan is `0` on all three shows, and only then step
4, the cap bump.** Closes item 2 (the two paths
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

  **CHECKED 2026-09-08. STEP 1 CONFIRMED** — the run landed on all three
  shows, the cap held (`repairing 0 this run`), and no `IDENTITY NOT
  CERTIFIED` / `STAMP JOIN BROKEN` / `REFUSED` / `CONFLICT` / traceback
  appears anywhere. The planner IS reading the key; the runbook's
  small-number warning did not fire. THREE backlog lines, not one aggregate,
  because `scheduled_job` calls `transcribe.remote` once per show:

  | Show | pre-registered | **ACTUAL** | `TRANSCRIBE` |
  |---|---|---|---|
  | The Jacob Shapiro Podcast | 349 | **345** | 1 (ep 354) |
  | Geopolitical Cousins | 76 | **76** *(derived)* | 1 (ep 77) |
  | The Observing Japan Podcast | 7 | **7** | 0 |
  | total | 432 | **428** | 2 |

  **The pre-registration was wrong by 4, for a nameable reason: it subtracted
  the feed roll-off but NOT `corpus/exclusions.py`.** The JSP block prints all
  three classes and they sum to the feed exactly — `4 EXCLUDE + 345
  IDENTIFY_ONLY + 1 TRANSCRIBE = 350` — giving `350 archive − 1 (ep 7 rolled
  off) − 4 (cross-posts 271, 273, and the two unnumbered 2026-07-29 /
  2026-07-31) = 345`. Both original corrections DID hold: ep 7 is unreachable
  (feed's oldest is ep 8, 2022-06-03), and the two 2026-09-07 episodes were
  stamped at ingest and never entered the backlog. Re-derived independently
  against all three live feeds and it reproduces 345 and 7 exactly.

  **`14/46 chunks stamped` on JSP 354 is NOT a bug.** `stamp_chunks` returns
  chunks that got a NAME; `identity_version` goes on all 46 regardless. One
  enrolled speaker, one unenrolled guest. The bad state is `0/n` WITH clusters
  named — the `STAMP JOIN BROKEN` predicate, which did not fire.

  **Cousins' 76 is DERIVED, not observed.** `modal app logs` returns the same
  ~3283 entries regardless of `--since` / `--until`, and the Cousins block had
  scrolled off the front — UNKNOWN, not absent. Re-derive with the committed
  probe: `PYTHONPATH=G:/audio-transcription/via-modal py
  probe_expected_backlog.py` (system `py`, because the `uv` env has no
  `feedparser`). The control that catches a roll-off: feed-episodes-
  at-or-before-boundary EQUALS the archive count per show — passes for Cousins
  (76) and Observing Japan (7), fails for JSP by exactly ep 7.

  **`probe_expected_backlog.py` IS WHAT PRODUCED THE WRONG 432, and it is now
  FIXED (2026-09-08).** It subtracted the feed roll-off but never imported
  `corpus.exclusions` at all, so `reachable` counted four episodes the planner
  checks `EXCLUDE` on FIRST. An excluded episode is emitted by the feed AND
  present in `assignments.json`, so it survives the reachability filter and
  the archive count both — invisible to every check that does not name it.
  The fix mirrors `decide_action`'s arm via `is_excluded`, passing the guid
  `parse_all_episodes` already carries, and the repaired probe reproduces the
  planner's 345 / 76 / 7 = 428 exactly. It also now prints why 433 (stamp set,
  store-based) and 428 (backlog, feed-based) are different numbers.

  **STEP 2 DONE 2026-09-08 — `wrote 28644; read back 28644/28644 records
  carrying the version`, terminal marker `STAMP COMPLETE`.** `written`
  equalled `planned`, so the gate passed with no drift between dry run and
  apply, and `total` equalled `planned`, so no non-episode rows sat inside
  the stamp set. `verify_stamped` re-reads from the store rather than
  trusting `apply_updates`' return value — which is the only way to catch a
  Chroma metadata-only `update()` reporting success and writing nothing.

  The dry run that produced that number: 29321
  records held, 433 episodes in the stamp set (350/76/7 — selection is by
  triple against the STORE, not the feed, so the roll-off does not touch it
  and ep 7 IS still stamped), **`planned 28644` chunk updates**, terminal
  marker `STAMP PLAN COMPLETE`, exit 0. The apply is
  **`--apply --expect 28644`** — NOT 433. **`--expect` takes the CHUNK-update
  count, not the episode count**; the two sit four lines apart in the same
  output and the earlier note here said 433, which is the wrong one.
  The 4 excluded cross-posts stay in the stamp set (built from
  `assignments.json`, which is not exclusion-filtered) and that is harmless:
  271 and 273 have no records since the 2026-09-01 hand deletion, so they
  plan zero chunks and `verify_stamped` only counts records that exist.
  Residual 29321 − 28644 = 677: 161 is the two 2026-09-07 episodes (already
  stamped at ingest, absent from a 2026-09-04 assignments file — the same 161
  by which `records=29160` in the runbook has grown); the other 516 the
  planner never reaches, since it enumerates from the FEED and every feed
  episode already has a classification line. BOUNDED, not decomposed — it
  cannot enter the stamp and cannot enter the backlog, and step 3 is the
  independent read on that.
  Then → **the night after the stamp should be `0` on all three shows**, not
  "the post-boundary episodes" as the runbook said: that wording predated the
  deploy, and because v7 went live BEFORE those two episodes were ingested
  they get stamped at ingest, while nothing was published between the archive
  boundaries and 2026-09-07 (verified against all three live feeds
  2026-09-08). **`428` there means the stamp did not land** — not 433, which
  is a number the planner cannot print.
  → only then raise the cap to 25. Raising the cap before the stamp queues
  all 428 episodes on night one.

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
