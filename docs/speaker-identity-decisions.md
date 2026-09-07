# Speaker identity: decisions, and the ones that were wrong

**Companion to [`superpowers/specs/2026-09-02-speaker-grouping-design.md`](superpowers/specs/2026-09-02-speaker-grouping-design.md).**
The spec says what was designed. This says what was *decided during the
build*, what those decisions cost, and which of them turned out to be wrong.

**Why this file exists.** The build ran under `subagent-driven-development`,
which keeps a progress ledger at `.superpowers/sdd/<plan>/progress.md`. That
directory is deliberately self-ignoring — the skill's own script writes a
`.gitignore` into it and calls the contents "one plan's short-lived
artifacts", scoped per plan so a stale ledger can never be misread as current
progress. It is a recovery map for work in flight, and it is **designed to be
deleted**. The skill says nothing about preserving what was decided inside
it. So the rulings that outlive the plan were moved here, deliberately,
before the workspace went. The skill's own phrasing for the failure this
avoids: *a roll-up nobody reads is a silent discard.*

---

## 1. Design rulings that shaped the code

**An unidentified chunk renders `unidentified voice (SPEAKER_01)`, not a bare
`unidentified`.** The spec's four-state table collapsed "declined to name" and
"never labelled" on the grounds that the difference is not the reader's
business. That reasoning is right and it missed something: `speaker` is the
only *within-episode* discriminator once identity has not resolved, so a bare
`unidentified` makes two different voices in one episode indistinguishable.
The amendment honours the spec's stated reason while keeping information the
table would have silently discarded. Three pre-existing tests in
`tests/test_rendering.py` pinned the old behaviour and were updated: the spec
is the binding authority and it supersedes them.

**`--limit` must not bypass the pre-registered corpus guard.** The guard
refuses the archive pass if the world has moved (433 episodes, 5,424 crops).
A `--limit` smoke test originally skipped it. The authorised run passed no
`--limit` and was therefore already protected — but the guard exists to
prevent an unknown-size operation against someone else's bandwidth, and a
live bypass in the one script that downloads 56 GB defeats it. Fixed rather
than deferred. Cost: `--limit` smoke tests now also require the full plan to
be intact.

**`--limit` truncates BEFORE the `--resume` filter, deliberately.** So
`--limit N` names the same N episodes of the full plan every run, which is
what makes a smoke test reproducible — and it means `--limit N --resume` does
**not** mean "do N more". Documented at the call site in `speaker_embed.py`.

**The `speaker-vectors` Modal volume was dropped.** Neither function wrote to
it; both return their records over the wire and the entrypoint writes locally.
A mounted-but-unused volume is dead configuration. Reinstate it only if
aggregate return size becomes a problem (~28 MB total today).

**Centroids are derived, never stored** — `corpus/enrolment.py` states the
rule and the ingest path obeys it. A draft of the ingest wiring persisted the
five enrolment centroids to a new artifact; that was reverted before landing.
Its justification was also factually wrong (it claimed deriving at run time
needed the 49 MB of `clips/` audio, when the clip *vectors* are already
computed and committed at 2.1 MB). Two hand-maintained copies drift
asymmetrically, and the stale one works right up until it matters.

---

## 2. The stop condition that fired, and what it took to clear

The plan pre-registered: if co-membership conflicts exceed 5% of matched
clusters, stop rather than backfill. **They came in at 30.8%.**

The plan's own remedy — raise the threshold — was measurably inapplicable. In
`The Jacob Shapiro Podcast - Episode 100` the transcript text shows SPEAKER_01
is Marko Papic, yet that cluster scored **0.879 against Jacob's centroid and
0.276 against his own**, and Jacob's centroid also beat Jacob's real cluster
(0.728). No threshold sits between 0.879 and 0.728.

The cause was not the threshold, the centroids, or the clustering. It was that
**the entire archive had been embedded from the wrong audio**: `torchaudio.load`
counts `frame_offset` in the file's native rate, and the code computed it
against a hardcoded 16000, so every crop landed at a third of its intended
timestamp. Found by treating the timestamp scale as a free parameter and
scanning it — agreement peaked at **0.916 at r = 1/3** against **0.492 at
r = 1**, and split by measured file rate the 48 kHz and 44.1 kHz episodes
peaked in *different* places, each losing 15–20 points at the other's ratio.

**Everything computed from that pass was void in both directions.** The
30.8% was void; so was the Ship 2 checkpoint's "1 qualifying group", which
pointed the same way as the real answer and was retracted anyway. *A negative
computed from corrupt input is not a negative.*

After the re-run: conflicts 45 of 704 (**6.4%**), still above the 5% bar. The
bar was not talked past — the harm it exists to prevent was measured
directly. Conflicting cluster pairs scored median **0.7345** against a
cross-person control of **0.1579**, and the weakest conflicting pair
outranked the control's 95th percentile. **45 of 45 were one voice split in
two by the diariser; none was two people merged.** Benign. The backfill
proceeded on that evidence, not on the rate.

---

## 3. Claims that were wrong

**"The enrolment clips never reach this arithmetic."** Stated in commit
`68231ba` *as a code-path verification*. They do — `embed_every_clip` calls
`Embedder.embed(path, 0.0, 3600.0)` directly. The conclusion survived for a
different reason: the offset is `0.0 * native == 0` at any rate, and 3600 s
exceeds every clip so the short-crop guard is suppressed by its own EOF
exemption. This is the one claim that licensed **not** re-running the
bake-off. A right answer reached for a wrong reason is an accident, and the
wrong reason would have been reused. Corrected in `corpus/voice.py`, where a
reader meets it, because a commit message cannot be edited.

**"So no future backfill is needed."** Overstated. Routine ingest needs none.
Two paths still produce records with no `speaker_id` and neither self-heals:
the EMBED_ONLY re-embed has no audio and cannot identify, and an enrolment
failure stores a batch unstamped. Planning decides SKIP/EMBED_ONLY on record
counts, never on `speaker_id` presence.

> **RESOLVED** by phase 1 of
> [the identity lifecycle design](superpowers/specs/2026-09-05-identity-lifecycle-design.md).
> `identity_version` records that identification RAN, `IDENTIFY_ONLY`
> queues the episodes it did not, and `repair_identity` re-earns them from
> audio with a metadata-only write. There was a third producer the two
> above missed: `identify_clusters` raising on one episode does the same
> for that episode. All three self-heal for the same reason -- they skip
> the call that writes the version.

---

## 4. Known exposure, not fixed

**A stale `speaker_id` survives a re-transcribe.** Confirmed by
`tests/test_write_path.py::test_a_stale_speaker_id_survives_a_re_transcribe`.
`corpus/writing.py` writes positional chunk ids and Chroma merges metadata,
so re-transcribing an already-identified episode renumbers the diarisation
clusters while chunk *i* keeps the previous run's value — a chunk labelled
`SPEAKER_02` carrying a name assigned when cluster 2 was someone else, and
`display_speaker` prefers `speaker_id`. The only cure is deleting the
episode's records before writing, and `upsert_then_prune` upserts first *on
purpose* so a crash mid-write leaves more data rather than less. Reversing
that is a write-path decision with its own failure modes.

> **STILL OPEN, and now designed.** Phase 2 of
> [the identity lifecycle design](superpowers/specs/2026-09-05-identity-lifecycle-design.md)
> fixes it with `replace_identity` -- a delete before the upsert on the two
> paths that move chunk boundaries. Two things this entry got wrong, both
> found by review: the exposure is **not confined to re-transcription** --
> the mechanism is positional ids plus metadata merge, so it fires on a
> `RULES_VERSION` bump and on the repair of a torn episode too -- and the
> delete is only affordable **because** phase 1's repair path now exists to
> re-earn a cleared name.
>
> Until phase 2 lands, `transcribe.py` withholds `identity_version` from
> any episode whose stored records already carry a name, so a stale stamp
> can be wrong but can never be **certified**. See `docs/operations.md`.

---

## 5. Process lessons, earned the hard way

**A check that cannot fail is worse than no check.** Three instances in one
build: a normalisation test whose fixtures were all unit-length; a
"documents are not rewritten" test against a fake that preserved documents by
construction; and a count guard whose only caller passed
`expected=len(updates)` from the same call that produced them. The last was
ruled up from Minor to Important and replaced with an `--expect` count the
operator carries from a dry run, so a collection that moved between the two
refuses.

**An idle notification is not proof an agent stopped working, and committed
state is not proof of working-tree state.** Two implementers were dispatched
onto one file concurrently because `git log` and a grep both showed a fix
absent — while it sat applied and uncommitted.

**"The job finished" from an agent means it stopped waiting, not that the
process exited.** Three `--limit 3` Modal runs were live at once, the third
started *after* its author reported cleanup. Detected only because one file's
size disagreed three ways within minutes. Confirm via the scheduler's state,
never an agent's report or a single file listing.

**A deferred minor is filed against the system as it was.** `embed_clips`
raising only on total failure was fairly minor while it fed a one-off
bake-off. It became serious the day `clip_vectors_*.json` became a mounted
input to the nightly ingest — the code never changed, its blast radius did.
Re-read deferred items against the current system.

**Pre-register the numbers.** Every production defect in this build was found
by writing the expected counts down before running and comparing, never by
re-reading code that had already been reviewed repeatedly.
