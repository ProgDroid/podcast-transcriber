# Design: speaker identification — a text baseline, and a voice model behind a gate

**Status: approved after two rounds of adversarial review, not yet implemented.
Written 2026-08-31.**

Implements [`docs/speaker-identification.md`](../../speaker-identification.md).
Ships after [corpus integrity](2026-08-31-corpus-integrity-design.md), a hard
dependency — §0.

The original design was a voice model with a deferred backfill. Two review
rounds measured the cheap alternative and inverted it: **phase B1 (text) ships;
phase B2 (voice) is specified but gated behind a measured trigger** it does not
currently clear.

## 0. Why corpus integrity is a hard dependency

`build_chunks` cuts a chunk boundary at every speaker change. When `SPEAKER_00`
and `SPEAKER_01` both resolve to `"Jacob Shapiro"`, the boundary disappears,
adjacent chunks merge, and the episode's chunk count drops. `upsert` overwrites
only `0..n-1`, stranding every higher index with `[SPEAKER_01]` still in its
document text, duplicating passages that now live in the merged chunk. Spec A's
upsert-then-prune writes make this safe by construction, and its full
replacement makes it safe for this spec to add a metadata key.

## 1. Measured evidence

From the 438-file volume unless noted. 42 clusters were audited by hand, each
judged independently of the rule that named it, unresolvable cases counted as
errors.

**Precision** (audited on the 400-file mirror):

| Rule | Audited | Correct | Wilson 95% |
|---|---|---|---|
| R1 first-person self-ID | 12 | 12 | [0.76, 1.00] |
| R2 vocative (GC only) | 15 | 15 | [0.80, 1.00] |
| Combined | 42 | 41 | [0.877, 0.996] |

**Coverage of name-agnostic first-person self-ID, per show** (whole volume,
first 600s of each episode):

| Show | Episodes with a hit | Top capture |
|---|---|---|
| The Jacob Shapiro Podcast | 295/355 (83%) | `Jacob Shapiro` ×231 |
| The Observing Japan Podcast | 6/7 (86%) | `Tobias Harris` ×5 |
| Geopolitical Cousins | 21/76 (28%) | noisy — see below |

Four findings shape the design:

**The dominant cluster is not the host.** The cluster that self-identifies as
Jacob is the *second*-largest in 191 of 283 Jacob Shapiro episodes and largest
in only 86 — guests out-talk hosts on an interview show. Naming by dominance
would misattribute the host across two-thirds of that show.

**Reading out an address is not claiming to be someone.** Geopolitical Cousins
episode 6, `SPEAKER_01`: *"reach out to us at jacob at jacobshapiro.com"* and
also *"**Jacob and Marco** had a fantastic conversation today"* — third person
about both hosts. It is the produced intro voice. R1 requires a **first-person
frame**, which eliminates the class by construction and *raises* coverage, since
self-introduction is far more common than an email read-out.

**The name must be a validated capture, not a literal.** Geopolitical Cousins'
28% hit rate is noisy: captures include `Rusillo`, `Orthodox`, `Los Angeles`,
`Slavic Saudi Arabia`, `Shaggy Marco`. "I'm Orthodox" is first-person and not a
self-identification. Validating the capture against a per-show roster rejects
all of those; fuzzy matching also rescues `Jacob Shapira`, a transcription error
rather than a different person.

**The rules are complementary by show, not redundant.** R1 carries Jacob Shapiro
(83%) and Observing Japan (5/7 name Tobias Harris outright — a single-host show,
close to ideal for it). R2 carries Geopolitical Cousins (58/59). Between them
all three shows are covered with no voice model.

## 2. Phase B1 — the text baseline

### R1 — roster-validated first-person self-identification

A name-agnostic capture inside a first-person frame (`I'm <name>`,
`I'm your host, <name>`, `my name is <name>`, `email/write/reach **me** at …`),
with the capture validated against a per-show roster by fuzzy match. Extending
to a new feed is one roster line.

### R2 — vocative inference

For a show with a known roster: the cluster that addresses another roster member
by name, and is not itself named, is the remaining member. Extending R2 to the
Jacob Shapiro show reclaims 27 of its 86 residual episodes (8h, 1.9% of corpus
speech) for zero GPU.

**The guard is a calibrated parameter, not a hand-wave.** R2 applies only when
exactly two clusters exceed a speech-time floor — and that floor decides its own
coverage: 30/59 Geopolitical Cousins episodes qualify at 30s, 42 at 60s, 45 at
100s, 46 at 150s, 51 at 300s. The floor swings applicability across 36% of the
show, so it is calibrated against **the 13 Geopolitical Cousins episodes that
have ≥3 clusters over 150s** — e.g. Ep 4 `[2145, 1746, 1519, 795]s`, Ep 31
`[2192, 1889, 1232]`, Ep 49 `[1799, 1469, 1051]` — labelled with one question:
*is the third cluster a real third person or an over-split host?* Mostly
over-splits means the guard discards good episodes and should become "two
clusters cover ≥X% of speech"; mostly real guests means R2's audited precision
was measured on the easy subset and its bound does not transfer.

### Conflict, fallback, provenance

R1 and R2 disagreeing on a cluster yields `unknown` — never a tiebreak. Two
clusters in one episode resolving to the same person both fall to `unknown`:
that means diarisation over-split, and naming both would launder a diarisation
error into a false attribution.

Unnamed clusters become `unknown_1`, `unknown_2`, by descending speech time so
re-runs are deterministic. This deviates from the design note's `guest_1`:
"guest" asserts something not established, since a host who failed a rule is not
a guest.

Every chunk gains `speaker_source` ∈ `{diarisation, text_r1, text_r2,
voiceprint}`, so consumers can filter by provenance and the eval can be
recomputed from the corpus. Safe to add only because Spec A made writes full
replacement.

### Applying it to the archive

Names change the embedded document text, so this is a full re-embed of 438
episodes — ~29,894 BGE encodes through Spec A's `EMBED_ONLY` path, with no
Whisper, no alignment, no diarisation, **and no audio download**. The
transcripts supply labels and timestamps; only the names change. This is the
backfill the original design deferred, and it is affordable precisely because
text rules need no audio.

**It must be volume-driven, not feed-driven.** Six early Jacob Shapiro episodes
have aged off the front of their feed and are invisible to the feed-iterating
planner (Spec A §4.6). A feed-driven re-embed would silently leave those six
carrying `SPEAKER_XX` while every other episode gained names — and the
reconciliation would report the corpus complete, because they are complete,
just not renamed.

## 3. Phase B2 — the voice model, gated

### Why it is gated rather than built

Auto-enrolment from R1's output would produce a store containing **one person**.
Reimplemented over 400 transcripts, R1 names 317 clusters: **316 Jacob, 1
Marco** — only one host habitually self-identifies. With a single centroid the
`margin` gate is inert (there is no second-best), per-episode contention is
meaningless, and B2 degrades to single-speaker open-set verification on a bare
cosine threshold, the most channel-sensitive calibration case, on the eval set
with the fewest labels.

Its honest ceiling is small. Residual after first-person rules is 86 episodes,
94h, 23.6% of speech; if a voice model named the host cluster *perfectly in
every one*, that is 32h — **8.0% of corpus speech**. And the cost the original
spec never stated: `transcribe.py:354` deletes the audio, so enrolment needs
~265 episodes re-downloaded and a backfill needs all 438 (~400h of mp3) plus GPU
ECAPA over every cluster.

Its stated justification — reaching Observing Japan — does not survive
measurement. Roster-validated R1 names Tobias Harris in 5 of 7 episodes
directly, at zero GPU cost.

### The trigger

Build B2 only when **both** hold:

1. Residual unnamed speech exceeds **10%** of corpus speech after
   roster-generalised R1 and R2-extended-to-all-shows.
2. A hand-labelled sample of the residual demonstrates that voice can recover
   it — i.e. the residual is unnamed because nobody self-identifies, not because
   diarisation is poor there.

Condition 2 matters because if the residual is dominated by bad diarisation, a
voice model inherits the same broken clusters and recovers nothing.

### Design, if the trigger fires

Modules stay pure of torch so the decision logic runs on a laptop in a second:
`spans.py` (both parsers emit the same `SpeechSpan` list, so `embed_spans` serves
live and backfill unchanged), `embedding.py` (the only torch module),
`voiceprints.py`, `policy.py`, `matching.py`, `eval.py`.

Model: `speechbrain/spkrec-ecapa-voxceleb` — 192-d, ~20 MB, ungated, unlike
`pyannote/embedding` whose terms-acceptance the README already documents as a
runtime-failure footgun. Verify the loading API against Context7 before coding.

Enrolment draws from **R1 ∪ R2**, not R1 alone — R2 is what names Marco, and
R1-only is what produces the one-person store. The store is a Modal volume
artefact, **never committed**: an ECAPA centroid is a biometric template, and
stripping the name yields pseudonymisation, not anonymisation, since anyone with
a few seconds of public audio can match it back. The repository ships the recipe,
not the biometrics, and reproducibility survives because the podcasts are public
RSS.

`model` and `dim` are stored and checked on load with a raise on mismatch.
**Contamination is reported per-sample, not as a mean.** 28 of 317 R1-named
clusters exceed 70% of episode speech; spot checks found genuine solo episodes,
so contamination is **unknown, not absent** — and `intra_cohesion` as a mean over
265 samples cannot detect a 5–9% contaminated minority. Report the per-sample
distance distribution and flag outliers individually.

Matching: three gates (`min_speech_seconds`, `threshold`, `margin`) then greedy
per-episode assignment, with the same one-name-per-episode constraint as B1.
Missing store, failed model/dim check, or ECAPA failing to load: log and fall
back to B1's names, then to diarisation labels. Identification is additive and
must never break transcription.

## 4. Evaluation

**Gold set: 15 episodes, 5 per show.** Every diarised cluster labelled with a
name or "not a known host". `eval/labels.json` holds names against `(episode,
cluster)` pairs — no vectors, committable, since who hosts a public podcast is
public. **Plus the 13 Geopolitical Cousins multi-cluster episodes**, labelled
only for the R2 guard question above.

**Split:** tune and holdout, stratified by show.

**Definitions.** An *assignment* is an `(episode, cluster)` pair that received a
name. *Precision* is correct ÷ assignments — the headline, its complement the
false-attribution rate. *Coverage* is speech-seconds in named clusters ÷ total.
Precision always carries a **Wilson 95% lower bound**; the review's own 42-cluster
audit returned 41 correct with a bound of 0.877, a reminder that the point
estimate is the least interesting number in the table.

**Observing Japan is reported as a count, not a rate.** The show has 7 episodes
total; 5 in the gold set is 71% of it, and a stratified holdout is ~2 episodes,
~4 clusters. At 4/4 correct the Wilson lower bound is 0.510, at 5/5 it is
0.566 — no bar worth setting is cleared. Report leave-one-out over all 7 as
"n of m clusters correct", and say why a rate is not quoted.

R1 and R2 are reported **separately**. Their coverage is disjoint and their
failure modes unrelated; one blended number would hide that R2 contributes
nothing outside one show.

**Precision is the objective, not accuracy.** Misattributing a claim to a named
real person is worse than declining to attribute it, because this corpus feeds a
briefing that produces position signals.

## 5. Pipeline integration

In `transcribe()`, between diarisation and writing the transcript: resolve names
and rewrite `segment["speaker"]` in place, so the saved transcript and the Chroma
chunks stay consistent. Order: R1, then R2 under its guard, then (if it ever
exists) voiceprint, then `unknown_N`, with `speaker_source` recording which
answered.

## 6. Testing

Unit, CPU-only, no network, no GPU:

- **R1** — requires a first-person frame; the GC episode 6 produced-intro-voice
  case is a regression test that must return `unknown`; a capture failing roster
  validation (`Orthodox`, `Los Angeles`) is rejected; `Jacob Shapira` fuzzy-matches
  to the roster; an unknown name is not invented.
- **R2** — the guard refuses an episode with three clusters over the floor;
  disagreement with R1 yields `unknown`; two clusters resolving to one person
  both fall to `unknown`.
- **fallback** — `unknown_N` follows descending speech time and is deterministic
  across runs.
- **eval** — precision, coverage and Wilson bound against a hand-computed
  confusion matrix; the leave-one-out count for a 7-episode show.
- **B2, if built** — each gate rejects independently; empty store yields all
  unknown; store round-trip; model and dim mismatch raise; per-sample
  contamination flags a planted outlier.

Fixtures are seeded and labelled synthetic in code and in results: they validate
the harness, explicitly not the accuracy claim.

CI: GitHub Actions, ubuntu, Python 3.12, ruff and pytest. No GPU, no secrets, no
Modal.

## Acceptance

**B1:** `eval/results.md` reports R1 and R2 precision and coverage separately,
each with a Wilson lower bound, on the holdout; Observing Japan as a
leave-one-out count. Names and `speaker_source` present across the archive. The
GC episode 6 intro-voice case returns `unknown`. The R2 floor is a number chosen
from the 13-episode calibration, with the calibration recorded.

**B2 gate:** `results.md` states the residual unnamed speech percentage and
whether the trigger fired. If it did not, that is a successful outcome and the
phase stays unbuilt.
