# Design: speaker identification, measured before it ships

**Status: approved in brainstorming after adversarial review, not yet
implemented. Written 2026-09-01.**

Supersedes the scoping half of
[`2026-08-31-speaker-identification-design.md`](2026-08-31-speaker-identification-design.md),
which remains the statement of *why* this is worth doing. That note argued its
own conclusion: "the eval set and the measured precision, written down —
without that number this is a demo." This spec builds the measurement, and
stops well before the production write path.

Depends on [`2026-08-31-corpus-integrity-design.md`](2026-08-31-corpus-integrity-design.md),
which shipped 2026-09-01. `RULES_VERSION` and `episode_id_prefix` are both
consumed here.

## Part 0 — Why this is two phases

An earlier draft of this spec pre-registered a gate — zero misattributions,
coverage >= 70% of speech — and an eval set sized to support a 98% precision
claim. Adversarial review killed both numbers, and the way they died is the
reason for the structure below.

**The coverage gate was unpassable by a perfect matcher.** Coverage was defined
as named speech seconds over *total* speech seconds. The denominator includes
guest speech, which the matcher is designed never to name, so the metric's
ceiling was set by each show's guest ratio rather than by matcher quality. On
The Jacob Shapiro Podcast — 341 of the 400 measurable transcripts — that
ceiling is 66.7%. The gate failed before any matcher existed.

**The sample size rested on a number nobody measured.** It assumed ~1.5 named
assignments per episode. Part 2 measured *clusters* per episode (2.39); it never
measured *enrolled hosts* per episode, and 1.5 is not derivable from it. The
plausible range runs from 1.15 to 1.96, which moves the achievable precision
bound from 97.4% (fails) to 98.5% (passes). Both gate numbers swing from
unpassable to comfortable on that one quantity.

**And it cannot be measured from the clustering alone.** Jacob Shapiro's
top-1/top-2 speech split is 66.7%/99.1%; Geopolitical Cousins' is 62.0%/96.7%.
The same shape. "One host plus a guest" and "two hosts" are indistinguishable
without labels — telling them apart *is* the identification problem. You need
labels to size the labelling.

The lesson generalises past these two numbers. The Wilson bound, the sampling
rule, the leakage guard — every part that looked like rigour was correct. What
broke were two unremarkable figures carried in from intuition and then allowed
to become load-bearing. **Pre-registration does not protect a number whose input
was guessed; it freezes the guess and makes it look decided.**

So: Phase 1 measures the quantities Phase 2's gate depends on, cheaply and
without a GPU. Phase 2's gate and sizing are **computed from Phase 1's outputs**
and are deliberately not fixed in this document.

## Part 1 — Scope

**Phase 1 (fully specified here).** A clip-cutting tool, a labelling CLI, a
~20-episode pilot label set, and four measurements: hosts per episode, host
speech share, cluster impurity, and the accuracy of a free baseline.

**Phase 2 (structure specified, numbers deferred).** The voice-embedding cache,
derived enrolment, the matcher, and the measured gate.

**Out of scope in both, deliberately.** Writing names into production chunks;
the archive re-embed; guest identification; auto-enrolment of recurring
unknowns.

The ordering is the point. A matcher wired into the corpus before it is
measured writes names that are expensive to retract, because the speaker label
is baked into the embedded text and a retraction is a re-embed of ~30k chunks.

## Part 2 — Measured facts

Everything here was probed. Where a figure is an estimate or an assumption, it
says so.

**Clusters per episode.** Measured 2026-09-01 across the 400 transcripts in
`downloaded/`:

| | median | mean | min | max |
|---|---|---|---|---|
| All | 2 | 2.39 | 1 | 7 |
| Geopolitical Cousins (n=59) | 3 | 3.19 | | |
| The Jacob Shapiro Podcast (n=341) | 2 | 2.25 | | |

Distribution: 1 cluster x17, 2 x262, 3 x89, 4 x21, 5 x5, 6 x3, 7 x3.

**Speech concentration.** Measured the same day, same parse as
`corpus/chunking.py`, turns under 1.5s dropped and single turns capped at 30s
per §4.2:

| | pooled top-1 share | pooled top-2 share | median top-1 | episodes with top-1 < 70% |
|---|---|---|---|---|
| The Jacob Shapiro Podcast | 66.7% | 99.1% | 68.2% | 193 / 341 |
| Geopolitical Cousins | 62.0% | 96.7% | 62.8% | 51 / 59 |
| All 400 | 65.7% | 98.6% | | |

389 of 400 episodes have a dominant cluster holding over 50% of speech, median
share 66.2%.

**Top-k by duration is an upper bound on coverage.** For any assumed number of
enrolled voices k, the top-k clusters by speech time bound what any correct
assignment of k people could cover, because top-k-by-duration is by
construction the maximum-duration k-subset. This makes the ceiling figures above
robust to not knowing which cluster is whom — but sensitive to k, which is
exactly what Phase 1 measures.

**Hosts per episode is UNKNOWN.** The claim "one host for The Jacob Shapiro
Podcast, two for Geopolitical Cousins" appears in the 2026-08-31 note and in
this spec's earlier draft. It was never measured. Marco Papic co-hosts
Geopolitical Cousins and also appears on the Jacob Shapiro feed — a
first-40-segment grep for `\bmarco\b|\bpapic\b` hits 37 of 341 Jacob Shapiro
episodes (10.9%) — and under §4.1's person-keyed enrolment he would be named on
both. Counting him raises the pooled coverage ceiling from 72.8% to 75.5%.

**"The dominant cluster is the host" is a HYPOTHESIS.** What is measured is a
property of the clustering (389/400 have a dominant cluster). That the dominant
cluster is a host is unverified, and Phase 1 tests it.

**`downloaded/` is a partial copy.** 400 of 438, containing **no Observing Japan
episodes**. Every figure above therefore describes two of three shows.

**Audio is not retained.** `transcribe.py:378` writes `/tmp/episode_N.mp3`;
`transcribe.py:455` deletes it. Historical work re-downloads from the feed
enclosure.

**Six episodes have no obtainable audio.** The `FEED_UNREACHABLE` set has aged
off its feeds and can never be identified. Permanent.

**The recorded transcript carries the clustering.** Every line is
`[SPEAKER_00] 12.3s - text` — label, start time, text. No end time.

## Part 3 — Phase 1: the pilot

No GPU, no embedding model, no Hugging Face gating. Audio is downloaded, clipped
and discarded.

### 3.1 Sample

**20 episodes**, chosen deterministically — every k-th by date within each show
— stratified across all three shows. Not picked. Picking inflates every figure
here, because episodes that look easy to label are the ones with clean audio and
no crosstalk.

Observing Japan must be sampled from the Modal volume, since it is absent from
`downloaded/`. At a mean of 2.39 clusters this is roughly **48 clusters**, about
twenty minutes of listening.

### 3.2 Clips are spread across the episode, not taken from one place

For each cluster, take three clips of ~10s: from its **earliest**, **middle**
and **latest** long turn.

This is the one change that attacks the deepest flaw in the earlier draft.
Ground truth is per-cluster, but the harm is per-turn: a cluster that is 80%
host and 20% guest, sampled three times from the middle, gets labelled "Jacob
Shapiro", matched "Jacob Shapiro", and scored **correct** — while a fifth of its
text ships under a real person's name. Spreading the clips across the episode
gives the labeller a chance to hear that the cluster holds two people.

The labeller may mark a cluster **impure**. That is a first-class outcome, not a
failed label.

### 3.3 What Phase 1 measures

1. **Hosts per episode (k)**, per show. Feeds Phase 2's sizing.
2. **Host speech share** — the true coverage ceiling, per show. Feeds Phase 2's
   coverage denominator and bar.
3. **Cluster impurity rate.** Feeds the validity of per-cluster ground truth
   itself.
4. **Baseline accuracy.** Does the top-1 cluster hold a host? For a show with
   two enrolled voices, do the top-2?

### 3.4 Decision rules, pre-registered now

These are rules about what to *do*, not predictions of unmeasured quantities, so
fixing them here is legitimate.

- **Impurity above 5% invalidates per-cluster ground truth.** At that rate
  diarisation error alone injects enough misattribution to make a 98% precision
  claim unsupportable no matter how good the matcher is. Phase 2's measurement
  design must change before it is built, not after it fails.
- **Size Phase 2's eval split from measured k** so that expected named
  assignments are **>= 150**, not the bare 133 the bound requires. n=133 yields
  exactly 0.98006 and n=132 yields 0.97991; the earlier draft sat one episode
  from failing on arithmetic alone.
- **If the baseline names hosts with zero errors and covers most host speech**,
  re-scope before building the voice pipeline. It costs no GPU, no audio
  embedding and no model gating, and the expensive design has to beat it rather
  than merely be checked by it.
- **The baseline cannot resolve co-hosts or cross-show appearances.** It cannot
  separate the two Geopolitical Cousins hosts, and it cannot recognise a host
  appearing as a guest elsewhere — which is the cross-show query the feature
  exists for. So a good baseline result narrows Phase 2's scope; it does not
  delete it.

## Part 4 — Phase 2: structure

Numbers deferred to Phase 1. The design decisions below stand.

### 4.1 Enrolment is keyed by person, not by (show, person)

One centroid per human, so Marco Papic on the Jacob Shapiro feed is the same
identity as Marco Papic on Geopolitical Cousins. This is what makes "what has X
said across the archive" answerable at all. The cost is that a bad enrolment
misfires everywhere at once — a reason to measure enrolment quality, not to
fragment identity across feeds.

### 4.2 Embed the recorded turns; do not re-run diarisation

The clustering is already in the transcript files. The archive pass is: fetch
audio, crop each recorded turn, embed, aggregate per cluster, discard audio.
**The embedding model only** — no whisper, no alignment, no diarisation
pipeline.

This is not merely cheaper. Re-running diarisation produces a *fresh* clustering
whose `SPEAKER_XX` numbering has no relationship to the numbering baked into the
stored transcript, so a name learned against cluster 2 of the new run could not
be written back onto cluster 2 of the old text. Embedding the recorded turns
keeps labels aligned **by construction**.

The price: a turn's end must be derived as the next line's start, over-including
trailing silence and any gap. Mitigations — drop turns under **1.5s**, cap a
single crop at **30s**, aggregate across the whole cluster.

**The live path would differ.** Identification on a new episode runs inside the
transcriber where real end-times exist. That fidelity gap is acceptable because
derived ends are the *looser* input: a matcher tuned on them is tuned on dirtier
audio than production supplies. Revisit if a production path is specified.

### 4.3 Enrolment centroids are derived, never stored

`speakers/labels.json` holds provenance only — which cluster of which episode is
which person. Centroids are recomputed from cached vectors on every use. No
vector is ever checked in.

This follows `corpus/exclusions.py`, where the exclusion lists are derived for
the same reason: two hand-maintained lists drift **asymmetrically**, the stale
one working right up until the moment it matters. A checked-in centroid blob is
that bug with a binary payload, undiffable and unreviewable.

```json
{"version": 1, "clusters": [
  {"episode": "Geopolitical_Cousins-ep73-2026-07-29",
   "cluster": "SPEAKER_00", "person": "Jacob Shapiro",
   "split": "enrol", "impure": false, "labelled_on": "2026-09-01"},
  {"episode": "Geopolitical_Cousins-ep73-2026-07-29",
   "cluster": "SPEAKER_02", "person": null,
   "split": "enrol", "impure": false, "labelled_on": "2026-09-01"}
]}
```

`episode` is `corpus.identity.episode_id_prefix(show, episode_number, date)`. A
second identity scheme in a new subsystem is the bug this repo already paid for
once; invariant 1 exists because of it.

**`person: null` is ground truth, not missing data.** It means "not one of the
recurring voices". Naming such a cluster is a precision violation, and it is the
one that matters most: the realistic failure is not confusing two hosts, it is
confidently stamping a guest with a host's name.

### 4.4 Assignment takes a threshold **and** a margin

```
best >= threshold  AND  (best - second_best) >= margin  ->  name
otherwise                                               ->  label unchanged
```

The dangerous case is not a low score but **two high scores a hair apart**,
where a threshold picks the higher one and reports confidence. The margin makes
ambiguity fail closed. An unmatched cluster keeps its existing `SPEAKER_XX`
label; no `guest_1` vocabulary is introduced.

Centroids are duration-weighted — `L2_normalise(sum(d_i * v_i) / sum(d_i))` —
since a 40-second turn is both less noisy and more representative than a
2-second interjection.

### 4.5 Sampling, splitting and tuning

Deterministic every-k-th-by-date sampling, as in §3.1, for the same reason.

**Split assigned before labelling**, deterministically from the episode prefix.
Assigning it afterwards would let a disappointing result be re-split into a
better one. `enrol()` **raises** if any episode appears in both splits —
asserted in code, matching how the `corpus/` invariants are pinned.

Enrolment needs far fewer episodes than evaluation, since each host recurs
across nearly every episode of their show. The split is deliberately lopsided
toward measurement.

**Threshold and margin are tuned within the enrolment split only**, by holding
out turns from enrolment episodes. Tuning on same-episode turns is optimistic —
same recording conditions — which biases the threshold **too loose**. That bias
surfaces as errors on the eval split rather than as a hidden pass, so it fails
in the detectable direction.

### 4.6 The measurement

- **precision** = named-and-correct / **named**. A `null`-truth cluster given a
  name counts against it. A real person left unnamed does not — that is recall.
- **coverage** = named host seconds / **true host seconds**. Host-relative, not
  total-speech-relative. The earlier definition measured each show's guest ratio
  rather than the matcher, and made the gate unpassable.
- **sweep** over (threshold, margin), producing the table the operating point is
  chosen from.

**The denominator is assignments the matcher makes**, not clusters labelled —
roughly k per episode, which Phase 1 measures.

Zero errors on n named assignments gives a one-sided 95% lower bound of
`n / (n + z^2)` with `z = 1.645`; this closed form is the Wilson interval's
all-successes case and was verified against the full computation. A **one-sided**
bound is correct here: the claim of interest is "precision is at least X", and
there is no use for an upper bound.

**Per-turn evaluation is rejected for matcher error and required for
diarisation error.** Matcher errors are perfectly correlated within a cluster —
if a centroid is wrong, every turn it covers is wrong — so per-turn scoring
would inflate the denominator without new information. Diarisation errors are by
definition *not* constant within a cluster, which is exactly why a per-cluster
metric cannot see them. §3.2's spread clips and the impurity flag are how that
blindness is addressed; they are not optional polish.

### 4.7 The gate

**Deferred to Phase 1's outputs.** The shape is fixed: zero misattributions on
the eval split, plus a coverage floor expressed against **host** speech. The
numbers are computed once k and the host speech share are known.

**The eval split is measured once.** Re-tuning after a failure and re-measuring
on the same episodes is selection on the test set. If the gate fails, the burned
split becomes a dev split and a **fresh** eval split must be labelled before any
new claim is made.

Both numbers are asserted in the test suite, so a sweep that does not clear them
refuses rather than reports.

### 4.8 Negative controls that cost no labelling

Across the whole cached archive: a host should be named in nearly every episode
of their own show, and a two-host show should not yield five named people in one
episode. These catch gross failure for free and are the checks most likely to
fire if enrolment is subtly wrong.

Note these are *controls*, not the baseline. The baseline is §3.4's
dominant-cluster hypothesis, and Phase 2 must beat it.

## Part 5 — Architecture

Pure logic in `corpus/`, Modal at the top level, matching the split argued in
`corpus/showplan.py`'s module docstring.

| Unit | Phase | Purpose | Depends on |
|---|---|---|---|
| `speaker_id.py` | 1 | Modal app: fetch audio, cut spread clips, discard audio | modal |
| `label_clips.py` | 1 | Local CLI: prompt, append to `labels.json` | — |
| `corpus/speaker_stats.py` | 1 | k, host share, impurity, baseline accuracy. Pure. | — |
| `speakers/labels.json` | 1 | Ground truth. Hand-made, reviewable, in git. | — |
| `corpus/speakers.py` | 2 | Centroids, cosine match, threshold + margin. Pure. | numpy |
| `corpus/speaker_eval.py` | 2 | Precision, coverage, sweep, leakage guard. Pure. | numpy |
| `voice-cache` volume | 2 | `{episode_id_prefix}.npz` per episode | — |

Every `corpus/` module imports no Modal and no audio library, so all of the
statistics, the matcher and the measurement run in the test suite on CPU.

**Phase 1 workflow:**

```
uv run modal run speaker_id.py::cut_clips --episodes 20   # -> clips/ locally
uv run python label_clips.py                              # -> speakers/labels.json
uv run pytest tests/test_speaker_stats.py                 # the four measurements
```

`label_clips.py` walks unlabelled clusters, plays a clip, takes a name, blank
for unknown, or a key for impure, and **appends after every answer** — so an
interrupted session resumes rather than restarts.

## Part 6 — UNKNOWNs

None of these may be treated as settled by assumption.

1. **Hosts per episode.** The quantity both Phase 2 gate numbers hinge on.
   Measured by Phase 1; unmeasurable from clustering alone.
2. **Observing Japan's cluster count and speech concentration.** Absent from
   `downloaded/`; every Part 2 figure covers two of three shows. Probe against
   the Modal volume when sampling.
3. **Cluster purity.** If diarisation clusters routinely mix speakers, the
   per-cluster ground-truth design is invalid regardless of matcher quality.
4. **Hugging Face gating on the embedding model** (Phase 2 only). `README.md`
   records accepting terms for `pyannote/speaker-diarization` and
   `pyannote/segmentation`. Whether that covers
   `pyannote/wespeaker-voxceleb-resnet34-LM` is unknown; a gated model fails at
   runtime with an authorisation error, not at setup.
5. **What the deployed image exposes** (Phase 2 only). pyannote is not pinned in
   `transcribe.py`; it arrives transitively via `whisperx==3.8.5` and was frozen
   at build time. Per `docs/operations.md`, replicate the image spec **verbatim**
   so content-addressing reuses the cache; build lines in the log mean the spec
   drifted and any versions read are not the deployed ones.
   `.add_local_python_source("corpus")` must stay **last** in the chain.
6. **Audio URL resolution.** Enclosure URLs can rot independently of the feed
   entry. A sampled episode whose audio 404s is replaced by the next episode in
   deterministic order, and **the substitution is recorded** — not silently
   skipped, which would reintroduce selection.

## Self-Review

**Coverage of the prior design note.** Its enrolment/matching/threshold/unknown
proposal → §4.1, §4.3, §4.4. Its "fail closed, and mean it" → §4.4 and the
`null` semantics in §4.3. Its "the part worth more than the feature" → Part 3
and §4.6–4.7. Its out-of-scope list is preserved and extended with the
production write path, which it left ambiguous.

**Three numbers from the earlier draft were wrong, and all three failed the same
way.** The coverage denominator, the hosts-per-episode figure, and the eval
sizing derived from it were estimates that became pre-registered gates. The
statistics around them — the Wilson closed form, verified against the full
computation; the sampling rule; the leakage guard — were correct throughout.
Part 0 records this because the failure mode is more reusable than the fix.

**What is deliberately still weak.** The live-path fidelity gap in §4.2 is real
and argued as failing safe rather than eliminated. Part 2's figures describe two
of three shows. Phase 2's gate is genuinely undetermined, and this document
should not be read as though it were.

**One scope boundary worth defending.** The archive backfill is out even though
`RULES_VERSION` makes it schedulable, because a bump alone cannot rename anyone:
`EMBED_ONLY` re-chunks from the **stored transcript** (`transcribe.py:313-325`,
verified), whose text still says `SPEAKER_00`. A backfill requires rewriting the
transcript files first — a separate, costed decision that belongs in its own
spec.
