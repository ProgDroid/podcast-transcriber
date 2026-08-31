# Design: speaker identification — a text baseline, then a voice model measured against it

**Status: approved after adversarial review, not yet implemented. Written 2026-08-31.**

Implements [`docs/speaker-identification.md`](../../speaker-identification.md).
Ships **after** [corpus integrity](2026-08-31-corpus-integrity-design.md), which
is a hard dependency and not merely an ordering preference — see §0.

Restructured from the original single-phase voice-model design after a review
measured a text-only baseline at 43% of speech time and ~0.98 precision, for
zero GPU cost, across the whole existing archive.

## 0. Why corpus integrity is a hard dependency

`build_chunks` cuts a chunk boundary at every speaker change. The moment
`SPEAKER_00` and `SPEAKER_01` both resolve to `"Jacob Shapiro"`, that boundary
disappears, adjacent chunks merge, and the episode's chunk count **drops**.
`upsert` can only overwrite indices `0..n-1`, so every index above the new count
survives, still carrying `[SPEAKER_01]` in its document text and duplicating
passages that now live inside the merged chunk.

This feature corrupts the corpus on today's write path. Full-replacement writes
(`delete(where=<triple>)` before upsert) must land first. Spec A also makes it
safe for this work to add a metadata key, since Chroma's upsert *merges*
metadata rather than replacing it.

## 1. Measured evidence

All figures from the 400-transcript local mirror (59 Geopolitical Cousins, 341
Jacob Shapiro, **0 Observing Japan**) unless stated. 42 clusters were audited by
hand, each judged independently of the rule that named it, with unresolvable
cases counted as errors.

| Rule | Audited | Correct | Wilson 95% | Clusters | % clusters | Speech | Episodes |
|---|---|---|---|---|---|---|---|
| R1 first-person self-ID | 12 | 12 | [0.76, 1.00] | 265 | 28% | 100h (25%) | 265/400 |
| R2 vocative (GC only) | 15 | 15 | [0.80, 1.00] | 125 | 13% | 75h (19%) | 58/59 GC |
| Combined | 42 | 41 | **[0.877, 0.996]** | 382 | 40% | 171h (43%) | 314/400 |

Three findings that shape the design:

**The dominant-cluster heuristic is dead.** The cluster that self-identifies as
Jacob is the *second*-largest in 191 of 283 Jacob Shapiro episodes and the
largest in only 86 — on an interview show the guest out-talks the host. Naming
by dominance would misattribute the host in roughly two-thirds of that show.

**The one false positive defines the rule.** Geopolitical Cousins episode 6,
`SPEAKER_01` says *"reach out to us at jacob at jacobshapiro.com"* and also
*"**Jacob and Marco** had a fantastic conversation today"* — third person about
both hosts. It is the produced intro voice. **Reading out someone's email
address is not claiming to be them**, so R1 requires *first person*, and the
class is eliminated by construction rather than by a blocklist. The refinement
also *raises* coverage — 265 clusters against 82 — because self-introduction is
far more common than an email read-out.

**The text rules cannot serve Observing Japan.** Both hardcode "Jacob" and
"Marco" and name zero clusters there. The show with the least stable host set is
exactly the one the cheap approach cannot reach, which is the whole argument for
phase 2.

## 2. Scope

**Phase B1 — text baseline.** Two independent rules, names written across the
entire existing archive. No GPU beyond the re-embed, no audio download, no
enrolment, no biometric data.

**Phase B2 — voice model.** ECAPA voiceprints, with enrolment centroids derived
automatically from B1's high-precision output. Extends coverage to Observing
Japan, to guests-turned-regulars, and to any future feed. Reported as a
**delta over B1**, never as a standalone number.

**Out of both:** guest identification, cross-show identity resolution,
auto-enrolment of recurring unknowns.

## 3. Phase B1 — the text baseline

### Two matchers, kept separate

They ship and report separately. Their coverage is disjoint, their failure modes
are unrelated, and a blended precision number would hide that R2 contributes
nothing outside one of three shows.

**R1 — first-person self-identification.** Per-show patterns requiring a
first-person frame: `I'm <name>`, `I'm your host`, `email/write/reach **me** at
<address>`. Generalises to a new feed with one line of configuration.

**R2 — vocative inference.** For a show with a known two-person roster: the
cluster that addresses the *other* host by name, and is not itself named, is the
first host. Structurally limited to shows where the full roster is known.

**R2 guard.** R2 applies only when exactly two clusters exceed a speech-time
floor. With a third substantial speaker present, a guest saying "Marco" would be
named Jacob. The audited GC episodes decompose as two hosts plus a sub-150s
fragment plus a one-second `UNKNOWN` artefact — but Geopolitical Cousins 73
(Peter Zeihan and Matt Gertken, genuinely four speakers) is **absent from the
audited mirror**, so this case is untested and the guard is not optional.

### Conflict and fallback

If R1 and R2 disagree on a cluster, the answer is `unknown` — never a
tiebreak. If two clusters in one episode resolve to the same person, both fall
to `unknown`: that means diarisation over-split, and naming both would launder a
diarisation error into a false attribution.

Unnamed clusters become `unknown_1`, `unknown_2`, ordered by descending speech
time so re-runs are deterministic. This deviates from the design note's
`guest_1`: "guest" asserts something not established, since a host who failed a
rule is not a guest.

### Provenance

Every chunk gains `speaker_source` ∈ `{diarisation, text_r1, text_r2,
voiceprint}`. Consumers can filter by how a name was derived, and the eval can
be recomputed from the corpus without re-running the matchers. Safe to add only
because Spec A made writes full-replacement.

### Applying it to the archive

Names change the embedded document text, so this is a full re-embed of 438
episodes — roughly 28k BGE encodes through Spec A's `EMBED_ONLY` path, with no
Whisper, no alignment and no diarisation. The transcripts on the volume supply
the speaker labels and timestamps; only the names change.

This is the backfill the original design deferred. It is affordable here
because the text rules need no audio.

## 4. Phase B2 — the voice model

### Architecture

The decision logic must not import torch, so it runs on a laptop in about a
second with no GPU, model download or network.

```
speaker_id/
  spans.py        SpeechSpan; from_whisperx_segments() | from_transcript_text()
  embedding.py    ECAPA wrapper — the only torch-touching module
  voiceprints.py  VoiceprintStore: load/save, centroid maths, schema versioning
  policy.py       MatchPolicy(threshold, margin, min_speech_seconds)
  matching.py     pure: identify(cluster_vectors, store, policy)
  text_rules.py   R1 and R2 from phase B1
  eval.py         sweep, precision/coverage, Wilson interval, B1→B2 delta
```

`spans.py` carries the live/backfill symmetry: both parsers emit the same
`SpeechSpan` list, so `embed_spans(audio, spans)` serves either path unchanged.

### Model

`speechbrain/spkrec-ecapa-voxceleb` — 192-d, ~20 MB, VoxCeleb EER ≈ 0.8%.
Chosen over `pyannote/embedding` because it is **not gated** (the README already
documents pyannote's terms-acceptance as a runtime-failure footgun) and over
harvesting pyannote's internal clustering vectors because that would force a
full re-diarisation on any backfill. Verify the loading API against Context7
before coding; do not rely on remembered signatures.

### Enrolment without a human in the loop

Take the clusters R1 named at high precision — 265 of them, across 265 episodes
— embed them, and average per person. No hand-labelling, no clip-playing CLI,
and no confirmation-bias risk, because no human is being shown a guess.

The store is a Modal volume artefact, **never committed**. An ECAPA centroid is
a biometric template; stripping the name yields pseudonymisation, not
anonymisation, since anyone with a few seconds of the person's public audio can
match it back. The repository is public and MIT-licensed, so it **ships the
recipe, not the biometrics** — and reproducibility survives, because the
podcasts are public RSS and the derivation code is committed.

Store schema carries `model` and `dim`, checked on load with a raise on
mismatch: a store built with ECAPA vectors is meaningless against pyannote
vectors, and the failure would otherwise be silent nonsense. It also records
`intra_cohesion`, the mean pairwise cosine within one person's samples — low
cohesion means the enrolment itself is wrong.

### Matching

```
per cluster, duration d, centroid v:
    d < min_speech_seconds              -> unknown
    best < threshold                    -> unknown
    best - second_best < margin         -> unknown
per episode:
    greedy by descending score; a name already taken -> unknown
```

### Fail-closed, and degradation

If the store is missing, fails its model/dim check, or ECAPA will not load: log
a warning and fall back to B1's text names, then to diarisation labels.
Identification is additive and must never break transcription.

## 5. Evaluation

**Gold set: 15 episodes, 5 per show — including Observing Japan**, which no text
rule can reach and which is therefore the only place phase B2's generalisation
can be measured at all. Every diarised cluster labelled with a name or "not a
known host". Stored as `eval/labels.json`: names attached to `(episode,
cluster)` pairs, no vectors, committable — who hosts a public podcast is public.

**Split:** tune and holdout, stratified by show so each contains all three.
Thresholds are selected on tune; the reported number comes from the untouched
holdout.

**Definitions.**

- **assignment** — an `(episode, cluster)` pair that received a name
- **precision** — correct ÷ assignments. The headline; its complement is the
  false-attribution rate
- **coverage** — speech-seconds in named clusters ÷ total speech-seconds
- **delta** — B2 coverage minus B1 coverage at equal or better precision. This
  is what justifies phase 2 existing

Precision is always reported with a **Wilson 95% lower bound**. The review's own
42-cluster audit returned 41 correct with a bound of 0.877 — a reminder that the
point estimate is the least interesting number in the table.

**Precision is the objective, not accuracy.** Misattributing a claim to a named
real person is worse than declining to attribute it, because this corpus feeds a
briefing that produces position signals. The threshold is tuned for precision
and the coverage it costs is reported, not hidden.

## 6. Pipeline integration

In `transcribe()`, between diarisation and writing the transcript: resolve names
and rewrite `segment["speaker"]` in place, so the saved transcript and the
Chroma chunks stay consistent. Resolution order is R1, then R2 under its guard,
then voiceprint, then `unknown_N` — with `speaker_source` recording which
answered.

## 7. Testing

Unit, CPU-only, no network, no GPU:

- **text rules** — R1 requires first person, and the produced-intro-voice case
  from GC episode 6 is a regression test that must return `unknown`; R2's guard
  refuses an episode with three substantial clusters; R1/R2 disagreement yields
  `unknown`; two clusters resolving to one person both fall to `unknown`.
- **matching** — each gate rejects independently; happy path assigns; per-episode
  contention resolves to one assignment plus one unknown; empty store yields all
  unknown; `unknown_N` follows descending speech time.
- **voiceprints** — round-trip; model and dim mismatch both raise; centroids
  L2-normalised; `intra_cohesion` matches a hand-computed value.
- **spans** — both parsers agree on the same episode; end time from the next
  segment's start, capped; `[UNKNOWN]` handled; malformed lines skipped.
- **eval** — precision, coverage, Wilson bound and delta against a hand-computed
  confusion matrix.

Fixtures are a seeded generator with controlled cosine separation, labelled
synthetic in code and in results: they validate the harness, explicitly not the
accuracy claim.

CI: GitHub Actions, ubuntu, Python 3.12, ruff and pytest. No GPU, no secrets,
no Modal.

## Acceptance

**B1:** `eval/results.md` reports R1 and R2 precision and coverage separately,
each with a Wilson lower bound, on the holdout. Names and `speaker_source` are
present across the archive. The GC episode 6 intro-voice case returns `unknown`.

**B2:** the same report adds voiceprint precision and the measured coverage
delta over B1, including a separate Observing Japan figure where B1 scores zero.
Removing the store degrades to B1 without failing. No voiceprint vector is
committed.
