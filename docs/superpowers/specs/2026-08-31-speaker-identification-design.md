# Design: speaker identification — stable named speakers across episodes

**Status: approved, not yet implemented. Written 2026-08-31.**

Implements [`docs/speaker-identification.md`](../../speaker-identification.md),
which states the problem and the rationale. This document is the buildable
design. Ships **after**
[corpus integrity](2026-08-31-corpus-integrity-design.md).

## Scope

**In:** the recurring hosts of the three feeds. Enrolment, cosine matching, a
tuned threshold, an `unknown` fallback, names written into new episodes, and a
measured precision number with a stated confidence interval.

**Out:** guest identification, retroactive backfill of the existing corpus,
auto-enrolment of recurring unknowns, cross-show identity resolution.

Backfill is deferred rather than dropped, and section 8 records what the design
does to keep it cheap.

## Evidence the premise holds

Checked against the live corpus on 2026-08-31 rather than assumed:

- **The `speaker` metadata field already carries real human names.**
  `Marko Papic` appears in 191 records, written by `upload_book.py`. Downstream
  consumers already tolerate a name where a `SPEAKER_XX` label would otherwise
  be, so this is a change of value, not of schema — as the design note claimed.
- **Diarisation produces few clusters per episode.** Corpus-wide the labels run
  `SPEAKER_00` … `SPEAKER_07`, with 00 and 01 accounting for 22,449 of 28,489
  records. Hosts dominate, as assumed.
- **Hosts are identifiable from the transcripts.** Geopolitical Cousins episode
  1: `SPEAKER_00` says "email me at jacob at jacobshapiro.com" and names Marco.

## 1. Architecture

The organising constraint: **the code that decides whether to name someone must
not import torch.** Everything from a cluster centroid to a name is pure
arithmetic over numpy arrays, so it runs on a laptop in about a second with no
GPU, no model download and no network — which is what makes it properly
testable.

```
speaker_id/                 pure python package, never imports modal
  spans.py        SpeechSpan; from_whisperx_segments() | from_transcript_text()
  embedding.py    ECAPA wrapper — the only torch-touching module
  voiceprints.py  VoiceprintStore: load/save, centroid maths, schema versioning
  policy.py       MatchPolicy(threshold, margin, min_speech_seconds)
  matching.py     pure: identify(cluster_vectors, store, policy) -> {cluster: name|None}
  eval.py         threshold sweep, precision/coverage, Wilson interval
enroll.py         local CLI: proposes clips, you label them, writes the store
tests/            pytest, CPU only, seeded synthetic fixtures
```

`spans.py` carries the backfill symmetry. `from_whisperx_segments()` and
`from_transcript_text()` emit the same `SpeechSpan` list, so
`embedding.embed_spans(audio, spans)` serves the live path and any future
backfill unchanged. Building both parsers now costs one function; retrofitting
the split later costs a rewrite.

## 2. Embedding model

`speechbrain/spkrec-ecapa-voxceleb` — 192-dimensional, roughly 20 MB, VoxCeleb
EER around 0.8%.

Chosen over `pyannote/embedding` (the design note's first suggestion) and over
harvesting pyannote's internal clustering embeddings, for two reasons:

- **It is not gated.** The README already documents pyannote's
  terms-acceptance as a footgun that fails at runtime rather than at setup.
  Adding a second gated model would add a second instance of that failure.
- **Backfill economics.** Harvesting pyannote's internal vectors costs nothing
  extra per live episode, but a backfill would then need a full re-diarisation
  per episode to obtain them. A separate pass needs only audio plus known time
  spans — and the spans are recoverable from the saved transcripts. The
  rejection is about cost asymmetry, not about coupling.

Cost of the separate pass: roughly 30–60 seconds of T4 time per episode, on top
of a pipeline that already runs Whisper large-v2 and alignment.

**Verify before implementing:** the exact SpeechBrain loading API and the
current pyannote 3.x `return_embeddings` surface, via Context7. Do not code
against remembered signatures.

## 3. The enrolment store

```json
{
  "schema_version": 1,
  "model": "speechbrain/spkrec-ecapa-voxceleb",
  "dim": 192,
  "created": "2026-09-01",
  "people": [
    {
      "name": "Jacob Shapiro",
      "centroid": [ "...192 L2-normalised floats..." ],
      "n_samples": 5,
      "intra_cohesion": 0.87,
      "sources": [
        {"show": "Geopolitical Cousins", "episode": "1",
         "date": "2025-03-14", "start": 52.8, "end": 61.4}
      ]
    }
  ]
}
```

Two details that are not decoration:

- **`model` and `dim` are checked on load, and a mismatch raises.** A store
  built with ECAPA vectors is meaningless against pyannote vectors, and without
  the check the failure is silent nonsense rather than an error.
- **`intra_cohesion`** is the mean pairwise cosine among one person's own
  enrolment clips. A low value means the *enrolment* is wrong — two people
  labelled as one, or a clip containing crosstalk. Surfacing it at enrolment
  time catches label errors before they contaminate every downstream number.
  It is the margin check applied to the human step.

### Where it lives

**Modal volume only. Never committed.** An ECAPA centroid is a biometric
template. Stripping the name off one yields pseudonymisation, not
anonymisation: anyone holding a few seconds of the person's public audio can
compute an embedding and match it back. Since the repository is public and
MIT-licensed, the store stays out of it, enforced by `.gitignore`.

Reproducibility does not suffer, because the podcasts are public RSS and
`enroll.py` is committed: **the repository ships the recipe, not the
biometrics.** A stranger can re-derive the store from the same public audio.

## 4. Matching

Three gates per cluster, then a per-episode assignment.

```
per cluster c with centroid v and speech duration d:
    d < policy.min_speech_seconds     -> unknown   (not enough evidence)
    best < policy.threshold           -> unknown   (not confident)
    best - second_best < policy.margin -> unknown   (not unambiguous)

per episode:
    sort surviving candidates by score descending
    assign greedily; a name already taken -> that cluster falls to unknown
```

The margin gate exists because two hosts with similar voices otherwise become a
coin flip at the threshold boundary. The per-episode constraint exists because
two clusters in one episode both scoring as the same person means diarisation
over-split that person — assigning both would launder a diarisation error into
a false attribution.

**Fallback label: `unknown_1`, `unknown_2`, …**, ordered by descending speech
time so re-runs are deterministic. This deviates from the design note's
`guest_1`: "guest" asserts something we have not established, since a host who
fell below threshold is not a guest. `unknown_N` states exactly what is known.
As with `SPEAKER_XX`, the numbering is meaningful only within one episode.

## 5. Evaluation

The eval is the deliverable. Without a measured number this is a demo.

**Labelled set.** Six episodes, two from each of the three shows. For each
episode every diarised cluster is labelled with a host's name or "not a known
host". Stored as `eval/labels.json` — names attached to `(episode, cluster)`
pairs, no vectors, so it is committable. Who hosts a public podcast is public.

**Split.** Three tune (one per show), three holdout (one per show). The
threshold and margin are selected on tune; the reported number comes from the
untouched holdout. Reporting a number you optimised against is the most common
flaw in a portfolio ML project and avoiding it deliberately is worth stating in
the README.

**Definitions**, stated precisely because they are the artefact:

- **assignment** — an `(episode, cluster)` pair that received a name.
- **precision** — correct ÷ total assignments. The headline; its complement is
  the false-attribution rate.
- **coverage** — speech-seconds inside named clusters ÷ total speech-seconds.
  This is what the threshold costs.
- Precision is reported with a **Wilson 95% lower bound**. A precision of 1.00
  over roughly 20 holdout assignments has a lower bound near 0.84, and saying
  so is a stronger claim than "100% accurate".

**Procedure.** Sweep `threshold × margin` over the tune set, keep the setting
with the greatest coverage among those holding target precision, then evaluate
once on holdout. `eval/results.md` is committed and regenerable by
`python -m speaker_id.eval`.

**Precision is the objective, not accuracy.** Misattributing a claim to a named
real person is worse than declining to attribute it, because this corpus feeds
a briefing that produces position signals. The threshold is tuned for
precision deliberately, and the coverage it costs is reported rather than
hidden.

## 6. Enrolment tool

`enroll.py`, run locally:

1. Read transcripts to find recurring clusters and their total speech time.
2. Download the corresponding audio from RSS, cut 3–5 clean clips per candidate
   (single speaker, no overlap with an adjacent cluster, at least a few seconds).
3. Play each clip and prompt for a name, or `skip`.
4. Compute the centroid and `intra_cohesion`, warn if cohesion is low, write
   the store.

**The tool does not pre-fill a name guess**, even though transcript text often
reveals one. Pre-filling an answer into a labelling task invites confirmation
bias on exactly the artefact everything else is measured against. The friction
saved would be typing about five names once; the risk is contaminating ground
truth. Text-based hints stay available for a later auto-enrolment feature,
which is out of scope.

## 7. Pipeline integration

In `transcribe()`, between diarisation and writing the transcript file:

```python
name_map = identify_speakers(audio, result["segments"], store, policy)
for segment in result["segments"]:
    segment["speaker"] = name_map.get(segment.get("speaker"), segment["speaker"])
```

Rewriting in place means the saved transcript and the Chroma chunks both carry
real names and stay consistent, since `build_chunks` reads `segment["speaker"]`
and embeds it into the document text as `[{speaker}] {text}`.

**Identification is additive and must never break transcription.** If the store
is missing, fails its model/dim check, or the ECAPA model will not load, log a
warning and fall through to the diarisation labels. An episode with
`SPEAKER_XX` is a worse episode; an episode that failed to transcribe is a lost
one.

## 8. What this does for the deferred backfill

Recorded so the later decision is cheap rather than a rewrite:

- Audio is deleted after transcription (`transcribe.py`), so voiceprints can
  never come from local state — but the 438 transcripts on the volume retain
  `[SPEAKER_XX] <start>s - text`, so the **diarisation result survives even
  though the audio does not**, and RSS still serves the audio.
- A backfill is therefore: re-download audio, rebuild spans with
  `spans.from_transcript_text()`, run the same ECAPA pass, match, rewrite the
  transcript, re-embed. **Whisper large-v2, alignment and diarisation are all
  skipped.**
- The expensive half is the BGE re-embed, because the speaker label is inside
  the embedded document text. That cost is unchanged wherever it runs, and the
  user has local GPU capacity that may make it free.
- The corpus-integrity spec must land first: a backfill re-upserts every
  record, and would otherwise reproduce the ID ambiguity at 28k scale.

## 9. Testing

Unit, CPU only, no network, no GPU:

- **matching** — each gate rejects independently (below threshold, inside
  margin, too short); the happy path assigns; two clusters competing for one
  name resolve to one assignment plus one unknown; an empty store yields all
  unknown; `unknown_N` numbering follows descending speech time.
- **voiceprints** — save/load round-trip; a model mismatch raises; a dim
  mismatch raises; centroids are L2-normalised; `intra_cohesion` matches a
  hand-computed value.
- **spans** — whisperx segments to spans; transcript text to spans with the end
  time taken from the next segment's start and capped; the `[UNKNOWN]` label;
  malformed lines skipped rather than crashing.
- **eval** — precision, coverage and the Wilson bound against a hand-computed
  confusion matrix; the sweep picks the documented setting from a fixed grid.

**Fixtures are a seeded generator** producing clusters with controlled cosine
separation. They are labelled synthetic in the code and in `results.md`: they
validate the harness, explicitly not the accuracy claim.

CI: GitHub Actions, ubuntu, Python 3.12, ruff plus pytest. No GPU, no secrets,
no Modal.

## Acceptance

- `eval/results.md` reports holdout precision with its Wilson 95% lower bound
  and the coverage at the chosen threshold, regenerable from the committed
  labels and a locally derived store.
- A new episode transcribed through the pipeline carries host names in both the
  volume transcript and the Chroma `speaker` metadata.
- Removing the store degrades the pipeline to `SPEAKER_XX` without failing.
- The unit suite passes in CI with no GPU and no secrets.
- No voiceprint vector is committed to the repository.
