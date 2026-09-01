# Design: naming speakers where it matters

**Status: approved in brainstorming after three rounds of adversarial review,
not yet implemented. Written 2026-09-01.**

Supersedes the scoping half of
[`2026-08-31-speaker-identification-design.md`](2026-08-31-speaker-identification-design.md)
and corrects one of its claims (§1.3). Depends on
[`2026-08-31-corpus-integrity-design.md`](2026-08-31-corpus-integrity-design.md),
shipped 2026-09-01; `RULES_VERSION` and `episode_id_prefix` are consumed here.

## Part 0 — What three review rounds changed

Two earlier drafts of this spec were killed by review, and the failures share a
shape worth recording, because it governs the structure below.

- Draft 1 pre-registered a coverage gate a *perfect* matcher could not pass: the
  denominator was total speech, which includes guest speech the matcher never
  names, so the ceiling was each show's guest ratio. 66.7% on the show that is
  341 of 400 measurable transcripts.
- Draft 1 sized the eval set on ~1.5 named assignments per episode, never
  measured. The plausible range 1.15–1.96 swings the achievable bound from
  97.4% (fails) to 98.5% (passes).
- Draft 2's pilot was underpowered for the two quantities it existed to measure:
  at Marco Papic's 10.9% co-host rate, a 20-episode stratified pilot draws zero
  co-host episodes **44.8%** of the time.
- Draft 2 claimed to defer its gate and then pinned eval size at ">=150
  assignments", which is only meaningful as the n implied by a 98% bound. The
  gate was pre-registered through the back door.
- Draft 2's §4.2 said "crop each recorded turn", but a transcript *line* is a
  whisper segment, not a turn. **262,802 segments at a 4.2s median versus 19,782
  merged turns at 38.0s** — a 13.3x error in embedding count, and near
  worst-case input to a speaker-embedding model.

**Every round found the inputs wrong and the reasoning right.** The Wilson
closed form, the sampling rule and the leakage guard survived all three passes;
four ordinary numbers went in unmeasured and came out as gates. That is why this
draft ships a cheap tier first and derives its expensive tier's scope from
measurements rather than from argument.

## Part 1 — What ships

### 1.1 Two tiers, sized to where the difficulty actually is

| Tier | Method | Surface | Cost |
|---|---|---|---|
| **1** | Dominant cluster is the host, cross-checked by name grep | ~76% of the archive | No GPU, no model, no audio embedding |
| **2** | Voice embedding and matching | ~24% | GPU, model gating, embedding cache |

Measured: Geopolitical Cousins is 59/400 episodes (14.8%) and has two hosts to
separate; 37 of 341 Jacob Shapiro episodes (9.2%) mention Marco Papic in their
first 40 segments. Those 24.0% are where voice identification is *uniquely*
needed. On the other 76.0% — single-host episodes — the dominant cluster is the
only candidate, and no voice model is required to pick it.

Tier 2 is therefore scoped to **Geopolitical Cousins co-host separation and
cross-show host detection**, and is specified only after Tier 1 is measured.

### 1.2 Tier 1 ships names, on new episodes and on the archive

**New episodes.** The name is written at transcribe time into both the `speaker`
metadata and the chunk text. Free — the chunks are being built anyway.

**The archive.** A **metadata-only** `collection.update()`. No GPU, no re-embed.

### 1.3 Correcting the 2026-08-31 note

That note states: "Renaming retroactively is a re-embed, not a cheap
`collection.update()` on metadata." That is true for the *embedding* and false
for *attribution*, and the distinction decides the backfill's cost.

`mcp_server.py:122` builds each search result's speaker field from
`meta.get("speaker")` — it reads **metadata**, and never parses the document
text. So a metadata-only update makes every search result attribute correctly,
immediately, at Chroma-update cost alone.

What a metadata-only backfill does *not* buy is the name inside the embedded
string. `corpus/chunking.py:83` builds `f"[{speaker}] {text}"`, so archive chunks
keep `[SPEAKER_00]` in the vector and a query naming a person gets no lexical
boost from those records. That is a real but bounded loss, and re-embedding to
recover it is a separate, costed decision — not a precondition for shipping
attribution.

## Part 2 — Measured facts

All probed 2026-09-01. Assumptions are marked as such.

**Clusters per episode**, across the 400 transcripts in `downloaded/`: median 2,
mean 2.39, range 1–7. Geopolitical Cousins median 3 (n=59); The Jacob Shapiro
Podcast median 2 (n=341). Distribution: 1x17, 2x262, 3x89, 4x21, 5x5, 6x3, 7x3.

**Segments versus turns.** A transcript line is a whisper segment. Merging
consecutive same-speaker segments, as `corpus/chunking.py::build_chunks` already
does for chunking:

| | count | median duration |
|---|---|---|
| Whisper segments | 262,802 | 4.2s |
| Merged turns | 19,782 | 38.0s |

**Any voice work operates on merged turns, never on raw segments.** 13.3x fewer
embeddings, far better input to a speaker-embedding model, and the derived
end-time problem shrinks to 19,782 real speaker-change boundaries instead of
262,802 arbitrary ones.

**Speech concentration**, turns under 1.5s dropped:

| | pooled top-1 | pooled top-2 | median top-1 | episodes top-1 < 70% |
|---|---|---|---|---|
| The Jacob Shapiro Podcast | 66.7% | 99.1% | 68.2% | 193 / 341 |
| Geopolitical Cousins | 62.0% | 96.7% | 62.8% | 51 / 59 |
| All 400 | 65.7% | 98.6% | | |

389 of 400 episodes have a dominant cluster holding over 50% of speech, median
share 66.2%. **That the dominant cluster is the host is the HYPOTHESIS Tier 1
tests**, not a measured fact.

**Hosts per episode is UNKNOWN** and not derivable from the clustering: Jacob
Shapiro splits 66.7/99.1 top-1/top-2 and Geopolitical Cousins 62.0/96.7 — the
same shape, so "one host plus a guest" and "two hosts" are indistinguishable
without labels. Tier 1's design avoids needing this number; Tier 2's does not.

**`downloaded/` holds 400 of 438 and contains no Observing Japan episodes** —
verified twice, by filename-prefix enumeration and by a case-insensitive search.
Every figure above therefore describes two of three shows.

**Audio is not retained** (`transcribe.py:378` writes `/tmp`, `:455` deletes).
**Six episodes have no obtainable audio** (`FEED_UNREACHABLE`, aged off feed) and
can never be identified.

## Part 3 — Tier 1: the baseline, and how it is measured

### 3.1 The rule

For an episode with one enrolled voice, assign the show's host to the cluster
holding the most speech time. Assign nothing else. Every other cluster keeps its
`SPEAKER_XX` label.

**Cross-check, not label:** grep the episode's first 40 segments for the host's
name and for known co-host names. A hit for a *second* enrolled voice routes the
episode to Tier 2 rather than to a Tier 1 assignment.

### 3.2 The measurement is one clip per episode

Tier 1 makes **exactly one assignment per episode**, so its precision
denominator is one per episode and verifying it needs **one clip**: listen, and
answer whether the dominant cluster is the host.

This is what makes a real statistical claim affordable. Zero errors on 150
episodes gives a one-sided 95% lower bound of `150 / (150 + 1.645^2) = 98.2%`,
clearing the >=98% bar with margin — where draft 1's n=133 sat one episode from
failing on arithmetic alone.

**Budget: 150 episodes x one 10s clip = 25 minutes of audio.** Wall clock runs
roughly double once replay and typing are counted, so call it **45–60 minutes**.
Draft 2 estimated "twenty minutes" for a design that was 24 minutes of audio
before any replay; that estimate is not repeated here.

### 3.3 Sampling

Deterministic every-nth-by-date within each show, stride recorded. Not picked —
episodes that look easy to label are the ones with clean audio, exactly the
population the baseline performs best on.

Labelling proceeds **in deterministic order until 150 single-host episodes are
labelled**. The stopping rule keys on the label count, never on matcher output,
so it cannot select for a flattering result.

### 3.4 The gate

- **Zero misattributions** across the labelled episodes.
- **Coverage >= 90% of true host seconds** on single-host episodes.

Coverage is host-relative by construction: Tier 1 either names the dominant
cluster or names nothing, so the denominator is the host's own speech, not the
show's guest ratio. This is stated as a principle rather than derived from a
small pilot — 90% of a host's speech is what "attribution works" means, and it
is not a quantity a 20-episode sample should be allowed to set.

**Measured once.** If the gate fails, the labelled set becomes a dev set and a
fresh sample must be labelled before any new claim.

## Part 4 — Tier 2: scope, deferred

Specified after Tier 1 is measured, and scoped to what Tier 1 provably cannot
do: separating the two Geopolitical Cousins hosts, and recognising a known host
appearing as a guest on another feed.

Design decisions that already stand:

- **Enrolment is keyed by person, not (show, person)** — so Marco on the Jacob
  Shapiro feed is the same identity as Marco on Geopolitical Cousins. This is
  what makes cross-show queries answerable, and it is Tier 2's entire reason to
  exist.
- **Embed merged turns, not segments** (Part 2), weighting centroids by **true**
  turn duration while embedding a duration-capped crop. Those are two different
  quantities; conflating them silently under-weights the longest, cleanest turns.
- **Centroids are derived from labels, never stored.** `speakers/labels.json`
  holds provenance only. This follows `corpus/exclusions.py`, where the lists are
  derived because two hand-maintained lists drift **asymmetrically** — the stale
  one works right up until the moment it matters.
- **Assignment needs a threshold and a margin.** The dangerous case is two high
  scores a hair apart, which a threshold alone resolves confidently and wrongly.
- **`person: null` is ground truth**, not missing data. Naming such a cluster is
  a precision violation, and it is the one that matters most.

## Part 5 — The impurity probe

Per-cluster ground truth is only valid if diarisation clusters hold one person.
A cluster that is 80% host and 20% guest gets labelled, matched and scored
**correct** while a fifth of its text ships under a real person's name.

**This is a binary existence question, not a rate.** Draft 2 tried to estimate a
rate and could not: three fixed clips detect a 20%-contaminated cluster only
48.8% of the time and a 10%-contaminated one 27.1%, so a 5% threshold was being
tested with roughly 2x downward bias.

Instead: take **15 clusters** and label **10 randomly-drawn turns** from each.
Ten draws detect 20% contamination 89.3% of the time and 10% contamination
65.1%. **150 turn clips, 25 minutes of audio.**

**Decision rule.** If *any* cluster shows two speakers, per-cluster ground truth
is invalid and Tier 2 must score per-turn — decided before Tier 2 is built,
rather than discovered after it fails.

## Part 6 — Architecture

Two files, not four. 150 labels does not need a module hierarchy.

| Unit | Purpose |
|---|---|
| `corpus/speakers.py` | Pure: merge segments into turns, dominant-cluster rule, name grep, scoring. Tested. |
| `speaker_tool.py` | Modal app plus local entrypoints: cut clips, prompt, report. |
| `speakers/labels.json` | Ground truth. Hand-made, reviewable, in git. |

`corpus/speakers.py` imports no Modal and no audio library, so the rule and the
scoring run in the test suite on CPU — matching the split argued in
`corpus/showplan.py`'s module docstring.

The labelling prompt must have **name autocomplete from names already used**.
Free-text entry across 150 answers reliably produces `Jacob Shapiro` and
`J. Shapiro` as distinct people, which silently splits a centroid in Tier 2 and
is invisible in Tier 1's counts. It must also support undo and skip, and append
after every answer so an interrupted session resumes.

```
uv run modal run speaker_tool.py::cut_clips     # -> clips/ locally
uv run python speaker_tool.py label             # -> speakers/labels.json
uv run pytest tests/test_speakers.py            # rule, scoring, gate
```

## Part 7 — UNKNOWNs

1. **Whether the dominant cluster is the host.** Tier 1's whole hypothesis.
2. **Cluster purity.** Part 5. Invalidates per-cluster ground truth if it fails.
3. **Observing Japan's clustering and speech concentration.** Absent from
   `downloaded/`; every Part 2 figure covers two of three shows. Probe against
   the Modal volume before sampling.
4. **Hosts per episode.** Needed by Tier 2 only, and unmeasurable from the
   clustering. Tier 1 is deliberately designed not to need it.
5. **Hugging Face gating** (Tier 2 only). `README.md` records accepting terms for
   `pyannote/speaker-diarization` and `pyannote/segmentation`; whether that
   covers an embedding model is unknown, and a gated model fails at runtime with
   an authorisation error, not at setup.
6. **What the deployed image exposes** (Tier 2 only). pyannote is unpinned,
   arriving transitively via `whisperx==3.8.5` and frozen at build time. Per
   `docs/operations.md`, replicate the image spec **verbatim** so
   content-addressing reuses the cache; build lines mean the spec drifted.
   `.add_local_python_source("corpus")` stays **last**.
7. **Audio URL rot.** A sampled episode whose audio 404s is replaced by the next
   in deterministic order and **the substitution is recorded** — never silently
   skipped, which would reintroduce selection.

## Self-Review

**What this draft cut, and why.** Draft 2's three modules became two files; its
four pilot measurements became one hypothesis test plus one existence probe; its
uniform sampling became a stopping rule keyed on label count. The cuts follow
from Part 1's measurement that voice identification is uniquely needed on 24% of
the archive — spending the first build on the other 76% was the misallocation
both reviews named.

**What is genuinely new here rather than reorganised.** §1.3's finding that
attribution reads from metadata, verified at `mcp_server.py:122`, which turns the
archive backfill from a ~30k-chunk re-embed into a `collection.update()` and is
the reason this design ships something. And Part 2's segment/turn measurement,
which was a defect in draft 2 rather than a scope choice.

**Where this is still weak.** Tier 1's gate assumes single-host episodes can be
identified as such *before* labelling; the name grep is the mechanism and its
false-negative rate is unmeasured — an episode with an unannounced second host
would be scored as a Tier 1 failure when it is really a routing failure. Part 2
covers two of three shows. And Tier 2 remains genuinely unspecified; this
document should not be read as though its scope were settled beyond §4's
boundary.

**One boundary worth defending.** Re-embedding the archive so names enter the
vector text stays out, even though `RULES_VERSION` makes it schedulable, because
§1.3 shows attribution does not require it. Doing it anyway would spend ~30k
chunks of GPU to improve lexical matching on a name — a real benefit, but one
that should be argued on its own and priced against a corpus that now attributes
correctly without it.
