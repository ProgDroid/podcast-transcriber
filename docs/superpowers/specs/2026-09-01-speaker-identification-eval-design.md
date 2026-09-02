# Design: naming speakers where it matters

**Status: MEASURED AND FAILED. Tier 1 does not ship — see Part 8. The gate was
run once as pre-registered, the dominant cluster is the host in under a third
of episodes, and the labelled set is now a dev set. Part 5 passed. Read Part 8
before anything else here. Written 2026-09-01, three defects
repaired the same day; Part 2 recomputed 2026-09-02.**

Its one precondition has shipped: `67e5720` made the MCP render emit `speaker`,
without which none of this reaches a consumer (§1.3). Since then
`corpus/speakers.py`, `corpus/transcripts.py`, `speaker_stats.py`,
`speaker_tool.py` and their tests have shipped; Part 2's figures are
reproducible rather than remembered, and the clip plan is built and reviewable
(Part 6). **The dominant-cluster rule itself is still unbuilt** — nothing yet
asserts that the dominant cluster IS the host, which is the hypothesis the
labelling exists to test — and so are the scoring and every gate.

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

**The archive.** Rewrite the transcript file's speaker labels **first**, then a
**metadata-only** `collection.update()`. No GPU, no re-embed.

**The transcript rewrite is not optional, and the reason is a trap.** A
metadata-only update changes `speaker` without changing the chunk text or the
`rules_version` stamp, and `corpus/identity.py:13-19` says that stamp exists to
answer "were those chunks built by current rules" — today, specifically, the
speaker labels. Both ways of resolving that look wrong at first:

- **Do not bump `RULES_VERSION`.** The records then claim to be current while
  their metadata disagrees with their own document text, and nothing detects it.
- **Bump `RULES_VERSION`.** `decide_action` returns `EMBED_ONLY` for every
  record stamped `"1"`, and `transcribe.py:313-325` re-chunks from the **stored
  transcript** — which still says `SPEAKER_00`. The nightly cron would then
  overwrite every name it just took a manual hour to assign. **The bump actively
  destroys the backfill.**

Rewriting the transcript files removes the dilemma: the transcript becomes the
single source of truth it already is everywhere else, a future re-embed
regenerates names rather than reverting them, and the bump becomes safe. The
metadata update is then only a shortcut to get attribution correct *before* that
re-embed is paid for.

### 1.3 Correcting the 2026-08-31 note — and a claim this spec got wrong first

That note states: "Renaming retroactively is a re-embed, not a cheap
`collection.update()` on metadata." That is true for the *embedding* and false
for *attribution*, and the distinction decides the backfill's cost.

**But attribution did not reach any consumer at all until `67e5720`.** An
earlier draft of this section cited `mcp_server.py:122` — which does copy
`meta["speaker"]` into the searcher's result dict — and concluded that
attribution therefore worked. It did not. Both MCP tools rendered their output
from a separate f-string ninety lines further down that **never emitted
`speaker`**, so the only speaker text any caller ever received was the
`[SPEAKER_00]` prefix inside the document. The docstring and the README
advertised attribution the entire time.

That was the-probe-measured-the-wrong-layer: a correct fact about the dict,
used to answer a question about the rendered string. The tell was available and
missed — had attribution worked, `[SPEAKER_00]` would already have been a
visible annoyance in daily use.

`67e5720` moved the render into `corpus/rendering.py` and made the metadata
field authoritative for what a caller sees. **Only now** is a metadata-only
`collection.update()` sufficient to change attribution, at Chroma-update cost
alone. This section is a precondition of the design, not a free property of it.

What a metadata-only backfill does *not* buy is the name inside the embedded
string. `corpus/chunking.py:83` builds `f"[{speaker}] {text}"`, so archive chunks
keep `[SPEAKER_00]` in the vector and a query naming a person gets no lexical
boost from those records. That is a real but bounded loss, and re-embedding to
recover it is a separate, costed decision — not a precondition for shipping
attribution.

## Part 2 — Measured facts

Recomputed 2026-09-02 over all 439 transcripts on the `podcast-transcripts`
volume. Assumptions are marked as such.

**Read the provenance before the numbers.** This section's first version was
computed once by hand; the script never existed in the repository, and a figure
that cannot be recomputed is not a measurement. Every number below now comes
from `corpus/speakers.py` driven by `speaker_stats.py`, and reproduces with:

```
uv run python speaker_stats.py --dir downloaded --cap 30
```

**Durations are derived, and the derivation is part of the figure.** A
transcript line carries a start time and nothing else, so a segment's duration
is the gap to the next segment's start, clamped to `GAP_CAP_S = 30.0`, and an
episode's final segment contributes nothing because it has no successor. The
cap is close to a no-op here: across 297,726 inter-segment gaps, p50 is 4.1s,
p99 21.9s, p99.9 29.3s and the maximum 127.1s, so a 30s ceiling trims **0.01%**
of derived time and only one gap in the corpus exceeds 60s. Whisper segments
are effectively contiguous; this corpus has no ad-break or music artefact to
correct for. `MIN_TURN_S` remains 1.5s.

**The replaced figures are reproduced, so what changed below is population and
not method.** Running this code uncapped over a byte-identical copy of the
400-file 2026-05-06 snapshot returns every COUNT exactly — 262,802 segments,
19,782 turns, clusters 2 / 2.39 / 1–7, the histogram 1x17, 2x262, 3x89, 4x21,
5x5, 6x3, 7x3, Geopolitical Cousins median 3 (n=59), Jacob Shapiro median 2
(n=341), 51 of 59 GC episodes under 70% — and every percentage to within 0.4pp.
The residual is consistently negative and moves exactly one episode across the
70% line and one across the 50% line. Its cause is end-of-episode handling and
it is **not recoverable**: the original rule was never written down.

**Clusters per episode**, across all 439: median 2, mean 2.46, range 1–7.
Geopolitical Cousins median 3 (n=76); The Jacob Shapiro Podcast median 2
(n=356); The Observing Japan Podcast median 2 (n=7). Distribution: 1x18, 2x273,
3x104, 4x27, 5x10, 6x4, 7x3.

**Segments versus turns.** A transcript line is a whisper segment. Merging
consecutive same-speaker segments:

| | count | median duration |
|---|---|---|
| Whisper segments | 298,165 | 4.1s |
| Merged turns | 23,852 | 31.4s |

**Any voice work operates on merged turns, never on raw segments.** 12.5x fewer
embeddings, far better input to a speaker-embedding model, and the derived
end-time problem shrinks to 23,852 real speaker-change boundaries instead of
298,165 arbitrary ones.

**`build_chunks` is not a source of turns, and an earlier draft said it was.**
It does merge consecutive same-speaker segments, but it also splits a long run
at `MAX_CHUNK_WORDS` and carries `CHUNK_OVERLAP_WORDS` across every boundary,
and it records `start_time` while never computing a duration. Its chunk count
is therefore not a turn count and cannot become one. `corpus/speakers.py`
implements the merge separately, and `tests/test_speakers.py` pins the
distinction with a fixture long enough that `build_chunks` splits it and
`merge_turns` does not.

**Speech concentration**, turns under 1.5s dropped:

| | pooled top-1 | pooled top-2 | median top-1 | episodes top-1 < 70% |
|---|---|---|---|---|
| The Jacob Shapiro Podcast | 66.5% | 98.7% | 68.1% | 198 / 356 |
| Geopolitical Cousins | 60.4% | 94.9% | 62.2% | 66 / 76 |
| The Observing Japan Podcast | 61.0% | 100.0% | 63.1% | 6 / 7 |
| All 439 | 64.9% | 97.8% | 66.2% | 270 / 439 |

421 of 439 episodes have a dominant cluster holding over 50% of speech, median
share 66.2%. **That the dominant cluster is the host is the HYPOTHESIS Tier 1
tests**, not a measured fact.

**Seconds and words disagree, in one direction, on every show.** The two are
computed independently and never derived from one another, so the gap is a
finding rather than rounding: the top-1 cluster's share of *words* runs below
its share of *seconds* by 2.0pp on Jacob Shapiro, 5.4pp on Geopolitical
Cousins and 1.6pp on Observing Japan. The dominant speaker holds more airtime
than text — they speak in longer stretches and absorb the inter-segment gaps
that a start-only transcript cannot distinguish from speech. It does not flip
which cluster is dominant on any show, which is all Tier 1 needs. It does bear
on the **coverage gate**, which is written in seconds: 90% of a host's seconds
is a slightly easier bar than 90% of their words.

**The corpus has changed shape since the 2026-05-06 snapshot, and a stale
sample would have hidden it.** The 39 episodes added since, measured alone:

| | older 400 | recent 39 |
|---|---|---|
| Clusters per episode | median 2, mean 2.39 | median 3, mean 3.15 |
| Merged turns, median | 38.4s | 13.3s |
| Segment-to-turn reduction | 13.3x | 8.7x |
| Pooled top-1 / top-2 | 65.6% / 98.6% | 59.2% / 91.3% |

Composition explains part of it — Geopolitical Cousins is 44% of the new set
against 15% of the old — but not all.

**It is a step change at 2026, not a blip and not a gradual trend**
(`speaker_stats.py --by-year`, 2026-09-02). A shift measured only against
"everything before it" cannot tell those apart, because the same recent
episodes are both the signal and the whole of the recent bucket. Split by
calendar year, the Jacob Shapiro Podcast gives each era its own n:

| Jacob Shapiro | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| n | 53 | 110 | 85 | 71 | 37 |
| median clusters | 2 | 2 | 2 | 2 | **3** |
| pooled top-2 | 99.5% | 98.7% | 99.4% | 98.5% | **96.5%** |
| median turn | 49.7s | 57.1s | 57.3s | 47.0s | **20.5s** |

Four stable years, then a break. The 2026 cohort is 37 episodes rather than the
15 an earlier draft of this paragraph worried about, so the small-sample doubt
is resolved. Geopolitical Cousins moves the same direction more gently (pooled
top-2 97.4% → 92.7%, median turn 19.7s → 13.3s across 2025 → 2026).

Note the shape: in 2026 the top-1 cluster's share **rises** to 70.0% while
top-2 falls — more speakers, with the dominant one talking more, which is what a
multi-guest panel with a moderating host looks like. **An alternative this data
cannot rule out is a diarisation change rather than a format change.** Audio is
not retained (`transcribe.py`), so testing it means re-fetching audio for an
older episode and re-running the current image; that is possible for episodes
still on their feed and impossible for the six that have aged off.

**The consequence for §3.3 is concrete.** 2026 is 84 of 439 episodes (19.1%),
so a deterministic every-nth-by-date stride draws roughly four fifths of the
precision set from the four stable years and about 29 episodes from the era the
corpus is actually moving into. The gate would then be measured on the past and
the matcher deployed on the future — a sampling rule that cannot produce an
unflattering result, which is Part 0's failure mode with a date on it.

**Resolved 2026-09-02: the gate splits at 2026-01-01 and both strata must clear
zero misattributions** (§3.3, §3.4). Sizing the recent stratum then turned up a
hard ceiling — the eligible 2026 population is 42 episodes, so it is labelled
exhaustively rather than sampled, and its bound is capped at 93.9% by how many
episodes the shows have published rather than by how much labelling is done.
Demanding 98% of both strata would have been a bar no matcher could clear.

**Hosts per episode is UNKNOWN** and not derivable from the clustering: Jacob
Shapiro splits 66.5/98.7 top-1/top-2 and Geopolitical Cousins 60.4/94.9 — the
same shape, so "one host plus a guest" and "two hosts" are indistinguishable
without labels. Tier 1's design avoids needing this number; Tier 2's does not.

**Observing Japan is the sharpest case of that, and it is now measured.** Seven
episodes, cluster median 2, pooled top-2 of **100.0%** — every episode is
exactly two voices — with the dominant cluster holding 61.0% and 6 of 7
episodes under the 70% line. The published titles are uniformly "…, with
James David Malcolm" / "with Joshua Walker" / "with James Brown" / "with Dan
Sneider", so it reads as a one-host interview show. **That is the shape most
likely to break Tier 1**: on an interview show the guest can out-talk the host,
and a 61/39 split is not a comfortable margin. It is also the cheapest thing in
this spec to settle — 6 real episodes, labelled exhaustively rather than
sampled, needs no statistics at all.

**Its turn lengths sharpen the worry rather than easing it.** Observing Japan's
median turn is **71.8s**, the longest of any show in any year measured — against
20.5s for the Jacob Shapiro Podcast in 2026 and 13.3s for Geopolitical Cousins.
Long turns and exactly two voices is what an interview sounds like when the
guest answers at length, and a guest answering at length is precisely how the
dominant cluster ends up being the person who is *not* the host. Tier 1 would
then misattribute every such episode confidently — the one failure the design
forbids.

**Decided 2026-09-02: Observing Japan stays in, and all seven episodes are
labelled.** They form part of the exhaustive 2026 stratum (§3.3), so they cost
six clips and no statistics. Ruling the show out by argument would have been
cheaper and would have thrown away the only direct evidence available on how
the dominant-cluster rule behaves on an interview format — which the drift
above makes a question about the corpus's future, not about one dormant show.

**Correcting this section's account of `downloaded/`.** The earlier text said
`downloaded/` "holds 400 of 438 and contains no Observing Japan episodes —
verified twice". Both probes were right and the joined sentence was misleading:
it invites the reading that the 38 missing files *are* Observing Japan. They
are not. `downloaded/` was a **stale snapshot taken ~2026-05-06** (newest local
file: Jacob Shapiro 2026-05-06, Geopolitical Cousins 2026-05-05), and the
volume has since gained 15 Jacob Shapiro, 17 Geopolitical Cousins and 7
Observing Japan episodes — 39, giving today's 439. Observing Japan was absent
for a mundane reason: **its first episode is 2026-05-12, six days after the
snapshot.** The show did not exist yet. `downloaded/` has been refreshed from
the volume and every figure above covers all three shows.

**Observing Japan is dormant, not broken.** Its feed returns HTTP 200 with
exactly 7 items dated 2026-05-12 to 2026-06-19, matching the 7 transcripts
one-for-one; nothing is being dropped and `scheduled_job` is healthy. The show
has simply not published since 2026-06-19. Its 2026-05-12 entry is a 2.5 MB
enclosure against 37–57 MB for the rest and yields a 2.3 KiB transcript — a
trailer, correctly transcribed.

**An earlier draft of this section excluded that trailer from the eval and was
wrong to.** It is a real episode in the corpus, Tier 1 will assign a name to it
in production, and it is the *easiest* case on the show: a 17-segment solo
introduction by Tobias Harris, single-cluster, with a 102-second turn and
comfortably clippable. Dropping it would have been dropping an episode for being
atypical — the selection bias this design rejects everywhere else. It was caught
only because `speaker_tool.py plan` counted 42 where this document had said 41;
the tool was right. All seven episodes are in, 1.6% of the archive, on a show
that may never grow.

**Audio is not retained** (`transcribe.py:378` writes `/tmp`, `:455` deletes).
**Six episodes have no obtainable audio** (`FEED_UNREACHABLE`, aged off feed) and
can never be identified.

## Part 3 — Tier 1: the baseline, and how it is measured

### 3.1 The rule

For an episode with one enrolled voice, assign the show's host to the cluster
holding the most speech time. Assign nothing else. Every other cluster keeps its
`SPEAKER_XX` label.

**Cross-check, not label:** grep the episode for known co-host **surnames**. A
hit for a *second* enrolled voice routes the episode to Tier 2 rather than to a
Tier 1 assignment.

**Surnames, because forenames do not discriminate.** Measured over the 341
Jacob Shapiro transcripts:

| Probe | Episodes |
|---|---|
| `marco` in first 40 segments | 36 |
| `papic` in first 40 segments | **10** |
| `marco` OR `papic` in first 40 segments | 37 |
| `papic` anywhere in the full transcript | **24** |
| first-40 hit containing no `papic` anywhere | **27** |

An earlier draft routed on `marco OR papic` in the first 40 segments and took
the resulting 37 as co-host episodes. **27 of those 37 never say `papic`
anywhere** — the signal is dominated by Marco Rubio, in a geopolitics podcast.
Real co-host evidence is at most 24 episodes (7.0%); within the first 40
segments it is 10 (2.9%).

Two consequences. The window must be the **whole transcript**, not the first 40
segments, or 14 of the 24 are missed. And the routing rate is low enough that
**Tier 2's surface is a fifth of the archive, not a quarter.**

**Re-measured 2026-09-02 on the full 439.** `papic` anywhere in the transcript
now hits **26 of 356** Jacob Shapiro episodes (7.3%), against 24 of 341 (7.0%)
before — the rate is stable, and only **2 of the 37** 2026 episodes hit, which
is what sizes the 2026 precision stratum at 42 (§3.3). Tier 2's surface is
therefore Geopolitical Cousins (76 of 439, 17.3%) plus those 26 (5.9%) =
**23.2%**. The first-40-segment figures in the table above were measured on the
341-episode population and have not been recomputed; nothing depends on them,
since §3.1 already routes on the whole transcript.

The false-positive rate is itself unmeasured for the surname probe: `papic`
could appear because Marco is *discussed* rather than present. Routing on it
sends such an episode to Tier 2, which fails safe — Tier 2 declines to name
rather than misattributing — but it inflates Tier 2's workload by an unknown
amount.

### 3.2 Two instruments, because precision and coverage are different questions

An earlier draft used one sample for both. **One clip per episode measures
precision and cannot measure coverage**, and the draft let a single instrument
carry a gate that needed two.

**Precision — one clip per episode, 192 episodes across two strata.** Tier 1
makes exactly one assignment per episode, so its precision denominator is one
per episode and verifying it needs one clip: listen, and answer whether the
dominant cluster is the host. Zero errors on the 150 sampled pre-2026 episodes
gives a one-sided 95% lower bound of `150 / (150 + 1.645^2) = 98.2%`, clearing
the >=98% bar with margin — where draft 1's n=133 sat one episode from failing
on arithmetic alone. The 42 episodes of the 2026 stratum are labelled
exhaustively rather than sampled (§3.3) and carry their own, necessarily
weaker, bound (§3.4).

**Coverage — every cluster, 30 episodes.** Coverage asks what fraction of the
host's speech the named cluster holds, so its denominator is the host's *total*
seconds across the episode. Answering that requires knowing which of the
episode's *other* clusters are also the host — i.e. labelling **every** cluster,
not just the dominant one. One clip cannot see it, and 121 of 400 episodes have
three or more clusters, so host speech split across clusters is the live risk
rather than a hypothetical.

**Drawn as a subset of the precision set** (20 pre-2026, 10 from 2026), so
each coverage episode's dominant-cluster clip is already planned and is reused
rather than cut twice. Measured 2026-09-02: 30 episodes, **69 clusters**, of
which 30 are already precision clips — so coverage costs **39 new clips**, not
the ~72 estimated here from a mean cluster count.

**Budget, as planned rather than as estimated.** `speaker_tool.py plan`
produces **381 clips**: 192 precision (150 pre-2026 + 42 for 2026), 39 new
coverage clips once the shared dominant-cluster clips are counted once, and
Part 5's 150 turn clips. At 10s for precision and coverage and 5s for impurity
that is **51 minutes of audio**, not the 69 first estimated here — the
difference is entirely clip REUSE and the shorter impurity clip, not a reduced
scope. Wall clock runs roughly double once replay and typing are counted, so
call it **~1.7 hours**, resumable. Taking the 2026 era whole rather than
sampling it costs 42 clips, about seven minutes. Draft 2 estimated "twenty
minutes" for a design that was 24 minutes of audio before any replay; estimates
in this document are now stated as audio-minutes first and wall clock second,
because that is the error that keeps recurring.

### 3.3 Sampling

**Two strata, split at 2026-01-01**, because Part 2 measured a step change
there and a single stride would draw four fifths of the set from the era the
corpus is leaving.

**Pre-2026 — sampled.** Deterministic every-nth-by-date within each show,
stride recorded. Not picked — episodes that look easy to label are the ones
with clean audio, exactly the population the baseline performs best on.
Labelling proceeds **in deterministic order until 150 single-host episodes are
labelled**. The stopping rule keys on the label count, never on matcher output,
so it cannot select for a flattering result.

**2026 — exhaustive, not sampled.** The eligible population is **42 episodes**
and is small enough to take whole: Jacob Shapiro's 37, less the 2 that the
co-host surname check routes to Tier 2, plus Observing Japan's 6 real episodes.
Geopolitical Cousins is excluded throughout — it is a two-host show and is
Tier 2's remit by §1.1, not a Tier 1 precision case.

Taking the era whole removes the sampling question from it entirely: no stride,
no stopping rule, no selection risk, and no argument about representativeness,
because it *is* the population. What it cannot do is grow. A census of 42 is a
complete statement about episodes published to 2026-09-02 and a weak one about
episodes not yet published, and §3.4 says so in the only place that matters.

### 3.4 The gate

- **Zero misattributions on the pre-2026 precision set** (150 episodes,
  sampled). One-sided 95% lower bound `150 / (150 + 1.645^2)` = **98.2%**.
- **Zero misattributions on the 2026 set** (42 episodes, exhaustive).
- **Coverage >= 90% of true host seconds** on the 30-episode coverage set.

**Both precision strata must clear zero. Neither substitutes for the other, and
the pooled figure is not the gate** — a matcher that works on the old format and
fails on the new one clears any pooled bar at a 4:1 mix, which is precisely the
failure the split exists to catch.

**The two strata do not carry the same bound, and requiring that they did would
be Part 0's mistake again.** Zero errors on 42 supports `42 / (42 + 1.645^2)` =
**93.9%**, not 98.2%, and no amount of labelling changes that: 42 is the entire
eligible 2026 population, so the bound is capped by how many episodes the shows
have published, not by effort. Writing "98% on both" into this gate would have
pre-registered a bar a *perfect* matcher cannot pass — the same shape as
draft 1's coverage denominator.

So the gate promises two different things on purpose. On pre-2026, 81% of the
archive: 98.2% precision, estimated from a sample. On 2026: **no misattribution
anywhere in the era**, counted rather than estimated, projecting to future
episodes at 93.9%. The second is weaker as a projection and stronger as a
statement of fact, and both are worth having.

The two bullets are measured by two different instruments (§3.2) and must not
be collapsed into one sample.

Coverage is host-relative by construction: Tier 1 either names the dominant
cluster or names nothing, so the denominator is the host's own speech, not the
show's guest ratio. This is stated as a principle rather than derived from a
small pilot — 90% of a host's speech is what "attribution works" means, and it
is not a quantity a 30-episode sample should be allowed to set. What the sample
does is *test* the principle, not choose it.

**Measured once.** If the gate fails, the labelled set becomes a dev set and a
fresh sample must be labelled before any new claim.

### 3.5 Clip selection

**Uniform at random over the cluster's eligible turns, seeded.** Not the
longest turn. The longest turn is the cleanest, longest uninterrupted speech in
the episode — exactly the condition the matcher performs best under — so
selecting it would bias the precision estimate the same way picking
easy-looking episodes would, one level further down. §3.3 rejects that at the
episode level; it has to be rejected at the turn level too or it comes back.

The seed is composed from a campaign seed and the episode's identity, so the
draw is reproducible from the record alone and a disputed clip can be re-cut.

**Redraws walk a seeded permutation.** A clip can be unusable — crosstalk,
music over the voice — and the prompt offers a redraw. Successive draws must
never return the turn just rejected, or "redraw until it sounds clean"
silently becomes the longest-clearest-turn selection this rule exists to
avoid. Every redraw is recorded with its draw index; an unrecorded redraw is
a silent resample.

**Measured 2026-09-02** (`speaker_stats.py --clippable`), at a 12s minimum
turn = 2s lead-in plus a 10s clip:

| stratum | n | dominant clippable | 1 eligible turn only | median eligible | all clusters | unclippable seconds |
|---|---|---|---|---|---|---|
| pre-2026 | 355 | 100.0% | 18 | 15 | 96.6% | 0.02% |
| 2026 | 84 | 100.0% | 3 | 23.5 | 89.9% | 0.05% |

Three things follow.

**The precision set does not shrink.** Every one of the 439 episodes has a
dominant-cluster turn long enough to clip, so unlike the 2026 population
ceiling in §3.3, nothing is lost here.

**21 episodes have no redraw path** (18 pre-2026, 3 in 2026): their dominant
cluster has exactly one eligible turn. An unusable clip there is a **drop**,
recorded as such and reducing the stratum's n — never a resample from a
shorter turn, which would be a clip the labeller cannot honestly answer.

**An unclippable cluster is an UNKNOWN, not a non-host**, and that matters
only for the coverage set, which labels every cluster. Assuming such clusters
are not the host overstates coverage; assuming they are understates it. The
size of that uncertainty is **0.02% of speech seconds pre-2026 and 0.05% in
2026** — three orders of magnitude below the 90% gate, so coverage records
them as unknown and proceeds. Note the shape: about a tenth of 2026's clusters
are unclippable and they hold a twentieth of one percent of the talking, so
cluster *count* materially overstates how many voices an episode really has.
That is consistent with the diarisation-change reading of Part 2's drift and
equally consistent with the format-change reading — more speakers also means
more small clusters — so it discriminates nothing and the unknown stands.

## Part 8 — RESULT: Tier 1 measured, and it fails (2026-09-02)

**The gate was run once, as pre-registered, and Tier 1 does not ship.**

381 clips cut, 379 labelled by ear. Scoring the precision set against §3.1's
rule -- the dominant cluster is the show's host:

| | planned | scored | correct | misattributions |
|---|---|---|---|---|
| pre-2026 | 150 | 72 | 51 | **21** |
| 2026 | 42 | 8 | 5 | **3** |
| The Jacob Shapiro Podcast | 185 | 78 | 54 | 24 |
| The Observing Japan Podcast | 7 | 2 | 2 | 0 |

The gate required **zero misattributions**. There are 24.

### 8.1 The verdict does not depend on the unlabelled clips

112 of 192 precision clips were skipped, so precision is bounded rather than
pointed:

| | |
|---|---|
| if **every** skip were the host | 87.5% (ceiling) |
| on identified clips only | 70.0% |
| if **no** skip were the host | 29.2% (floor) |

**Even the ceiling fails the >=98% gate by more than ten points**, so no
assignment of the unknown 112 rescues Tier 1. That is what makes this a
verdict rather than an estimate.

The labeller's own account settles where in the range the truth sits: the
skips are *recognised voices whose names he could not recall*. He named Jacob
Shapiro 148 times, so a skip is "familiar, but not the host" -- which is a
misattribution. **The realistic figure is near the 29.2% floor.** The dominant
cluster is the host in under a third of episodes.

### 8.2 The cross-check cannot be repaired

The failure is not that guests occasionally dominate. The Jacob Shapiro
Podcast has **recurring co-hosts who out-talk the host** -- Rob Larity appears
40 times in the labels and Marko Papic 19. **Larity was not in §3.1's roster
at all**; the design was scoped against a cast list that was wrong.

Completing the roster does not fix it, for two measured reasons:

- **Surface collapses.** `larity` appears in 149 of 356 Jacob Shapiro
  transcripts. A complete roster routes 161 of 356 to Tier 2, leaving Tier 1
  with 195 -- so its archive coverage falls from the claimed **~76% to 46%**
  (195 + 7 of 439). Tier 1's entire justification was covering three quarters
  of the corpus without a GPU.
- **7 misattributions survive it.** 13 of the 24 occur in episodes that never
  name the speaker anywhere, and 7 remain uncatchable even with every known
  surname in the roster. A text cross-check is structurally blind to a
  co-host who is present and unnamed.

### 8.3 The bound in §3.1 was inverted

§3.1 concluded "real co-host evidence is **at most** 24 episodes (7.0%)",
treating spoken-name hits as a ceiling on co-host presence. It is a **floor**.
Co-hosting without being named is invisible to a text probe, so the true rate
can only be higher -- and it is roughly four times higher. That single phrase
carried Tier 1's claim to cover 76% of the archive.

### 8.4 Text cannot identify speakers in this corpus

Two independent probes now fail the same way. §3.1's `marco` hit 36 episodes
of which 27 were Marco Rubio. And an attempt to recover the skipped names from
spoken introduction frames ("joined by X", "my name is X") fired on 55% of
episodes and returned Xi Jinping, Donald Trump, El Nino, New Zealand and Bruce
Willis -- people being *discussed*, not *speaking*.

**In a geopolitics corpus the discussed vastly outnumber the present, and both
appear in identical grammatical frames.** This is a property of the corpus, not
a weakness of one regex, and it forecloses every text-only route to speaker
identity.

### 8.5 What passed, and what carries forward

**Part 5 passed.** 0 of 15 clusters show more than one person (12 carried
enough labels to judge; 3 were skipped entirely). Per-cluster ground truth is
valid, so Tier 2 may score per-cluster rather than per-turn -- the decision
rule Part 5 existed to settle, settled.

Carried forward: 214 voice-verified labels across Jacob Shapiro (148), Rob
Larity (40), Marko Papic (19), Tobias Harris (6) and Matt Gertken (1) -- an
enrolment seed at the `basis: voice` standard. The clip pipeline, the
plan/dry-run/cut loop and the scoring transfer unchanged.

**Per §3.4 these labels are now a dev set.** Learning from them that Larity is
a co-host is exactly what a dev set is for; making any new precision claim
requires a fresh sample.

### 8.6 The labelling task was mis-specified

The prompt asked "who is this?", which demands **recall of a name**. What the
labeller can actually do -- and what the task needed -- is **recognition**:
"is this the same voice as that one?" Those are different cognitive jobs, and
the harder one was built.

This matters beyond ergonomics, because it is what a voice-embedding model
computes. **Grouping is not naming, and the corpus only needs grouping.**
Stable identity is what makes "what has this recurring voice said about
Taiwan, and has the view moved" answerable; the name is an optional label
attached to a group afterwards, or never -- `unknown_1` is a perfectly good
identity. The 2026-08-31 note listed auto-enrolment of recurring unknowns as
an optional extra. Given a listener who recognises voices but cannot name
them, it is the main feature.

### 8.7 The cost case inverts

Part 1 held that voice identification was *uniquely* needed on ~24% of the
archive and that a cheap tier covered the rest. The measurement says the cheap
tier's hypothesis is false wherever it was load-bearing, so voice
identification is needed on essentially all of it.

Tier 1 was attractive precisely because it needed no model, and that is the
same reason it could not answer the question. **Dominance is a fact about
airtime; identity is a fact about voice, and no amount of text analysis
converts one into the other.**

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
65.1%. **150 turn clips, 12.5 minutes of audio.**

**Drawn from the coverage set's clusters**, because those are exactly the
clusters whose per-cluster ground truth this probe exists to validate — testing
purity on clusters nothing depends on would answer a question nobody asked.

**Shorter clips at a lower floor: 5s at a 6s minimum turn, against 10s at 12s
elsewhere.** Deliberate, and the reason inverts the usual one. A cluster is
contaminated by BRIEF interjections from another voice, so a 12s minimum would
systematically exclude the very turns most likely to contain the second
speaker — the probe would then be least sensitive exactly where the fault
lives. 5s is enough to hear that the speaker changed, which is the only
question asked here.

Measured 2026-09-02: 11 of the 69 coverage clusters have fewer than 10 turns
even at the 6s floor and are not eligible; 15 are drawn from the remaining 58.

**Decision rule.** If *any* cluster shows two speakers, per-cluster ground truth
is invalid and Tier 2 must score per-turn — decided before Tier 2 is built,
rather than discovered after it fails.

## Part 6 — Architecture

Two files, not four. 192 precision labels does not need a module hierarchy.

| Unit | Purpose |
|---|---|
| `corpus/speakers.py` | Pure: merge segments into turns, dominant-cluster rule, name grep, scoring. Tested. |
| `speaker_stats.py` | Local, no Modal: recomputes Part 2 from a transcript directory. |
| `speaker_tool.py` | Modal app plus local entrypoints: cut clips, prompt, report. |
| `speakers/labels.json` | Ground truth. Hand-made, reviewable, in git. |

**Built as of 2026-09-02:** `corpus/speakers.py` (`merge_turns`,
`speech_shares`, `count_non_monotonic`, `pick_clip_turn`, `clip_window`,
`evenly_spaced`, `routes_to_tier2`, and the `GAP_CAP_S` / `MIN_TURN_S` /
`CLIP_*` constants); `corpus/transcripts.py`; `corpus/feed.py`'s
`episode_number_of`, extracted from `transcribe.py` so the clip tool resolves a
transcript filename back to its feed entry by the same rule that named the file;
`speaker_stats.py`; `speaker_tool.py` (`plan`, `cut_clips`, `label`);
`tests/test_speakers.py` (29 tests).

**The dominant-cluster rule and the scoring are still NOT built.** Nothing in
the repository asserts that the dominant cluster is the host — `dominant_speaker`
reports which cluster holds the most speech and says nothing about whose voice it
is. That assertion is the hypothesis the labelling exists to test, and writing it
alongside the labels is how a threshold gets chosen to fit them.

`speaker_stats.py` is separate from `speaker_tool.py` rather than another
entrypoint on it because it needs no Modal at all: the measurement must stay
runnable on a laptop with no Modal auth, which is the same argument
`corpus/showplan.py` makes for keeping planning off the GPU.

`corpus/speakers.py` imports no Modal and no audio library, so the rule and the
scoring run in the test suite on CPU — matching the split argued in
`corpus/showplan.py`'s module docstring.

The labelling prompt must have **name autocomplete from names already used**.
Free-text entry across 264 answers reliably produces `Jacob Shapiro` and
`J. Shapiro` as distinct people, which silently splits a centroid in Tier 2 and
is invisible in Tier 1's counts. It must also support undo and skip, and append
after every answer so an interrupted session resumes.

```
modal volume get podcast-transcripts / downloaded --force   # refresh the corpus
uv run python speaker_stats.py --dir downloaded --cap 30    # -> Part 2
uv run python speaker_tool.py plan              # -> speakers/clip_plan.json
modal run speaker_tool.py::cut_clips --dry-run  # resolve URLs, cut nothing
uv run python speaker_tool.py plan              # substitute what rotted
modal run speaker_tool.py::cut_clips            # -> clips/
uv run python speaker_tool.py label             # -> speakers/labels.json
uv run pytest tests/test_speakers.py            # rule, scoring, gate
```

**Plan, dry-run, re-plan until the dry run reports nothing unresolved.** The
loop is not ceremony: audio rot is only discoverable by asking the feed, and a
substitute can itself be rotted — the first round of substitutes here landed on
two episodes that were also gone. Two rounds converged; the cumulative ledger
is what makes it converge rather than oscillate.

**The dry run is worth its own step.** It resolves every clip's audio URL and
downloads nothing, so the shape of the loss is visible before any spend — and
critically, it tells you *which stratum* the loss falls in, which is the
difference between a substitution and a reduced bound.

**Feed resolution is flaky, and an empty result is not an empty feed.** This
same code returned 0 entries for Observing Japan and 7 for the identical feed
minutes later. Since an unresolved clip counts as reducing n, a transient
failure would silently shrink the eval and be indistinguishable from real rot.
`cut_clips` therefore asserts the resolved count against the episodes in the
plan and refuses on a shortfall rather than recording it as rot.

On Git Bash the volume pull needs `MSYS_NO_PATHCONV=1`, or the bare `/`
remote path is rewritten to a Windows path and the command fails with a bare
"No such file or directory" that names neither the path nor the cause.

## Part 7 — UNKNOWNs

1. **Whether the dominant cluster is the host.** Tier 1's whole hypothesis.
2. **Cluster purity.** Part 5. Invalidates per-cluster ground truth if it fails.
3. ~~**Observing Japan's clustering and speech concentration.**~~ **CLOSED
   2026-09-02.** Measured in Part 2: 7 episodes, cluster median 2, pooled top-2
   100.0%, top-1 61.0%. The show is dormant since 2026-06-19 and one of the 7
   is a trailer, so six usable episodes remain. Every Part 2 figure now covers
   all three shows. The scope question it left behind — whether a dormant
   six-episode interview show is worth Tier 1's remit — is **decided**: the
   show stays in, and all seven episodes are labelled as part of the
   exhaustive 2026 stratum (§3.3). It is the only direct evidence available on
   how the rule behaves on an interview format, which the drift below makes a
   question about the corpus's future rather than about one dormant show.
   **Opened and then closed by the same recompute:** the corpus's 2026 drift
   is real and is a step change, not a small-sample artefact — four stable
   years then a break, at n=37 (Part 2). §3.3 and §3.4 are amended for it. One
   residue stays open: whether the cause is a format change or a diarisation
   change is **UNKNOWN** and costly to test, since audio is not retained and
   six episodes have aged off their feeds entirely.
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
7. ~~**Audio URL rot.**~~ **IMPLEMENTED and measured 2026-09-02.** The Jacob
   Shapiro feed carries 350 of its 356 episodes; the missing six are its
   oldest, and four of them fell in the pre-2026 draw. They are substituted by
   the next available episode in date order and the substitutions are recorded
   in the plan. It took two rounds to converge — the first round's substitutes
   (Episodes 3 and 5) were themselves rotted. **No 2026 episode is affected**,
   so §3.4's 42 and its 93.9% bound stand as written.
   The two strata must differ here, and now do: pre-2026 is sampled and has a
   neighbour to substitute, while 2026 is exhaustive and has none, so rot there
   reduces n rather than being repaired. The ledger of unobtainable audio is
   `speakers/unavailable.json`, **versioned and cumulative** — it decides which
   episodes get substituted, and a run that overwrote it would erase the reason
   for its own substitutions and re-select them next time.
   The original rule, unchanged: a sampled episode whose audio 404s is replaced
   by the next in deterministic order and **the substitution is recorded** —
   never silently
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

**Three defects found after drafting, repaired here.** §1.3 cited the wrong
layer and concluded attribution worked when no consumer had ever seen a speaker
name. §3.1 routed on a forename and took 37 hits as co-host episodes when 27 of
them were a different Marco. §3.2 used one instrument for two gates, and the
coverage half was unmeasurable by it. All three were plausible numbers or
plausible readings that became load-bearing without a probe — the same failure
Part 0 records, recurring twice more after Part 0 was written.

**Where this is still weak.** The surname probe's own false-positive rate is
unmeasured: `papic` in a transcript may mean Marco is being discussed rather
than present, which inflates Tier 2's workload by an unknown amount (it fails
safe, since Tier 2 declines rather than misattributes). Tier 2 remains
genuinely unspecified. And the 90% coverage principle is asserted, not derived
— the 30-episode set tests it and cannot justify it.

**Amended 2026-09-02.** Part 2 now covers all three shows, and its figures are
reproducible for the first time. Two of its findings were not sought and are
the more consequential: the corpus has drifted toward more speakers and shorter
turns since May 2026, which puts §3.3's date-ordered sampling rule under the
same suspicion Part 0 records; and Observing Japan turns out to be a dormant
six-episode interview show, which is simultaneously the hardest case for Tier 1
and the cheapest to settle exhaustively. What made both visible was replacing a
remembered number with a recomputed one — the third time in this document's
history that an unmeasured input turned out to be load-bearing.

**Both were decided the same day.** The gate splits at 2026-01-01 with zero
misattributions required on each side, and Observing Japan stays in. Sizing the
recent stratum then produced the finding that matters most here: at 42 eligible
episodes it is capped at a 93.9% bound however much labelling is done, so the
two strata are deliberately promised different things. Phrased as "98% on
both" — which is how the split was first put — this document would have
pre-registered an unpassable gate for the fourth time. The only reason it did
not is that the population was counted before the bar was set.

**One boundary worth defending.** Re-embedding the archive so names enter the
vector text stays out, even though `RULES_VERSION` makes it schedulable, because
§1.3 shows attribution does not require it. Doing it anyway would spend ~30k
chunks of GPU to improve lexical matching on a name — a real benefit, but one
that should be argued on its own and priced against a corpus that now attributes
correctly without it.
