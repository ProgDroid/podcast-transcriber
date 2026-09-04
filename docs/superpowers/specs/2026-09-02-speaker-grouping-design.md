# Design: grouping voices, and naming them only when asked

**Status: DESIGNED, not built. Written 2026-09-02, after the three UNKNOWNs
blocking it were measured the same day and all three came back permissive.**

Supersedes Part 4 ("Tier 2: scope, deferred") of
[`2026-09-01-speaker-identification-eval-design.md`](2026-09-01-speaker-identification-eval-design.md),
whose design decisions survive intact and are carried forward here. That
document's Part 8 is the reason this one exists: the cheap text-only tier was
measured against 379 hand labels and failed, so voice is the only remaining
route to speaker identity, and its scope is the whole archive rather than a
21% residual.

Depends on
[`2026-08-31-corpus-integrity-design.md`](2026-08-31-corpus-integrity-design.md)
(`RULES_VERSION`, `episode_id_prefix`) and on §1.3 of the eval design, which
established that attribution reads from Chroma metadata and so the archive
backfill is a `collection.update()` rather than a ~30k-chunk re-embed.

## Part 0 — What was measured before this was designed

The eval design's resume block named two UNKNOWNs that "fail late and
expensively"; a third — how much of the archive can even be re-fetched —
followed from Part 8 moving Tier 2's scope to the whole corpus. All three were
settled on 2026-09-02, before any of the design below was argued. **This is the
first draft in this document's history to be written after its inputs were
measured rather than before.**

### 0.1 What the deployed image exposes (closed)

Probed by replicating `transcribe.py`'s image spec verbatim; Modal reused the
cache and printed no build lines, so these are the deployed versions and not a
fresh resolve.

| | |
|---|---|
| `pyannote.audio` | **4.0.7** |
| `torch` / `torchaudio` | 2.8.0+cu128 |
| `huggingface_hub` | 0.36.2 |
| `transformers` | 4.57.6 |
| `speechbrain` | **ABSENT** |

`speechbrain/spkrec-ecapa-voxceleb`, one of the two candidates named in
`docs/speaker-identification.md`, is **not reachable without an image change**
and is dropped from consideration. The pyannote major version is also newer
than the 3.x the eval design's language assumed — an unpinned transitive
dependency moved while the design was being written.

### 0.2 Hugging Face gating (closed, and it does not block)

| model | verdict |
|---|---|
| `pyannote/embedding` | **GATED** — clean 403 `GatedRepoError` |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | **PRESENT**, ungated, dim 256 |

The acceptance recorded in `README.md` does **not** extend to
`pyannote/embedding`. It does not need to: wespeaker loads with the current
token and, probed end-to-end, returns a real `(256,)` vector through
`Inference(model, window="whole")`. **Voice embedding is available today on the
current image and token, with no new terms to accept.**

The probe's positive control — constructing the diarisation pipeline exactly as
`transcribe.py:415` does — produced the more consequential finding. whisperx
loads **`pyannote/speaker-diarization-community-1`**, which no document in this
repo names, and that pipeline carries its own embedder inline:

```
.embedding:  {'checkpoint': 'pyannote/speaker-diarization-community-1',
              'subfolder': 'embedding'}
._embedding: PyannoteAudioPretrainedSpeakerEmbedding
```

It also exposes `plda` / `_plda`: community-1 scores speaker similarity with
PLDA, not bare cosine. Both facts feed Part 3.

**`README.md:53-55` is therefore wrong about setup** — it instructs a new clone
to accept terms for two models the pipeline does not load. Whether community-1
is gated-and-already-accepted or simply ungated is **UNKNOWN** (§6.3); either
way the instructions do not describe what runs. Correcting them is out of scope
here and recorded so it is not lost.

### 0.3 How much of the archive can be re-fetched (closed)

Tier 2 needs audio and audio is not retained, so every episode must come back
from its feed. Part 7 §7 bounded this for one show and left two unmeasured.
Measured across all three by HEAD request (falling back to a one-byte ranged GET
where a CDN refuses HEAD, so a 405 is never misread as rot):

| show | total | fetchable | absent | dead |
|---|---|---|---|---|
| Geopolitical Cousins | 76 | 76 | 0 | 0 |
| The Jacob Shapiro Podcast | 356 | 350 | 6 | 0 |
| The Observing Japan Podcast | 7 | 7 | 0 | 0 |
| **TOTAL** | **439** | **433** | **6** | **0** |

**Tier 2's denominator is 433 of 439 — 98.6%.** The six losses are exactly the
six already in `speakers/unavailable.json` (the Jacob Shapiro feed's oldest,
May 2022), so the ledger was already complete and **rot is not spreading**.

Two guards make that negative trustworthy rather than merely printed. The
`>=90%` refusal from `cut_clips` was reused, because a half-failed feed fetch is
real here — Observing Japan returned 0 entries and then 7 for the identical feed
minutes apart on 2026-09-02 — and without it a transient failure records as
permanent rot. And the DEAD path was proved able to fire: a corrupted URL from
the same CDN returns `DEAD 404` while a real one returns `OK 172277513B`.
**Zero dead links is a measured negative, not a probe that can only say yes.**

Written to `speakers/audio_availability.json` with its inputs stamped.

### 0.4 Cost, computed rather than estimated

Over the 433 fetchable episodes, at `GAP_CAP_S=30.0` and `CLIP_MIN_TURN_S=12.0`:

| | |
|---|---|
| distinct speaker buckets | 1,067 (2.46 per episode) |
| of which are real `SPEAKER_*` clusters | **1,014** |
| real clusters with a croppable turn | **970** |
| crops at k=3 | **2,712** |
| audio to download | **56.1 GB** (433 files, mean 130 MB) |

**The 53-bucket gap is not rounding.** `chunking.py:71` substitutes `UNKNOWN`
for a segment carrying no speaker, and 53 episodes therefore hold an `UNKNOWN`
pseudo-cluster. It is audio belonging to nobody in particular, so building a
centroid from it would manufacture exactly the silent over-merge §3.2 names as
the error that matters. **Non-`SPEAKER_*` buckets are excluded by rule**, not
by luck: only 3 are croppable today, and a rule that holds at 3 keeps holding
when the count changes.

An earlier estimate in conversation put the crop count near 5,400. It was **2x
too high**, which is precisely why the number is computed here. The **44** real
clusters with no turn of at least 12s receive no centroid and stay unidentified;
that residual is measured and visible rather than discovered later (§6.2).

## Part 1 — What ships

### 1.1 The deliverable

**Stable, filterable speaker identity in MCP results, across the whole
archive.** Every transcript chunk gains a global identity so that "what has this
recurring voice said about Taiwan, and has the view moved" is answerable, and
answerable across shows.

Per Part 8 §8.6 the deliverable is **grouping, not naming**. `grp_007` is a
complete identity: it makes the query work. A name is an optional label attached
to a group afterwards, or never.

### 1.2 Four decisions taken by the user, 2026-09-02

1. **Identity reaches Chroma metadata**, not merely the rendered transcript — so
   cross-show filtering works rather than only cross-show reading.
2. **Seed the names from the existing labels, then name on demand.** No
   scheduled labelling session. Groups matching the 208 voice-verified labels
   inherit real names automatically; everything else ships as `grp_NNN` until
   the user happens to care about one.
3. **A fresh recognition gate before the archive is written**, sized at ~45
   pairs / ~15 minutes (§4.2).
4. **Two ships with a measurement between them** (§1.3), enrolment first — and
   the gate stays on Ship 2 rather than being replaced by a held-out split of
   the dev set.

Decision 2 follows the standing constraint that friction governs adoption: a
design that adds a step the user must remember loses to one that does not. Here
the remembering is put on the tooling — naming is available at the moment of
curiosity and never scheduled.

### 1.3 Build order: two ships, with a measurement between them

**The audio pass is unavoidable and costs the same under every design
considered.** To assign identity to a cluster in episode 312 you must fetch,
decode, crop and embed episode 312, whether you then compare its centroid to
five enrolled voices or to 969 other centroids. **No version of this skips the
56.1 GB or the GPU**, so the pass is not where a simpler design saves anything.
The saving is entirely in what is built *after* the vectors land.

That is where the build splits.

#### Ship 1 — enrolment against the known cast

Match clusters against the five people already labelled. This needs no
clustering code, no new labelling, and no time from the user.

**Its seeds cost zero new audio.** `clips/` still holds all 381 cut clips
(49 MB), and **all 208 `basis: voice` clips are present, none missing** —
verified 2026-09-02. So the five enrolment centroids are built from files
already on disk. Two consequences, both large:

- **§3.3's embedder question is settled before the pass, not after.**
  community-1's embedder versus wespeaker can be compared on real,
  voice-verified labelled clips at CPU cost. UNKNOWN 1 stops being a risk
  carried into a GPU run and becomes a local bake-off.
- Ship 1 delivers something usable on its own: the five known voices named
  across the whole archive, which is most of the archive's speech.

Assignment is §3.1's threshold-and-margin applied to a five-way decision. No
match writes nothing.

#### The measurement between the ships

After the pass, before a line of `corpus/grouping.py` exists: cluster the
unmatched centroids locally and **print the group-size histogram** — how many
identities recur across how many episodes. Seconds of compute on vectors
already held.

This answers the question Ship 2 assumes and no one has measured: **how many
recurring unnamed voices are there?** A one-off guest has no "has the view
moved" question, so grouping them buys nothing; a fat tail of recurring
unnamed voices is the whole case for Ship 2. **Pre-register the threshold
before the histogram is seen.**

**Checkpoint pre-registration, 2026-09-02.** Ship 2 is built if the
unmatched centroids yield **at least 15 groups appearing in 5 or more
distinct episodes**. Below that the recurring-unnamed-voice population is
too thin to justify archive-wide grouping, and Ship 1's five identities
stand as the deliverable. Fixed before the histogram is seen.

#### Ship 2 — grouping the unnamed

Archive-wide clustering, naming-on-demand, and the recognition gate (§4.2).
Built only if the histogram justifies it, and then built with the score
distribution in hand — which also closes §6.7, where threshold and margin are
deliberately left unset.

**This structure came from a fresh-context review** which observed that draft 1
committed to the grouping machinery before taking the one cheap measurement
that sizes it — the same error as its predecessor, which built a tier around a
coverage figure it had not measured.

**One part of that proposal was refused.** It also recommended replacing §4.2's
recognition gate with a held-out split of the 208 voice labels: zero user time,
a computed precision figure. Rejected for two reasons. §3.4 of the eval design
makes those labels a dev set, and the five-person cast list was *derived from
them*, so the enrolment set is defined by the thing being measured — honest for
"can we recognise these five", optimistic as a general claim. More decisively,
it cannot validate the deliverable: **no existing label can validate a merge
between two voices nobody has named**, which is precisely what Ship 2 does. The
gate is not replaced; it moves to where it is the only instrument that works.

### 1.4 What is deliberately NOT in scope

- **Re-embedding the archive so names enter the vector text.** §1.3 of the eval
  design shows attribution does not require it. It stays out for the same reason
  it stayed out there, and the boundary is now load-bearing: Part 2 below is
  designed specifically so the backfill never touches a document or an
  embedding.
- **Diarisation itself.** Clusters are taken as given, on Part 5's purity
  result — which must be quoted at its true strength: **0 contaminated of 12
  clusters that carried enough labels to judge**, with 3 unjudged, out of 15
  drawn. Not "0 of 15". The probe's own stated power was 89.3% at 20%
  contamination, so this licenses per-cluster centroids while leaving a real
  chance of undetected impurity — which is why §4.1's co-membership flag and
  Approach C's fallback both stay in the design.
- **Naming every group.** Most groups will never be named and that is the
  design, not a shortfall.

## Part 2 — Architecture

### 2.1 The audio pass, and why it emits raw crops

One Modal function, one episode per container, reusing the existing
`max_containers=8` cap — the same courtesy `cut_one_episode` already observes
toward a single podcast host, and the same protection against a rate-limit
midway through a run whose partial failures then look like audio rot.

Per episode: fetch audio, decode with the ffmpeg already in the image, and for
each cluster take the k longest turns at or above `CLIP_MIN_TURN_S`, crop each
with the same lead-in `pick_clip_turn` uses (the previous speaker's tail bleeds
across a boundary), embed, then **discard the audio**.

**Persist per-crop embeddings, not centroids.** The audio pass is the one
expensive and irreversible step in this design: **56.1 GB** of someone else's
bandwidth, and it cannot be re-run cheaply to change a parameter. So it emits
the most raw artifact that is still small — 2,712 vectors of 256 floats, single
digit MB. Everything downstream of it (k, the weighting, the centroid rule, the
clustering threshold, the margin) then becomes a **local** decision that re-runs
in seconds, and no tuning costs another GPU-hour or another 56 GB.

That figure is measured, by summing `content-length` across all 433 URLs (433
of 433 reported one). **Two estimates preceded it and both were wrong** — 43 GB
in this spec's first draft, and 25.5 GB from a reviewer computing 436.6 hours at
128 kbps. The real average is 286 kbps. A spec whose thesis is measured-not-
estimated does not get to swap one guess for another.

The one parameter this does not rescue is **crop window length**, which is fixed
at embed time. It is therefore hedged: embed each crop at two window lengths in
the same pass. GPU cost is minutes either way, and it converts the last
irreversible parameter into a local one.

Centroids are then **derived** from the persisted crops, weighted by **true**
turn duration while the embedded crop is duration-capped. Those are two
different quantities and conflating them silently under-weights the longest,
cleanest turns — carried forward unchanged from Part 4.

### 2.2 Where identity lives: add a key, never overwrite one

Reading the consumer side first found a trap that the obvious backfill walks
into. `chunking.py:84` bakes the speaker into the embedded document text —
every document is `"[SPEAKER_01] words..."` — and `rendering.py:29` strips that
prefix using **the metadata value**, `f"[{speaker}] "`. Overwrite metadata
`speaker` with `"Rob Larity"` and the strip stops matching: every result renders
as `[SPEAKER_01] words... | speaker: Rob Larity`, leaking the raw cluster label
back into the text it was meant to replace.

So the backfill is strictly additive:

| key | value | status |
|---|---|---|
| `speaker` | `SPEAKER_01` — the raw cluster label | **unchanged** |
| `speaker_id` | `grp_007` — the global identity | **new** |

and the name lives in neither. `speakers/names.json` maps `grp_007 → "Rob
Larity"`, resolved by the MCP layer at render time.

Three properties follow, and each is the reason for the shape:

- **Nothing existing changes.** The renderer's strip keeps working; no document
  and no embedding is touched; the boundary in §1.4 is kept by construction
  rather than by discipline.
- **It is inert, but NOT removable.** Any consumer that ignores `speaker_id`
  behaves exactly as it does today, so a wrong grouping is disarmed by ceasing
  to read the key. It cannot be *unwritten*: `corpus/writing.py`'s module
  docstring records that Chroma **merges** a record's metadata, so "a metadata
  key can never be removed by rewriting the episode, only by deleting the
  record" — and deleting 30k records is the re-embed §1.4 puts out of scope.
  **An earlier draft of this section claimed the corpus could be returned
  byte-identical to today. That was false**, and it was load-bearing: §4.2
  originally justified a small gate partly on reversibility. The correct
  claim is weaker and still sufficient — the change is additive and ignorable,
  never that it can be undone. Overwriting `speaker` would have been worse
  still, since the original value would then be unrecoverable by any means.
- **Naming costs no write at all.** Because the name is resolved at render, an
  edit to `names.json` propagates across all 433 episodes instantly.

That last point is the same derivation principle as `corpus/exclusions.py`: two
hand-maintained copies drift asymmetrically and the stale one works right up
until the moment it matters. One derived mapping cannot disagree with itself.

Filtering by person: the MCP layer translates a name to its group id and adds a
`speaker_id` clause to the `where` it already assembles at
`mcp_server.py:102-106`.
Because groups are keyed by **person, not by (show, person)** — Part 4,
unchanged — the same identity spans feeds, which is the entire reason this
exists.

### 2.3 The output contract

The deliverable is identity **in MCP results**, so the output line is part of
the design rather than an implementation detail. `format_result`
(`corpus/rendering.py:34-52`) currently emits `... | speaker: {speaker}` and
uses that same value to strip the document prefix. After this change it needs
two values, and there are **four states**, each specified:

| state | strip with | renders as |
|---|---|---|
| named group | `speaker` (cluster) | `speaker: Rob Larity` |
| unnamed group | `speaker` (cluster) | `speaker: unidentified voice (grp_007)` |
| no `speaker_id` (threshold/margin refused) | `speaker` (cluster) | `speaker: unidentified` |
| `speaker: UNKNOWN` | `speaker` (cluster) | `speaker: unidentified` |

The strip **always** uses the raw cluster label, never the display name. That
is the §2.2 trap, stated as a rule so it cannot be reintroduced.

A bare `grp_007` is never emitted alone: an LLM consumer will invent meaning
for an opaque token, so the group id appears only as a parenthetical after
words that say what it is. States three and four render identically on purpose
— the distinction between "declined" and "never had a label" is real but is not
the consumer's business.

**The filter needs designing too, not just enabling.** `search_podcasts` gains
a `speaker` parameter accepting a name or a group id; the MCP layer resolves a
name to its id and adds the `speaker_id` clause. Its **docstring is what drives
tool selection**, so it must say that speaker filtering exists and that
identities span shows — an undocumented parameter is an unreachable one.

### 2.4 The backfill

A metadata-only `collection.update()` over the archive's ~30k chunks. Each
chunk already carries its episode identifiers and its cluster label in
`speaker`, so the mapping from chunk to (episode, cluster) to `speaker_id`
needs no audio and no re-derivation.

**Read-modify-write, not blind write** (§6.3 records that Chroma's metadata
update semantics — replace versus merge — are unverified, and the safe
assumption is replace). `corpus/store.py`'s `paged_get_ids` is the existing
paging path.

Pre-register the chunk count and assert it in the script, so a collection that
moved since the probe causes a refusal rather than an unknown-size write.

### 2.5 Naming on demand

**The entrance must not require recall either.** `py speaker_tool.py groups`
lists recurring unnamed groups ranked by episode count — the discovery path,
without which the user would have to remember a group id to reach a prompt
designed to spare them remembering a name. A review of this spec's first draft
caught that omission: the prompt was the right shape and its entrance was not.

From that listing, `py speaker_tool.py name grp_007` plays three clips drawn
from that group and then asks for a name. The user is never asked to identify a voice from nothing —
only to name one just heard and already recognised. This is Part 8 §8.6's
correction applied to the tooling: **recognition, never recall.** The prompt has
a blank field where a proper noun goes, which by the standing rule means it must
be preceded by the audio that makes the field answerable.

**Seed only from `basis: voice`.** `labels.json` is 208 `voice` + 6 `context`
+ 165 `skipped`; the 6 were inferred from episode text, and `speaker_tool.py:705`
warns in its own comment that such labels score a matcher correct for the wrong
reason. Importing them would carry a text inference into voice ground truth —
the route Part 8.4 forecloses — so the seed filters on basis rather than on
presence of a name.

Seeding is otherwise automatic and free. Each of the 208 labels sits on an (episode,
cluster) pair; that cluster's centroid lands in some group; the group inherits
the name. Those labels are a **dev set** for any precision claim (§3.4 of the
eval design) and this is exactly what a dev set is for.

**And the old prompt is retired, not merely supplemented.**
`speaker_tool.py:691` still reads `input("  who is this? ")` — the precise
prompt Part 8.6 identifies as the mis-specification that produced 112 blank
answers out of 192. Adding a correct command beside it would leave the wrong
one reachable, and the next person to label would find it first.

## Part 3 — Grouping

### 3.1 The rule

Agglomerative clustering over the 970 cluster centroids, archive-wide, into
global identities. Assignment requires **a threshold and a margin**: the
dangerous case is two candidate identities scoring a hair apart, which a
threshold alone resolves confidently and wrongly. When threshold and margin
disagree, the cluster is assigned **no identity** — `speaker_id` absent.

**`speaker_id: null` is ground truth, not missing data.** Naming such a cluster
is the precision violation that matters most.

### 3.2 Under-merge is preferred to over-merge, deliberately

Two identities for one person is an annoyance the user can repair by naming both
groups. One identity for two people **silently attributes one person's views to
another**, which corrupts exactly the "has the view moved" query this design
exists to serve. The error is also invisible: nothing in the output distinguishes
a merged identity from a real one.

Thresholds are therefore set conservative, and the residual of unidentified
clusters is accepted as the price.

### 3.3 Which embedder

Two are reachable. The bake-off below settles which one to use, though its
second finding shows the two are near-identical in the space that matters
most, the raw vectors:

- **community-1's internal embedder — preferred.** It is the space the
  archive's clusters were actually cut in, and it brings the PLDA scoring the
  pipeline itself uses to decide "same speaker". Cosine on raw embeddings is a
  weaker instrument than what diarisation already achieved.
- **`pyannote/wespeaker-voxceleb-resnet34-LM` — the fallback.** Verified
  end-to-end in §0.2: it loads ungated and returns a `(256,)` vector.

**The preference rests on an unverified capability** (§6.1): the probe confirmed
community-1's embedder is *present and instantiated inside the pipeline*, not
that it can be used standalone on arbitrary crops.

**This is settled before the pass, on real data, for free.** All 208
`basis: voice` clips are on disk (§1.3), so the two embedders are compared by
building the five enrolment centroids with each and scoring the labelled
clusters — a CPU bake-off on voice-verified ground truth, run before a byte of
the 56.1 GB is spent. The choice is then measured rather than argued, and
UNKNOWN 1 stops being a risk carried into a GPU run.

If community-1's embedder cannot be used standalone, wespeaker is verified
end-to-end and the design proceeds unchanged.

**Bake-off pre-registration, 2026-09-02.** The embedder with the higher
leave-one-out accuracy over the 208 `basis: voice` clips wins. If the two
are within 2 percentage points, community-1 wins on the tie-break, because
it is the space the archive's clusters were cut in. This rule is fixed
before either number is seen. **The resulting figure is a dev-set
measurement and is not a precision claim.**

**Result, 2026-09-02 (`bakeoff_embedder.py`, `speakers/bakeoff.json`).**
Leave-one-out over the 208 voice-verified clips (207 scored, 1 skipped as
the sole clip for its person): both embedders score **1.000 accuracy
(207/207)**. Exactly tied, so the pre-registered tie-break applies and
**community-1 wins**.

**A second finding, not anticipated by the pre-registration: the two
embedders' raw vectors are numerically near-identical**, not merely
correlated. Cosine similarity between community-1's and wespeaker's vector
for the same clip is 0.9999999999808–0.9999999999994 across all 381 clips
(float32-precision-level agreement, ~3e-7 absolute per-component
difference) — the same order of magnitude as the accuracy tie being exact
to six decimal places in every reported statistic (threshold, margin, and
both score percentiles). Two independently-trained architectures scoring
identically on 207 held-out clips is possible; producing bit-level-adjacent
*vectors* is not, and is the stronger evidence. The likely explanation is
that community-1's `embedding` subfolder wraps the same
`wespeaker-voxceleb-resnet34-LM` weights used as the standalone fallback,
making this less a bake-off between two embedding spaces than a
confirmation that they are the same space. **This was not independently
verified against the model's config** (no local cache of the HF repo to
inspect) and is flagged for confirmation rather than asserted as fact. It
does not change the decision — the tie-break still selects community-1 —
but it means the §3.3 "brings the PLDA scoring the pipeline itself uses"
argument is the only real remaining distinction between the two choices;
the raw embedding space is not one.

**Suggested threshold and margin for Task 6, both embedders agreeing to
six decimal places: threshold=0.516, margin=0.074.**

If community-1's embedder cannot be used standalone, wespeaker is verified
end-to-end and the design proceeds unchanged.

## Part 4 — Validation

Two instruments, in this order, because one is free and one costs the user's
time. **The free one runs first and may change the sizing of the second.**

### 4.1 The seed conflict check — free, and it runs before the gate

If two labels that cannot describe the same person land in the **same** group,
that is an over-merge, detected automatically and for free. Two label pairings
qualify:

- **named vs. differently-named** — `Jacob Shapiro` and `Rob Larity` in one
  group is a contradiction.
- **named vs. `person: null`** — a null cluster landing in the Jacob Shapiro
  group is equally a contradiction. This roughly doubles the check's reach:
  labelled clusters go from 208 to 373, and covered episodes from 99 to 190 of
  439.

  **With a caveat that must not be lost.** The stored field is `basis:
  skipped`, not "not one of the named people". "A recognised voice whose name
  the labeller could not recall" is Part 8.1's *interpretation* of his account,
  and two other skip causes exist — an unintelligible clip, and fatigue. Either
  can conceal the host, and both raise a **false** alarm. That direction is
  fail-safe, since it spends attention investigating a group that was fine, but
  the detector rests on soft evidence and the spec should say so. It is not a
  new assumption: Part 8.1 already leaned on the same reading to place Tier 1's
  truth near its 29.2% floor.

**State its reach honestly, which an earlier draft of this section did not.**
That draft claimed "214 independent over-merge checks". It is neither 214 nor
independent: `labels.json` holds 379 entries covering only **5 distinct people**
plus 165 nulls, over a minority of the 970 clusters. Two *null* clusters merging
is undetectable — they may or may not be the same person — which is precisely
where most of the archive's risk lives, because Part 8.1 records the skips as
recognised-but-unnamed voices, i.e. the recurring guests this design exists to
group.

So: **the seed check's hits are a FLOOR on over-merging, never a ceiling.** It
finds breakage cheaply and proves nothing about the rate. Claiming otherwise
would repeat §8.3's error — the one that carried Tier 1's false coverage claim —
in a document written to avoid it.

### 4.1b Same-episode co-membership — the free detector with full coverage

**Two clusters from the SAME episode landing in one global group.** This needs
no labels, so unlike §4.1 it covers all 970 clusters rather than the 38% that
carry one — including the unnamed voices where §4.1 is blind and where most of
the population lives.

It is a **flag, not a gate**, because the signal is genuinely ambiguous in a
useful way. Diarisation can over-segment one person into two clusters, and
reuniting those is *correct* — arguably the feature working. Two genuinely
different people merging is the error. The flag cannot tell them apart, but it
partitions the archive into the cases worth looking at, at zero cost, and the
two outcomes are exactly the two things worth knowing.

It also answers a question Part 5 never asked. That probe tested whether one
cluster holds two people; it never tested whether one person holds two
clusters. **The over-segmentation rate is unknown**, and it falls out of this
same run for free.

This detector was raised by a fresh-context review of this spec's first draft,
which correctly judged it strictly better than §4.1 on coverage. §4.1 is kept
because the two answer different questions: co-membership finds merges *within*
an episode, the seed check finds them *across* episodes, and a recurring
co-host merged with the host across 149 episodes would be invisible to the
first and obvious to the second.

### 4.2 The recognition gate — pre-registered

**Belongs to Ship 2, not Ship 1** (§1.3). Ship 1 assigns only the five enrolled
identities and is validated by §4.1 and §4.1b. This gate exists for the thing
those cannot reach: a merge between two voices nobody has named.

The user is shown pairs of clips and answers only **same voice / different
voice**.

- **Within-group pairs come from DIFFERENT episodes.** A same-episode pair would
  only re-test diarisation, which is not what is on trial.
- **Between-group decoys** are an attention check on the *answering*, not a
  measurement of the system — under-merge is the accepted error. They exist so
  that "same" does not become reflexive.
- Order is randomised and the pair type is not shown, or the answer leaks.

**Sizing, and the trap it avoids.** The bar must be sized to the sample. For a
perfect score, the Wilson 95% lower bound is:

| pairs | bound at 100% correct |
|---|---|
| 30 | 88.7% |
| **45** | **92.1%** |
| 60 | 94.0% |
| 110 | 96.6% |
| 150 | 97.5% |

A `>=98%` bar is **unreachable below roughly 190 pairs however well the system
performs**. Pre-registering one at 45 would repeat the failure Part 0 of the
eval design records — a gate a perfect system cannot pass — for the fifth time
in this document's history. (During this design's own conversation, 110 pairs
was stated as sufficient for a 98% bound. It is not; it yields 96.6%. The error
was caught by computing the curve instead of quoting it.)

**PRE-REGISTERED, before any pair is drawn or scored:**

> 45 within-group pairs. The gate is **zero confirmed over-merges**, with a
> Wilson 95% lower bound of **>=92%** on the same-voice rate. This bounds
> **gross** over-merging. **It is not a 98% precision claim and must never be
> quoted as one.**

The gate is proportionate because the change it guards is **additive and
ignorable** (§2.2) — not because it is reversible, which §2.2 now records it is
not, and not because §4.1 bounds the error rate, which §4.1 now records it does
not. What remains is: a wrong grouping is disarmed by ceasing to read one
metadata key, and §4.1 catches gross breakage before the user is asked for a
minute of their time.

**Both of those supports were overstated in the first draft of this spec and
corrected the same day.** They are stated here in their weakened form because a
gate justified by a claim that does not hold is the failure this document's
lineage keeps repeating. If §4.1 surfaces over-merges, the sitting is re-sized
deliberately rather than by reflex.

## Part 5 — Files and tests

**Ship 1** — everything needed to name the five known voices archive-wide:

| file | role | tested |
|---|---|---|
| `corpus/embedding.py` | crop selection, duration-weighted centroids | unit, pure |
| `corpus/enrolment.py` | build centroids from labels, score, threshold+margin | unit, pure |
| `speaker_embed.py` | Modal app: fetch → crop → embed → discard | dry-run |
| `mcp_server.py`, `corpus/rendering.py` | resolve `speaker_id` → name; speaker filter | unit on render |
| backfill script | additive `collection.update()`, counts pre-registered | unit on the where-clause |

§4.1b's co-membership flag ships here too — it is a few lines and covers all
970 clusters, against the 373 the label check reaches.

**Ship 2** — built only if the histogram justifies it (§1.3):

| file | role | tested |
|---|---|---|
| `corpus/grouping.py` | archive-wide clustering, seeding, conflict detection | unit, pure |
| `speaker_tool.py` | gains `groups`, `name`, `gate-plan`, `gate-score`; retires the `who is this?` prompt | unit on plan/score |

The pure modules hold all the logic worth being wrong about and need no GPU,
no audio and no network, which is what allows them to be built test-first rather
than validated by running the expensive pass.

**A guarantee stated in prose is intent, not enforcement.** Each of these must
name the test that fails if it is false:

- the backfill never writes a document or an embedding
- `speaker` is never overwritten
- a cluster failing threshold-or-margin gets no `speaker_id`
- the renderer's prefix strip still matches after `speaker_id` exists

## Part 6 — UNKNOWNs

1. **RESOLVED 2026-09-02.** Community-1's embedder loads and runs standalone
   — `speaker_embed.py::embed_clips` produced 381 vectors from it with zero
   failures. The bake-off (§3.3) tied both embedders at 1.000 accuracy
   (207/207 leave-one-out), so the pre-registered tie-break selects
   **community-1**. The bake-off additionally found the two embedders'
   vectors are numerically near-identical (cosine 0.99999999998+ on every
   clip), suggesting community-1's embedder and the wespeaker fallback may
   share weights — flagged for confirmation, not yet verified against the
   model config, and does not change the decision.
2. **The 44 real clusters with no turn >=12s.** They get no centroid and no identity.
   Whether lowering the crop floor recovers useful ones is unmeasured; the floor
   trades embedding quality against coverage and the trade is not yet sized.
3. **Chroma metadata update semantics** — whether `collection.update()` replaces
   or merges a metadata dict. Assumed replace, hence read-modify-write. Cheap to
   settle against a scratch collection and must be settled before the backfill.
4. **Whether `pyannote/speaker-diarization-community-1` is gated.** It loads
   with the current token; that is consistent with both "ungated" and
   "gated and already accepted", and the two imply different `README.md`
   corrections.
5. **The over-segmentation rate** — how often diarisation splits one person
   into two clusters within an episode. Part 5 tested the inverse and never
   this. Unknown, consequential for reading §4.1b's flag, and free from the
   same run.
6. **How many recurring UNNAMED voices the archive holds.** Ship 2's whole
   justification (§1.3). Measured at the between-ships checkpoint, from vectors
   Ship 1 already produced — not before, and not by argument.
7. **Threshold and margin values.** Deliberately unset here. They are local
   decisions by §2.1's design and must be derived from the measured score
   distribution, not asserted in a spec.

## Self-Review

**What this draft cut.** Enrolment-only matching (Approach B) fails the chosen
deliverable, because grouping unknown voices *is* the feature rather than an
extra. Turn-level embedding (Approach C) costs ~4x and buys protection against a
failure Part 5 looked for and did not find; it is the documented fallback if the
gate shows over-merging traceable to impure clusters, not the first build.

**What is genuinely new rather than reorganised.** The additive-metadata design
in §2.2, which came from reading `rendering.py` rather than reasoning about it —
the obvious backfill breaks the renderer, and no amount of care about the
*write* would have caught a defect in the *read*. And §2.1's decision to persist
per-crop embeddings, which makes every parameter except window length a local
one, on the principle that the irreversible step should emit the most raw
artifact it can.

**Where this is weak.** §3.3's embedder preference rests on an unverified
capability, though the fallback is verified and the probe is cheap. Threshold
and margin are unset, so the thing most likely to determine whether this works
is not yet specified — deliberately, but the risk is real. §4.2's gate bounds
gross over-merging only, and its 92% figure will be tempting to quote as a
precision result; §4.2 says in terms that it is not one. The 44-cluster residual
is measured but its consequence for query coverage is not. And §4.1's null-label
detector rests on an interpretation of a `skipped` field rather than on the
field itself.

**What the first draft got wrong, corrected the same day.** A fresh-context
review of this document found six defects, all confirmed against the repo
before being accepted:

| claim in draft 1 | truth |
|---|---|
| "drop `speaker_id` and the corpus is byte-identical" | Chroma **merges** metadata; a key is removable only by deleting the record (`corpus/writing.py`) |
| "214 independent over-merge checks" | 5 distinct people over a minority of clusters; blind to null-vs-null, where most risk lives |
| "214 voice-verified labels" | 208 `voice` + 6 `context`; the 6 must not seed |
| "Part 5 measured them pure (0 of 15)" | 0 of **12** judgeable, 3 unjudged, at 89.3% power |
| "1,067 clusters" | 1,067 buckets, of which **1,014** are real; 53 are `UNKNOWN` pseudo-clusters |
| "43GB of bandwidth" | **56.1 GB**, measured from `content-length` |

Two of those — the reversibility claim and the 214 checks — were **load-bearing
for §4.2's gate sizing**, which is the one number a reader would act on. The
review also contributed §4.1b's co-membership detector (better coverage than
anything this draft had), §1.3's two-ship structure (the machinery was committed
before the measurement that sizes it), and §2.3's output contract (the
deliverable is "identity in MCP results" and draft 1 never wrote the output
line).

**The single most useful thing it found was already on disk.** `clips/` still
holds all 381 cut clips, and all 208 `basis: voice` clips are present with none
missing. Draft 1 treated the embedder choice as a risk to carry into a GPU run;
it is instead a CPU bake-off on voice-verified ground truth that can be run
today, and the enrolment seeds cost no new audio at all. Nothing had to be
computed to discover this — the artifacts of the failed Tier 1 evaluation were
simply still there, and no one had looked.

**One recommendation was refused, and the refusal is recorded in §1.3**: it
proposed replacing the recognition gate with a held-out split of the 208
labels. Cheaper, and it measures the wrong thing — no label validates a merge
between two voices nobody has named.

**What is different about this draft.** Every load-bearing number was measured
before the design was argued, and the ones that were not got caught by a review
that checked them. Four separate figures stated confidently in conversation
turned out wrong: the crop count (2x high), the pair count for a 98% bound (110
does not reach it; ~190 does), the download size (43 GB stated, 56.1 GB
measured — and a reviewer's 25.5 GB counter-estimate was wrong too, which is why
it was measured rather than argued), and the cluster count. **Every one was
caught by computing instead of quoting.** The four preceding drafts in this
document's lineage each died of exactly that failure, uncaught.
