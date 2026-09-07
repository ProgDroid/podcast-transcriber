# Design note: naming speakers, and keeping identity stable across episodes

> **This note is the ORIGINAL PROPOSAL, kept for its reasoning.** It was
> picked up and built. For what actually exists, read
> [`docs/speaker-identity-decisions.md`](speaker-identity-decisions.md)
> (what was decided, and which claims turned out wrong) and
> [the identity lifecycle design](superpowers/specs/2026-09-05-identity-lifecycle-design.md)
> (how a name is cleared and a missing one healed). Two things the
> proposal below did not anticipate, both of which cost real work:
>
> - **Identity has a lifecycle, not just a moment.** Naming at ingest is
>   the easy half. Records outlive the run that named them, Chroma merges
>   metadata onto positional ids, and a name can therefore go stale or go
>   missing with nothing detecting either. `identity_version` exists
>   because of that, and it is the piece the proposal has no equivalent of.
> - **"Unknown as the fallback" is not one state.** A cluster nobody
>   enrolled matches (27% of them, legitimately) and a cluster nothing ever
>   tried to match look identical in `speaker_id`, and only one of them
>   needs repairing. Conflating them is a queue that never terminates.

**Status: SUPERSEDED and PARTLY WRONG. Written 2026-08-31 as a design note;
built differently and shipped 2026-09-04.**

**Read
[`superpowers/specs/2026-09-02-speaker-grouping-design.md`](superpowers/specs/2026-09-02-speaker-grouping-design.md)
instead.** This file is kept because its statement of the *problem* (below)
is still the clearest one, and because the reasoning that led away from it is
worth seeing. Its **plan** is not: three specific passages, marked inline,
would lead a contributor to build the exact defect the shipped design exists
to prevent. Each is struck where it appears rather than only here, because a
reader arriving from a search lands mid-document and never sees this banner.

What actually shipped: five voices enrolled from 208 voice-verified clips,
matched against every cluster in the archive, written to Chroma as a NEW
`speaker_id` key by a metadata-only `collection.update()` -- 22,483 of 29,160
chunks, no GPU, no re-embed. Since 2026-09-04 the nightly ingest assigns
identity itself, so no further backfill is needed.

## The problem

Diarisation is not identification. WhisperX gives us `SPEAKER_00`, `SPEAKER_01`
and so on, and those labels are **only meaningful inside a single audio file**.
`SPEAKER_00` in Monday's episode and `SPEAKER_00` in Tuesday's are unrelated.
The clustering is per-file, so the numbering falls out of whoever happens to talk
first.

That is why the current corpus can tell you *that* two people disagreed in one
episode, but cannot tell you what any named person thinks across the archive.

## Why this is worth doing

The interesting query is not "find me chunks about Taiwan". It is "what has
Jacob Shapiro said about Taiwan, and has his view moved?" That means modelling a
point of view per analyst, which is exactly what a briefing consumer wants:
attribution, and a way to notice when a recurring voice changes position.

`latest_on_topic` and the date-sorted results already provide the "has it moved"
half. Stable identity is the missing half.

## Why it is tractable here specifically

Three things make this much easier than the general speaker-ID problem:

1. **The shows have stable hosts.** Three feeds, each with one or two recurring
   voices. The enrolment set is a handful of people, not an open population.
2. **Hosts dominate speech time**, so getting only the hosts right captures most
   of the value. Guest identification is the genuinely hard part and can be
   skipped indefinitely.
3. **The metadata field already exists.** `speaker` is already written into every
   Chroma record and already flows through `search_podcasts` results into
   consumers. This is a change of *value*, not of schema.

## The approach

Speaker **identification** on top of the existing diarisation:

1. **Embed voices, not just words.** ~~Run a speaker-embedding model
   (`pyannote/embedding`, or `speechbrain/spkrec-ecapa-voxceleb`)~~ over each
   diarised segment to get a fixed-size voice vector.

   > **WRONG — both models were measured unusable on 2026-09-02.**
   > `pyannote/embedding` returns a 403 `GatedRepoError` under the token this
   > project holds, and `speechbrain/spkrec-ecapa-voxceleb` is absent from the
   > image entirely. What is used instead is the embedder that ships INSIDE
   > `pyannote/speaker-diarization-community-1`, in its `embedding`
   > subfolder -- the same checkpoint that produced the diarisation clusters,
   > so no second model is loaded. See `corpus/voice.py`.
2. **Enrol once, by hand.** For each known host, pick a few clean segments and
   label them. Store the centroid vector per person. This is a one-off of maybe
   an hour, and it is the only manual step.
3. **Match per episode.** For each diarised cluster in a new episode, compute its
   centroid and cosine-match it against the enrolled centroids. Above threshold,
   assign the name. Below threshold, leave it as `guest_1`, `guest_2`.
4. ~~**Write the name into the existing `speaker` metadata field.** Everything
   downstream picks it up for free.~~

   > **DO NOT DO THIS. It is the specific defect the shipped design exists to
   > prevent, and it fails in a way your tests will not catch.**
   > `corpus/chunking.py` bakes the cluster label into the document text as a
   > `[SPEAKER_01] ` prefix, and `corpus/rendering.py` strips that prefix
   > using the value of the `speaker` metadata field. Overwrite `speaker`
   > with a person's name and the strip no longer matches, so `[SPEAKER_01]`
   > leaks back into the very text the name was meant to replace -- in the
   > rendered output only. Unit tests over the metadata pass; the product is
   > visibly corrupt.
   >
   > Identity is written to a NEW `speaker_id` key instead, leaving `speaker`
   > untouched. See spec section 2.2 and `backfill_speaker_id.py`.

Optionally, recurring unknowns can be auto-enrolled: if the same unmatched voice
appears across several episodes, it is probably a regular and worth a name.

## ⚠️ The gotcha that decides the cost

**The speaker label is baked into the embedded text, not just the metadata.**
`chunk_by_speaker` builds each chunk as:

```python
"text": f"[{speaker}] {text}"
```

So `SPEAKER_00` is inside the string that gets embedded by BGE and stored as the
document. Two consequences:

- ~~**Renaming retroactively is a re-embed, not a metadata update.**
  Backfilling names across the existing corpus (~27k records at the time of
  writing) means re-running the embedding step, which is GPU time. It is not a
  cheap `collection.update()` on metadata.~~

  > **FALSIFIED by the 2026-09-04 backfill, which was exactly the cheap
  > metadata-only `collection.update()` this paragraph says is impossible:**
  > 22,483 of 29,160 chunks, no GPU, no re-embed, documents and embeddings
  > preserved (verified by reading them back). The claim only held under the
  > plan above, where the name overwrites `speaker` and therefore changes the
  > document text. Writing a separate `speaker_id` key does not touch the
  > document, so nothing needs re-embedding. Measured on chromadb 1.5.9:
  > `update()` preserves document and embedding and MERGES metadata, while a
  > metadata-only `upsert()` is rejected client-side.
- **Going forward, a real name in that string is an improvement, not just
  cosmetic.** `[Jacob Shapiro] ...` gives the embedding a meaningful token that a
  query like "what does Jacob Shapiro think" can partially match on, where
  `[SPEAKER_00]` is noise occupying the same space.

So the sensible sequencing is: build identification, apply it to **new**
episodes first, and treat the historical backfill as a separate, costed decision
rather than as part of the same change.

## Fail closed, and mean it

Misattributing a claim to a named real person is worse than declining to
attribute it. This corpus feeds a briefing that produces position signals, so a
confident wrong name propagates into analysis that someone acts on.

**Below the confidence threshold, the answer is `unknown`, never the closest
guess.** Precision matters far more than recall here, and the threshold should be
tuned that way deliberately rather than optimised for accuracy overall.

## The part that is worth more than the feature

Tuning that threshold requires a small labelled set: some episodes where the
speakers are known, a decision about what counts as a correct assignment, and a
precision bar the matcher has to clear before it ships.

That is an **evaluation harness**, and it is a better artefact than the feature
it gates. It is also cheap here, because ground truth is obtainable by listening
to a few minutes of a handful of episodes.

## Suggested scope if this gets picked up

- **In:** the recurring hosts of the three feeds. Enrolment, cosine matching, a
  threshold, `unknown` as the fallback, names on new episodes only.
- **Out, at least initially:** guest identification, retroactive backfill,
  auto-enrolment of unknowns, and any attempt at cross-show identity resolution.
- **Deliverable that makes it real:** the eval set and the measured precision,
  written down. Without that number this is a demo.
