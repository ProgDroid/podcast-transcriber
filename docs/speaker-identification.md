# Design note: naming speakers, and keeping identity stable across episodes

**Status: not built. Design note only, written 2026-08-31.**

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

1. **Embed voices, not just words.** Run a speaker-embedding model
   (`pyannote/embedding`, or `speechbrain/spkrec-ecapa-voxceleb`) over each
   diarised segment to get a fixed-size voice vector.
2. **Enrol once, by hand.** For each known host, pick a few clean segments and
   label them. Store the centroid vector per person. This is a one-off of maybe
   an hour, and it is the only manual step.
3. **Match per episode.** For each diarised cluster in a new episode, compute its
   centroid and cosine-match it against the enrolled centroids. Above threshold,
   assign the name. Below threshold, leave it as `guest_1`, `guest_2`.
4. **Write the name into the existing `speaker` metadata field.** Everything
   downstream picks it up for free.

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

- **Renaming retroactively is a re-embed, not a metadata update.** Backfilling
  names across the existing corpus (~27k records at the time of writing) means
  re-running the embedding step, which is GPU time. It is not a cheap
  `collection.update()` on metadata.
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
