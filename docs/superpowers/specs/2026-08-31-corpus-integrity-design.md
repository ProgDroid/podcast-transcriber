# Design: corpus integrity — episode identity, full-replacement writes, and completeness

**Status: approved after adversarial review, not yet implemented. Written 2026-08-31.**

Ships **before** [speaker identification](2026-08-31-speaker-identification-design.md).
That ordering is not tidiness: section 3 shows the speaker feature actively
corrupting the corpus if it lands on today's write path.

## Part 1 — The defect

Seven of the 438 transcripts on the `podcast-transcripts` Modal volume have no
chunks in Chroma. They transcribed successfully, they sit on the volume, and
they are invisible to `search_podcasts`.

Measured against `geo-podcasts-us` (28,489 records) on 2026-08-31:

| Show | Episode | Date | Cause |
|---|---|---|---|
| Geopolitical Cousins | 73 | 2026-07-29 | link 4 |
| Geopolitical Cousins | 74 | 2026-07-31 | link 4 |
| The Jacob Shapiro Podcast | 243 | 2024-11-08 | links 2+3 |
| The Jacob Shapiro Podcast | Unknown | 2026-07-29 | links 1+2+3 |
| The Jacob Shapiro Podcast | Unknown | 2026-07-31 | links 1+2+3 |
| The Observing Japan Podcast | Unknown | 2026-05-12 | links 1+2+3 |
| The Observing Japan Podcast | Unknown | 2026-06-05 | links 1+2+3 |

### Root cause

`episode_number` is used as an episode's identity, but it is a display
attribute. It is **absent** — the feed supplies no `itunes_episode` and the
title has no `Ep. N`, so `parse_all_episodes` substitutes the literal string
`"Unknown"`, 5 times — and it is **duplicated**: `Episode 243` exists twice, on
2024-11-07 and 2024-11-08.

### The chain

1. **`parse_all_episodes`** substitutes `"Unknown"` for a missing number.
2. **`embed_and_store`** builds IDs as `{show}-ep{episode_number}-{i}`. Three
   groups collide: `The_Jacob_Shapiro_Podcast-ep243` (2),
   `The_Jacob_Shapiro_Podcast-epUnknown` (2),
   `The_Observing_Japan_Podcast-epUnknown` (3).
3. **`bulk_embed`** checks presence with `where {show, episode_number}, limit 1`.
   For a colliding group this matches the first member and **skips every other
   member**. This is why the damage is omission rather than overwrite —
   confirmed by probe: zero ID prefixes are currently shared by more than one
   episode.
4. **`transcribe`** decides what to skip by testing whether the transcript file
   exists. That proves *transcription* ran and says nothing about whether
   *embedding* landed, so an episode whose embed failed is skipped on every
   subsequent run, forever. This is the only explanation for the three missing
   episodes unrelated to `"Unknown"`.

### The key that actually works

- `(show, episode_number)` — 3 duplicate groups.
- `(show, date)` — 4 duplicate pairs: Geopolitical Cousins 2026-05-22; The
  Jacob Shapiro Podcast 2023-11-20, 2025-03-28, 2025-06-13.
- `(show, episode_number, date)` — **438 distinct keys across 438 files.**

Keying on the date alone would have fixed 7 collisions and created 4 new ones.

## Part 2 — Verified Chroma Cloud semantics

Everything below was measured against a throwaway Cloud collection in
`geo-podcasts-us`, then deleted. `podcast_transcripts` was never touched. These
are behaviours, not documentation claims, and the design leans on all of them.

| Behaviour | Result |
|---|---|
| `delete(where=<triple>)` | **Supported.** Removed exactly the matching records, 3 → 1 |
| `delete(where=)` matching nothing | Deletes nothing. No "empty filter wipes the collection" trap |
| `upsert` on an existing id | Document replaced wholesale; **metadata is MERGED, not replaced** |
| `get(limit=301)` or higher | Raises `ChromaError` — loud |
| `get()` with no explicit limit | **Silently returns 300** of 320 records |
| `upsert` of >300 records in one call | Raises `ChromaError` — **300 per request** |

Two of these are load-bearing and neither was in the original spec:

**Metadata merges.** A key absent from the new metadata survives from the old
record — upsert "updates existing items as per the `update` method", and
`update` is partial. This is invisible today because `build_chunks` always
writes the same fixed key set, so merge and replace coincide. It stops being
invisible the moment the speaker work changes that key set, at which point a
record can hold a key combination no single write ever produced. Full
replacement via delete-then-upsert removes the hazard rather than documenting it.

**Upsert caps at 300 per request.** `embed_and_store` currently upserts an
episode's chunks in one call. The largest episode is 257 chunks (Geopolitical
Cousins 47), so the pipeline runs at 86% of a hard cap it does not know exists.
A single long episode would raise, land in `transcribe`'s
`except Exception: continue`, print one line, and be skipped forever by link 4.

## Part 3 — Why this must precede the speaker work

`build_chunks` cuts a chunk boundary at every speaker change:

```python
if speaker != current_speaker and current_words:
    flush_chunk(...)
```

When `SPEAKER_00` and `SPEAKER_01` both resolve to `"Jacob Shapiro"`, that
boundary disappears, adjacent chunks merge, and the episode's chunk count
strictly **drops**. `upsert` can only overwrite indices `0..n-1`, so every index
above the new count survives — still carrying `[SPEAKER_01]` baked into its
document text by `build_chunks`, and duplicating passages that now also live
inside the merged chunk. `search_podcasts` returns them with no way for a caller
to tell.

The speaker feature would therefore corrupt the corpus it is improving. Only
full-replacement writes prevent it.

## Part 4 — The design

### 1. One identity helper

```python
def episode_id_prefix(show: str, episode_number: str, date_str: str) -> str:
    """Stable, unique per-episode ID prefix.

    Keyed on the full triple because neither episode_number nor date is unique
    alone: the feeds omit episode numbers and occasionally repeat them, and two
    episodes of one show can share a publication date.
    """
    return f"{show}-ep{episode_number}-{date_str}".replace(" ", "_")
```

`episode_number` keeps its current metadata value, `"Unknown"` included. We are
fixing identity, not inventing a number the feed never published.

### 2. Full-replacement writes

`embed_and_store` becomes: `delete(where=<triple>)`, then upsert **batched at
250**. This closes three things at once — the orphan tail from a shrinking
re-embed, the metadata-merge hazard, and the 300-per-request upsert cap.

Not atomic: a crash between delete and upsert leaves an episode with zero
chunks. That is exactly the state the self-healing branch repairs, so the
failure mode degrades into the convergence property rather than into corruption.

### 3. Presence is a completeness check, not an existence check

Write `n_chunks` into every chunk's metadata. An episode counts as present only
when its stored chunk count equals `n_chunks`; any mismatch triggers a full
re-embed.

A boolean check is broken, and the way it breaks is silent. After a collision
clobber, the older episode retains one orphan chunk — so
`episode_in_chroma(show, "243", "2024-11-07")` returns `True`, the self-healing
branch classifies a two-thirds-destroyed episode as healthy, and the
reconciliation passes. Green lights on the exact failure the check exists to
catch.

**The count query must page at 250.** Chroma has no `count(where=...)`, so the
count comes from `len(get(where=...))`, and an unlimited `get()` silently
returns 300. Unpaged, Geopolitical Cousins 47 (257 chunks) would eventually
report short, be judged incomplete, and be **re-embedded on every cron run
forever** — a permanent GPU bill produced by a check meant to prevent waste.
Tested at the 250/300/301 boundary.

### 4. `transcribe` becomes self-healing

| Transcript on volume | Episode complete in Chroma | Action |
|---|---|---|
| no | — | full pipeline |
| yes | no | **embed from the saved transcript** — no GPU transcription |
| yes | yes | skip |

`build_chunks_from_text` already parses the format the pipeline writes, so the
middle branch is nearly free. All seven broken episodes are reachable from
today's feeds — verified, 7/7 — so the existing 09:00 UTC cron repairs them
unattended, and future embed failures self-correct.

Expressed as a pure `decide_action(transcript_exists, complete_in_chroma)`
returning `TRANSCRIBE | EMBED_ONLY | SKIP`, so the branching is testable without
Modal, GPU or Chroma.

### 5. Split planning from execution

A CPU-only function computes the work list from feed, volume and Chroma and
returns it; the GPU function starts only if that list is non-empty.

Without this, the completeness check turns ~438 local `os.path.exists` calls
into ~440 Chroma round trips **inside a `gpu="T4"` container billed for the
whole wait**, growing with the catalogue. It also fixes something that predates
this work: the cron currently starts a T4 and loads BGE every single day just to
discover there is nothing to do.

### 6. Bidirectional reconciliation

The original acceptance criteria could only see missing records. They must also
assert:

- every distinct triple in Chroma maps to a file on the volume (no extras),
- every episode's chunk indices are contiguous from 0 (no orphan tails),
- no ID prefix is shared by two episodes.

A one-directional check keeps passing while the corpus rots.

### 7. ID migration: copy, validate, swap

Retained at the user's decision. With full-replacement writes it is no longer
*required* — old- and new-scheme IDs provably cannot collide, since the old form
ends in an integer and the new in `YYYY-MM-DD-{int}`, verified by set
intersection — but a uniform corpus is worth the one-time cost and removes a
mixed-scheme state that would otherwise persist indefinitely.

1. Create `podcast_transcripts_v2` from the source collection's serialized
   schema (`create_dest_collection` — pass **schema only**, never `schema=` and
   `metadata=` together).
2. Page the source at 250, including embeddings, documents, metadatas, uris.
3. Compute new IDs; upsert into v2 **in batches of 250**.
4. Validate v2 against v1 through the ID map: counts, then per-record documents,
   metadata, uris and embeddings (`allclose`, `atol=1e-4`).
5. Cut over via the collection-name constant; redeploy both apps; confirm search.
6. Delete v1 only after the user confirms — a separate, explicit step. Unlike
   the July EU→US migration, a rollback path exists until then.

**Records not matching the podcast ID pattern are copied verbatim.**
`upload_book.py` writes `Geopolitical_Alpha-p{n}` for the *Geopolitical Alpha*
book (191 records, `episode_number="N/A"`, `date="2021-01-01"`). Those IDs are
already unique and carry no episode concept. Only records whose ID matches
`^(?P<show>.+)-ep(?P<ep>[^-]+)-(?P<i>\d+)$` **and** whose metadata triple
reconstructs that prefix are re-IDed; everything else passes through unchanged,
and the count of each class is reported.

### 8. Collection name becomes a constant

`"podcast_transcripts"` is hardcoded in `transcribe.py`, `mcp_server.py` and
`upload_book.py`. Read it from `CHROMA_COLLECTION` with the current name as the
default, so cutover is a secret change rather than three code edits.

## Part 5 — Cross-posted episodes

Four episodes are Geopolitical Cousins content republished on the Jacob Shapiro
feed. Two transcripts were diffed directly: byte-identical content, 1694 lines
each, identical timestamps, differing only in the header show name. Each was
downloaded and transcribed twice on a T4.

| Title | Date | State |
|---|---|---|
| Riding on the Hog of a Fiscal Orgy | 2025-04-04 | **already duplicated in Chroma** |
| Let Them Drink Bleach | 2025-04-08 | **already duplicated in Chroma** |
| This Is The Way The World Ends | 2026-07-29 | among the missing 7 |
| Lessons Learned | 2026-07-31 | among the missing 7 |

**Decision: embed under Geopolitical Cousins only.** The repair covers 5 of the
7; the two Jacob Shapiro copies are listed as deliberate exclusions. The two
2025 duplicates already in Chroma are removed as a **separate, explicitly
approved step** — it deletes production data, so it does not ride along with a
bugfix.

**No automatic cross-post detection in the pipeline.** Enclosure URLs differ
across feeds (zero shared — Captivate re-hosts), so the only available signal is
a fuzzy title-and-date match. Code that silently discards an episode because a
regex thought two titles matched is a failure mode with no alarm on it,
defending against 0.9% duplication. The bidirectional reconciliation *reports*
suspected cross-posts; exclusions are an explicit list a human maintains.

## Testing

Unit, CPU-only, no network:

- `episode_id_prefix` — spaces replaced; stable across calls; the three known
  colliding groups and the four known duplicate `(show, date)` pairs all produce
  distinct prefixes.
- ID re-mapping — old ID plus metadata to new ID; book `-p{n}` IDs pass through
  untouched; an ID whose metadata does not reconstruct its prefix passes through
  and is counted, never guessed at.
- `decide_action(transcript_exists, complete_in_chroma)` — all four combinations.
- Completeness — count equal to `n_chunks` is complete; short is incomplete;
  paging assembles the right total across the 250 boundary; an episode of 257
  chunks is judged complete rather than looping.
- Batching — an upsert of 257 chunks issues two requests, neither over 250.
- `build_chunks_from_text` — round-trips the written format, including
  `[UNKNOWN]` and malformed lines skipped.

Integration, run once and recorded: re-run the volume↔Chroma reconciliation
after cutover.

## Acceptance

- Reconciliation reports zero missing, zero extra, contiguous indices for every
  episode, and no shared ID prefix — with the two cross-post exclusions listed
  by name rather than silently absent.
- `podcast_transcripts_v2` validates against v1 record-for-record through the ID
  map, book records included and unchanged.
- A simulated shrinking re-embed leaves no orphan records.
- Both Modal apps deploy against the new collection and `search_podcasts` returns.
- The unit suite passes in CI with no GPU and no secrets.

## Out of scope

- Speaker identification (its own spec).
- Deleting `podcast_transcripts` v1 — a follow-up once the new collection has
  been in use.
- Recovering an episode number for the five `"Unknown"` episodes. The feed does
  not publish one; inventing one is the category error this spec exists to fix.

## Housekeeping noted, not fixed here

The user's local `CHROMA_API_KEY` and `CHROMA_DATABASE` point at
`geo-podcasts`, the EU database deleted on 2026-07-09, and the key is scoped to
it — every probe behind this spec ran through Modal's `podcast-secrets` instead.
Worth refreshing before the speaker spec's eval tooling needs local Chroma reads.
