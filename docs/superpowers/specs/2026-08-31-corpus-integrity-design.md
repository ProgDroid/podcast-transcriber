# Design: corpus integrity — episode identity, self-healing embeds, and an ID migration

**Status: approved, not yet implemented. Written 2026-08-31.**

Ships **before** [speaker identification](2026-08-31-speaker-identification-design.md),
because that work re-upserts records and would otherwise inherit the ambiguity
described here at 28k scale.

## The defect

Seven of the 438 transcripts on the `podcast-transcripts` Modal volume have no
chunks in Chroma at all. They were transcribed successfully, they sit on the
volume, and they are invisible to `search_podcasts`.

Measured 2026-08-31 against `geo-podcasts-us` (28,489 records):

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
attribute. It is **absent** (the feed gives no `itunes_episode` and the title
has no `Ep. N`, so `parse_all_episodes` falls back to the literal string
`"Unknown"` — 5 times) and it is **duplicated** (`Episode 243` exists twice, on
2024-11-07 and 2024-11-08).

### The chain

1. **`parse_all_episodes`** substitutes `"Unknown"` for a missing number — a
   display value pressed into service as a key.
2. **`embed_and_store`** builds IDs as `{show}-ep{episode_number}-{i}`. Three
   groups collide: `The_Jacob_Shapiro_Podcast-ep243` (2),
   `The_Jacob_Shapiro_Podcast-epUnknown` (2),
   `The_Observing_Japan_Podcast-epUnknown` (3).
3. **`bulk_embed`** checks presence with
   `where {show, episode_number}, limit 1`. For a colliding group this matches
   the first member and **skips every other member**. This is why the damage is
   omission rather than overwrite — confirmed by probe: zero ID prefixes are
   shared by more than one episode, so no record was ever clobbered.
4. **`transcribe`** decides what to skip by testing whether the transcript file
   exists. That proves *transcription* ran; it says nothing about whether
   *embedding* landed. An episode whose embed step failed is skipped on every
   subsequent run, forever. This is the only explanation for the three missing
   episodes that have nothing to do with `"Unknown"`.

### The key that actually works

Neither part of the pair is unique on its own:

- `(show, episode_number)` — 3 duplicate groups (above).
- `(show, date)` — 4 duplicate pairs: Geopolitical Cousins 2026-05-22; The
  Jacob Shapiro Podcast 2023-11-20, 2025-03-28, 2025-06-13.
- `(show, episode_number, date)` — **438 distinct keys across 438 files.**

Keying on the date alone would have fixed 7 collisions and created 4 new ones.
The triple is the key.

## The design

### 1. One identity helper

```python
def episode_id_prefix(show: str, episode_number: str, date_str: str) -> str:
    """Stable, unique per-episode ID prefix.

    Keyed on the full triple because neither episode_number nor date is unique
    on its own: the feeds omit episode numbers and occasionally repeat them,
    and two episodes of one show can share a publication date.
    """
    return f"{show}-ep{episode_number}-{date_str}".replace(" ", "_")
```

Used by `embed_and_store` for ID generation and by the presence check below.
`episode_number` keeps its current value in metadata, `"Unknown"` included — we
are fixing identity, not inventing an episode number the feed never published.

### 2. One presence check

```python
def episode_in_chroma(collection, show, episode_number, date_str) -> bool:
    """Whether this episode's chunks are already stored, keyed on the full triple."""
```

Replaces the two-field query in `bulk_embed`, and is also the new third branch
in `transcribe`.

### 3. `transcribe` becomes self-healing

Three branches instead of two:

| Transcript on volume | Episode in Chroma | Action |
|---|---|---|
| no | — | full pipeline (download, transcribe, align, diarise, embed) |
| yes | no | **embed from the saved transcript** — no GPU transcription |
| yes | yes | skip |

The middle branch is nearly free: `build_chunks_from_text` already parses the
`[SPEAKER_XX] 12.3s - text` format the pipeline writes. It means the existing
09:00 UTC cron repairs all seven episodes on its next run, and any future embed
failure self-corrects with no manual step to remember.

This is the actual fix for link 4. It replaces a bug with a property: **the
pipeline converges on a complete corpus rather than assuming it produced one.**

### 4. ID migration: copy, validate, swap

The new scheme changes every podcast record's ID, so the 28,489 existing
records must move. No GPU is involved — Chroma returns stored embeddings
verbatim, so this is I/O only.

**Copy into a new collection rather than rewriting in place.** The July
EU→US migration left no rollback path; this one keeps one until the moment of
cutover.

1. Create `podcast_transcripts_v2` in `geo-podcasts-us` from the source
   collection's serialized schema (`create_dest_collection` — pass **schema
   only**, never `schema=` and `metadata=` together).
2. Page the source at `PAGE = 250` (Chroma Cloud caps `get()` at 300),
   including `embeddings`, `documents`, `metadatas`, `uris`.
3. For each record, compute the new ID and upsert into v2.
4. Validate v2 against v1 through the ID map: counts, then per-record
   documents, metadata, uris, and embeddings (`allclose`, `atol=1e-4`).
5. Cut over by changing the collection name constant, redeploy both apps,
   confirm search works.
6. Delete v1 only after the user confirms — a separate, explicit step.

**Records not matching the podcast ID pattern are copied verbatim.**
`upload_book.py` writes `Geopolitical_Alpha-p{n}` IDs for the *Geopolitical
Alpha* book (191 records, metadata `episode_number="N/A"`, `date="2021-01-01"`).
Those IDs are already unique and carry no episode concept; applying the podcast
scheme to them would corrupt them. The migration re-IDs only records whose ID
matches `^(?P<show>.+)-ep(?P<ep>[^-]+)-(?P<i>\d+)$` and whose metadata triple
reconstructs that prefix; everything else is copied unchanged, and the count of
each class is reported.

### 5. Collection name becomes a constant

`"podcast_transcripts"` is currently hardcoded in `transcribe.py`,
`mcp_server.py` and `upload_book.py`. Hoist it to one place per module, read
from `CHROMA_COLLECTION` with the current name as the default, so cutover is a
secret change rather than three code edits.

## Reuse

From `migration/chroma_migrate.py`: `PAGE = 250`, `INCLUDE`,
`drop_all_none` (Chroma rejects an all-`None` column — pass `None` for the
whole argument), `_embeddings_to_list`, `create_dest_collection`, and
`validate_collection` extended to take an optional `id_map`. The re-ID copy
itself is a new function; `copy_collection` preserves IDs by design and is left
alone.

## Testing

Unit, CPU-only, no network:

- `episode_id_prefix` — spaces replaced; the same triple is stable across
  calls; the three known colliding groups produce distinct prefixes; the four
  known duplicate `(show, date)` pairs produce distinct prefixes.
- ID re-mapping — old ID + metadata → new ID for the podcast pattern; book IDs
  (`-p{n}`) pass through untouched; an ID whose metadata does not reconstruct
  its prefix is passed through and counted, never guessed at.
- `transcribe` branch selection — a pure `decide_action(transcript_exists,
  in_chroma)` helper returning `TRANSCRIBE | EMBED_ONLY | SKIP`, so the
  three-way logic is testable without Modal, GPU or Chroma.
- `build_chunks_from_text` — round-trips the format `transcribe` writes,
  including the `[UNKNOWN]` speaker label and malformed lines being skipped.

Integration, run once against the real corpus and recorded:

- Re-run the volume↔Chroma reconciliation after cutover and assert
  `MISSING_FROM_CHROMA` is empty.

## Acceptance

- All 438 volume episodes are present in Chroma; the reconciliation reports
  zero missing.
- `podcast_transcripts_v2` validates against v1 record-for-record through the
  ID map, book records included and unchanged.
- No ID prefix is shared by two episodes.
- Both Modal apps are deployed against the new collection and `search_podcasts`
  returns results.
- The unit suite passes in CI with no GPU and no secrets.

## Out of scope

- Speaker identification (its own spec).
- Backfilling speaker names.
- Deleting `podcast_transcripts` v1 — proposed as a follow-up once the user has
  used the new collection for a while.
- Recovering an episode number for the five `"Unknown"` episodes. The feed
  does not publish one; inventing one would be the same category error this
  spec exists to fix.

## Housekeeping noted, not fixed here

The user's local `CHROMA_API_KEY` and `CHROMA_DATABASE` still point at
`geo-podcasts`, the EU database deleted on 2026-07-09, and the key is scoped to
it. Local Chroma access currently requires going through Modal's
`podcast-secrets`. Worth refreshing before the eval tooling in the speaker spec
needs local Chroma reads.
