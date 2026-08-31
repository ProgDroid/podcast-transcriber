# Design: corpus integrity — episode identity, safe writes, and completeness

**Status: approved after two rounds of adversarial review, not yet implemented.
Written 2026-08-31.**

Ships **before** [speaker identification](2026-08-31-speaker-identification-design.md),
which is a hard dependency rather than an ordering preference — see §3.

## Part 1 — The defect

Seven of the 438 transcripts on the `podcast-transcripts` Modal volume have no
chunks in Chroma. They transcribed successfully, they sit on the volume, and
they are invisible to `search_podcasts`.

Measured against `geo-podcasts-us` (28,489 records) on 2026-08-31:

| Show | Episode | Date | Chunks | Cause |
|---|---|---|---|---|
| Geopolitical Cousins | 73 | 2026-07-29 | **431** | upsert cap |
| Geopolitical Cousins | 74 | 2026-07-31 | **321** | upsert cap |
| The Jacob Shapiro Podcast | Unknown | 2026-07-29 | **431** | upsert cap + collision |
| The Jacob Shapiro Podcast | Unknown | 2026-07-31 | **321** | upsert cap + collision |
| The Jacob Shapiro Podcast | 243 | 2024-11-08 | 71 | collision |
| The Observing Japan Podcast | Unknown | 2026-05-12 | 1 | collision |
| The Observing Japan Podcast | Unknown | 2026-06-05 | 60 | collision |

### Root cause 1 — identity

`episode_number` is used as an episode's identity, but it is a display
attribute. It is **absent** — no `itunes_episode`, no `Ep. N` in the title, so
`parse_all_episodes` substitutes the literal `"Unknown"`, 5 times — and it is
**duplicated**: `Episode 243` exists twice, 2024-11-07 and 2024-11-08.

1. `parse_all_episodes` substitutes `"Unknown"` for a missing number.
2. `embed_and_store` builds IDs as `{show}-ep{episode_number}-{i}`. Three groups
   collide.
3. `bulk_embed` checks presence with `where {show, episode_number}, limit 1`,
   which matches the first member of a colliding group and **skips the rest**.
   This is why the damage is omission, not overwrite — probe-confirmed: zero ID
   prefixes are currently shared by two episodes.
4. `transcribe` decides what to skip by testing whether the transcript file
   exists, which proves *transcription* ran and says nothing about whether
   *embedding* landed.

**The key that works.** `(show, episode_number)` has 3 duplicate groups;
`(show, date)` has 4 duplicate pairs (Geopolitical Cousins 2026-05-22; Jacob
Shapiro 2023-11-20, 2025-03-28, 2025-06-13); `(show, episode_number, date)` has
**438 distinct keys across 438 files**. Keying on date alone would have fixed 7
collisions and created 4 new ones.

### Root cause 2 — the 300-record upsert cap

Chroma Cloud rejects an upsert of more than 300 records **per request**, and
`embed_and_store` sends an episode's chunks in a single call. The raised error
lands in `except Exception: continue` (`transcribe.py:356`) and prints one line.

This is not a latent hazard. Two distinct pieces of audio already exceed the
cap — 431 and 321 chunks — and because both were cross-posted to a second feed,
they account for **four of the seven missing episodes**. Chunk counts are
trending up: 257 (2026-03), 259 (2026-06), 431 (2026-07), all Geopolitical
Cousins.

An earlier draft of this spec put the maximum at 257 chunks with "43 chunks of
headroom". That figure came from the stale 400-file local mirror, which excludes
precisely the episodes that failed. The measurement was accurate about the wrong
population.

## Part 2 — Verified Chroma Cloud semantics

Measured against a throwaway Cloud collection in `geo-podcasts-us`, then
deleted. `podcast_transcripts` was never touched.

| Behaviour | Result |
|---|---|
| `delete(where=<triple>)` | **Supported** — removed exactly the matching records, 3 → 1 |
| `delete(where=)` matching nothing | No-op. No "empty filter wipes the collection" trap |
| `upsert` on an existing id | Document replaced wholesale; **metadata MERGED, not replaced** |
| `get(limit=301)` or higher | Raises `ChromaError` |
| `get()` with no explicit limit | **Silently returns 300** of 320 |
| `upsert` of >300 records | Raises `ChromaError` — 300 per request |

Metadata merging means a key absent from the new metadata survives from the old
record. Invisible today because `build_chunks` writes a fixed key set, so merge
and replace coincide — and it stops being invisible the moment the speaker work
changes that key set. Full replacement removes the hazard rather than
documenting it.

## Part 3 — Why this must precede the speaker work

`build_chunks` cuts a chunk boundary at every speaker change. When `SPEAKER_00`
and `SPEAKER_01` both resolve to `"Jacob Shapiro"`, that boundary disappears,
adjacent chunks merge, and the episode's chunk count strictly **drops**. `upsert`
overwrites only `0..n-1`, so every index above the new count survives — still
carrying `[SPEAKER_01]` in its document text and duplicating passages that now
live inside the merged chunk. The speaker feature would corrupt the corpus it is
improving.

## Part 4 — The design

### 1. Identity: guid first, triple as fallback

```python
def episode_id_prefix(show, episode_number, date_str) -> str:
    return f"{show}-ep{episode_number}-{date_str}".replace(" ", "_")
```

**`episode_guid` is stored from day one**, taken from feedparser's `entry.id`.
The triple is derived from three mutable RSS fields, and the most likely feed
change in the next six months breaks it silently: if Captivate backfills
`itunes_episode` for the five `"Unknown"` episodes, the triple changes, the
filename changes, `transcript_exists` goes False, a T4 re-transcribes audio
already on the volume, and the old records survive as permanent reconciliation
extras. `parse_all_episodes` compounds this with a naive `strftime`, so a
pubDate timezone change shifts the date by a day.

The guid is the prune key when present, the triple otherwise. It is **optional**
by necessity: 355 Jacob Shapiro transcripts are on the volume against 350 feed
entries, so at least 5 episodes can never be assigned one. Reconciliation
reports guid coverage. Adding this later would be a second corpus rewrite;
adding it during a migration already running is nearly free.

### 2. Writes: upsert first, then prune

```
new_ids = [f"{episode_id_prefix(...)}-{i}" for i in range(len(chunks))]
upsert(new_ids, ...)                       # batched at 250
stale = paged_get(where=<guid or triple>).ids - set(new_ids)
delete(ids=stale)
```

**Not delete-then-upsert.** A new episode's delete matches nothing, so the
trickle case has no window at all — but the very next phase re-embeds all 438
*healthy* episodes, which under delete-first is 438 destructive windows opened
on records that were fine. And the failure that matters is not an exception:
`transcribe` runs `timeout=7200` for a whole show's pending list, and a Modal
timeout kills the container without raising anything the `except` clause can
see. Under delete-first that is a deleted-and-never-rewritten episode with no
log line. Under upsert-then-prune a crash leaves a **superset**, never a hole,
and the next run prunes it. Same round-trip count.

Batching at 250 is load-bearing — see root cause 2.

### 3. Presence is completeness, not existence

Write `n_chunks` into every chunk's metadata. An episode is present only when
its stored chunk count equals `n_chunks`.

A boolean check fails silently: after a collision clobber the older episode
retains one orphan chunk, so `episode_in_chroma(...)` returns `True`, the
self-healing branch calls a two-thirds-destroyed episode healthy, and
reconciliation passes.

**The count query must page at 250.** Chroma has no `count(where=...)`, and an
unlimited `get()` silently returns 300. Unpaged, the 431- and 321-chunk episodes
would report short, be judged incomplete, and be **re-embedded every night
forever**. Tested at the 250/300/301 boundary and above it.

`n_chunks` is absent on all 28,489 existing records. The migration writes it
during the copy; `decide_action` treats a missing value as **incomplete**, never
as satisfied — the silent-default reading would mean old episodes are never
checked at all.

### 4. `decide_action` — four states, exclusion first-class

```
decide_action(transcript_exists, complete_in_chroma, excluded, parses_to_chunks)
    excluded                     -> EXCLUDE      (terminal)
    not parses_to_chunks         -> UNPARSEABLE  (terminal, reported)
    not transcript_exists        -> TRANSCRIBE
    transcript and not complete  -> EMBED_ONLY
    otherwise                    -> SKIP
```

`EXCLUDE` exists because the cross-post decision in Part 5 is otherwise
unexpressible: both excluded episodes have transcripts on the volume, so a
three-argument `decide_action` returns `EMBED_ONLY` and the 09:00 cron re-embeds
them the next morning — and the approved deletion of the 2025 duplicates reverts
on the same schedule, indefinitely.

`UNPARSEABLE` is defensive. Measured across all 438 volume files: **zero** parse
to zero chunks, none is empty, the minimum is 1 chunk. Nothing triggers it
today, but a state machine that is total is not the same as one no current input
breaks.

All five branches are pure and unit-tested without Modal, GPU or Chroma.

### 5. Plan on CPU, execute on GPU

A CPU-only function computes the work list from feed, volume and Chroma; the GPU
function starts only if it is non-empty. Otherwise the completeness check turns
~438 local `os.path.exists` calls into ~440 Chroma round trips inside a
`gpu="T4"` container billed for the wait. It also fixes something predating this
work: the cron starts a T4 and loads BGE every day just to find nothing to do.

**Whether `EMBED_ONLY` can also run CPU-only is left open pending measurement.**
An earlier claim that `upload_book.py` proves BGE runs CPU-side does not hold —
that file has no `@app.function` decorator, it is a local script, and
`SentenceTransformer` with no `device=` auto-selects CUDA. The repair is ~450
chunks; the phase-B1 archive re-embed is ~29,894. Measure BGE-large CPU
throughput before splitting the tier.

### 6. Bidirectional reconciliation

Assert all of: no volume episode missing from Chroma; no Chroma triple absent
from the volume; chunk indices contiguous from 0 per episode; no ID prefix
shared by two episodes; excluded triples hold **zero** records. A
one-directional check keeps passing while the corpus rots.

### 7. ID migration: copy, validate, swap

Retained by decision. With upsert-then-prune it is not strictly required —
old- and new-scheme IDs provably cannot collide, since the old form ends in an
integer and the new in `YYYY-MM-DD-{int}`, verified by set intersection — but it
is the natural place to write `n_chunks` and `episode_guid`, which the design
needs on every record.

1. **Freeze the cron first.** `copy_collection` raises
   `SOURCE CHANGED during copy` if the count moves, so the 09:00 job must be
   stopped across copy *and* validate.
2. Create `podcast_transcripts_v2` from the source's serialized schema
   (`create_dest_collection` — pass **schema only**).
3. Page the source at 250; compute new IDs; add `n_chunks` and `episode_guid`;
   upsert into v2 **in batches of 250**.
4. Validate v2 against v1 through the ID map: counts, then per-record documents,
   metadata, uris, embeddings (`allclose`, `atol=1e-4`).
5. Cut over via the collection-name constant; redeploy; confirm search.
6. Delete v1 only after explicit confirmation. Unlike the July EU→US migration,
   a rollback path exists until then.

**Records not matching the podcast ID pattern are copied verbatim.**
`upload_book.py` writes `Geopolitical_Alpha-p{n}` for the *Geopolitical Alpha*
book (191 records). Only records whose ID matches
`^(?P<show>.+)-ep(?P<ep>[^-]+)-(?P<i>\d+)$` **and** whose metadata triple
reconstructs that prefix are re-IDed; the rest pass through, and the count of
each class is reported.

### 8. Collection access

Read the name from `CHROMA_COLLECTION`, defaulting to the current value. **Use
`get_collection`, not `get_or_create_collection`, in `mcp_server.py`** — all
three call sites currently use get-or-create, so a typo at cutover would
silently create an empty third collection and make `search_podcasts` return
"No results found." rather than erroring. The reader must fail loudly; the
writers may still create.

Note that `mcp_server` reads this in `@modal.enter()`, so a warm `PodcastSearch`
container keeps the old collection until it cycles. Cutover must force a new
container.

## Part 5 — Cross-posted episodes

Four feed entries are two Geopolitical Cousins episodes republished on the Jacob
Shapiro feed. Two transcripts were diffed: byte-identical, 1694 lines each,
identical timestamps, differing only in the header show name. Each was
downloaded and transcribed twice on a T4 — and at 431 chunks each, both copies
then failed the upsert cap.

| Title | Date | State |
|---|---|---|
| Riding on the Hog of a Fiscal Orgy | 2025-04-04 | already duplicated in Chroma |
| Let Them Drink Bleach | 2025-04-08 | already duplicated in Chroma |
| This Is The Way The World Ends | 2026-07-29 | among the missing 7 |
| Lessons Learned | 2026-07-31 | among the missing 7 |

**Embed under Geopolitical Cousins only.** The repair covers 5 of 7; the two
Jacob Shapiro copies go on the committed exclusion list. The two 2025 duplicates
already in Chroma are removed as a **separate, explicitly approved step**, and
only after `EXCLUDE` exists — otherwise the cron restores them the next morning.

**No automatic cross-post detection.** Enclosure URLs differ across feeds (zero
shared — Captivate re-hosts), so the only signal is a fuzzy title-and-date match.
Code that silently discards an episode because a regex thought two titles matched
is a failure mode with no alarm on it, defending against 0.9% duplication.
Reconciliation *reports* suspected cross-posts; exclusion is a human decision in
a checked-in list.

## Testing

Unit, CPU-only, no network:

- `episode_id_prefix` — the three colliding groups and the four duplicate
  `(show, date)` pairs all yield distinct prefixes.
- ID re-mapping — old ID plus metadata to new ID; book `-p{n}` IDs pass through;
  an ID whose metadata does not reconstruct its prefix passes through and is
  counted, never guessed.
- `decide_action` — every combination across all five states; `excluded` wins
  over `EMBED_ONLY`; missing `n_chunks` reads as incomplete.
- Completeness — count equals `n_chunks` is complete; short is incomplete;
  paging assembles the right total across the 250 boundary; a 431-chunk episode
  is judged complete rather than looping.
- Batching — 431 chunks issue two requests, neither over 250.
- Upsert-then-prune — a shrinking re-embed leaves no orphan; a crash simulated
  between upsert and prune leaves a superset, and a second run converges.
- `build_chunks_from_text` — round-trips the written format, `[UNKNOWN]` and
  malformed lines included.

Integration, run once and recorded: full bidirectional reconciliation after
cutover.

## Acceptance

- Reconciliation reports zero missing, zero extra, contiguous indices, no shared
  prefix, and zero records for each excluded triple — with exclusions listed by
  name rather than silently absent.
- The 431- and 321-chunk episodes are stored complete.
- `podcast_transcripts_v2` validates against v1 record-for-record, book records
  unchanged, with `n_chunks` and `episode_guid` present.
- A simulated shrinking re-embed leaves no orphans; a simulated mid-write crash
  leaves a superset that the next run prunes.
- Both apps deploy against the new collection and `search_podcasts` returns.
- Unit suite passes in CI with no GPU and no secrets.

## Out of scope

- Speaker identification (its own spec).
- Deleting `podcast_transcripts` v1.
- Recovering an episode number for the five `"Unknown"` episodes — the feed does
  not publish one, and inventing one is the category error this spec exists to
  fix.

## Housekeeping

The local `CHROMA_API_KEY` and `CHROMA_DATABASE` point at `geo-podcasts`, the EU
database deleted 2026-07-09, and the key is scoped to it — every probe behind
this spec ran through Modal's `podcast-secrets`. Worth refreshing before the
speaker spec's eval tooling needs local Chroma reads.
