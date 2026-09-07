# Design: the identity lifecycle — clearing a stale name, and healing a missing one

**Status: approved in brainstorming 2026-09-05, red-teamed the same day, not
yet implemented. Lands in two phases (Part 7).**

Closes the two items left open by
[`docs/speaker-identity-decisions.md`](../../speaker-identity-decisions.md)
§3 and §4. They are one design, not two, for the reason given in Part 2.

Depends on [corpus integrity](2026-08-31-corpus-integrity-design.md) — it
amends that design's **invariant 2**, and the amendment is only sound because
of that design's **invariant 3**.

---

## Part 1 — The two defects

### 1.1 A stale `speaker_id` survives a re-write

Confirmed, not suspected, by
`tests/test_write_path.py::test_a_stale_speaker_id_survives_a_re_transcribe`.

`corpus/writing.py` writes **positional** chunk ids and Chroma **merges**
metadata on upsert. `corpus/identify.py::stamp_chunks` can add a `speaker_id`
and can never remove one. So a chunk that carried a name in run *N* keeps it
in run *N+1* even when run *N+1* decided that position belongs to a different
cluster, or to nobody. `display_speaker` prefers `speaker_id`, and
`mcp_server.py:115` filters on it, so the reader gets a confident wrong name
and the filter returns a passage the named person did not say.

**The exposure is not confined to re-transcription.** The mechanism is
*positional ids plus metadata merge*, and it fires on any write that moves a
chunk boundary:

| Path | Boundaries move? | Stale stamp? | Same transcript? |
|---|---|---|---|
| Fresh transcription of an already-identified episode | yes — diarisation renumbers `SPEAKER_XX` | **yes** | **no** |
| Re-chunk under a bumped `RULES_VERSION` | yes — the constant exists to force this | **yes** | yes |
| Re-embed of a torn episode | positions **above** the old count are new | **yes**, and non-uniformly | yes |

Row 3 is the one a first draft of this spec got wrong, and it is worth stating
precisely: a torn episode is by definition missing records, so its repair is
the *growing* case. Old positions keep their merged `speaker_id` and
`identity_version`; positions above the old chunk count are brand-new records
carrying neither. The episode ends up **non-uniform**, which matters for §3.1.

Rows 2 and 3 are latent today — `RULES_VERSION` has never moved, and an
episode is normally transcribed once. Latent is not absent: `RULES_VERSION`
exists so a chunking change forces a re-embed, and the first use of it would
carry every stale name across a boundary shift with no alarm.

### 1.2 Two paths produce records with no identity, and neither recovers

`decide_action` branches on `complete_in_chroma` and `RULES_VERSION` and never
on identity, so nothing ever schedules a repair. Three producers, not the two
originally listed:

- an `EMBED_ONLY` re-embed has no audio and cannot identify;
- an enrolment failure stores the whole batch unstamped (`transcribe.py`
  reports it loudly and continues, by design — identity failure must not stop
  the corpus growing);
- `identify_clusters` raising on one episode does the same for that episode.

### 1.3 Why `speaker_id`-absence cannot be the signal

`identify_clusters` returns **only** the clusters that matched — an absent key
*is* the unidentified state, and 266 of the 970 archive clusters (27%) are
legitimately in it. A planner that queued "no `speaker_id`" would re-download
the audio of every episode nobody enrolled appears in, every night, forever.
That is the non-terminating shape invariant 3 was written to kill.

The signal has to record that identification **ran**, not that it **matched**.

---

## Part 2 — Why these are one design

Not because they touch the same file. Because **each one's fix is what makes
the other's affordable**, in both directions:

- Clearing a stale `speaker_id` means deleting records, and today a deleted
  `speaker_id` is *unrecoverable* — the 433 backfilled episodes cost 56 GB of
  someone else's bandwidth and a GPU pass, and nothing re-earns them. Metadata
  merge is what preserves them. **§1.2's repair path removes that
  constraint**: once an episode with no current identity is detected and
  re-identified, a cleared name is a scheduled repair rather than a loss.
- Conversely, §1.2's repair is only *safe* because §1.1's delete clears the
  record first. The repair writes metadata-only and can therefore add a name
  but never remove one — `stamp_chunks`'s limitation one layer up. A stale name
  sitting on a record when the repair ran would survive any cluster the repair
  declined to match, permanently, and the repair would report success.

The second bullet is what makes the phase order in Part 7 the only safe one.

---

## Part 3 — The design

### 3.1 `identity_version`, uniform across an episode

A new metadata key, defined in `corpus/identity.py` beside `RULES_VERSION`.

```python
IDENTITY_RULES_VERSION = "1"
IDENTITY_VERSION = f"{IDENTITY_RULES_VERSION}@{RULES_VERSION}"   # "1@1"
```

It answers one question: *did identification run over these chunks, under
these rules?* It **never carries an outcome**. `speaker_id` keeps its contract
unchanged — present means matched, absent means unmatched — and this key
carries what absence could never express. There is still no sentinel in
`speaker_id`, and the MCP filter is unaffected.

**The composite guards a reachable class, and its worked example is real.**
Any caller reaching `upsert_then_prune` outside the planner re-chunks without
carrying identity: bump `RULES_VERSION`, run it, and records acquire
`rules_version="2"` while keeping their old identity stamp. A plain scalar
compares equal, the episode reaches `SKIP`, and its stamps are positionally
stale. `transcribe.py::bulk_embed` was such a caller — §3.7 brings it under
§3.2's rule, so it is no longer one — and `migration/reid.py` still is. The
composite is kept because the class outlives its instances: it costs one
string comparison, and the alternative is a `SKIP` on stale stamps every time
someone adds a direct caller.

**Uniformity is a requirement, and §3.2 is what delivers it.**
`episode_state` reads metadata from one arbitrary record (`ids[:1]` after an
offset-paged get with no ordering guarantee — "the first id" is not "chunk
0"). That read is only sound if every record of an episode agrees. The
alternative considered and rejected was widening `episode_state` to fetch all
metadatas and require agreement: it detects the mixed state instead of
preventing it, and it pays a much larger nightly payload on every episode to
find a condition §3.2 makes unreachable.

### 3.2 Identity is carried forward, not recomputed, when the transcript is unchanged

**The cluster→person mapping is invariant under re-chunking.** `EMBED_ONLY`
and `RECHUNK` both re-chunk the *same transcript*, so its `[SPEAKER_XX]`
labels are unchanged and `build_chunks_from_text` derives the same `speaker`
values. Only the positional layout moves. The existing records already hold
the mapping.

So both paths read `speaker → speaker_id` from the episode's current records
before writing, and re-apply it to the new chunks **joined on `speaker`,
never on position**:

```python
# corpus/identify.py
def carry_identity(stored: list[dict], chunks: list[dict],
                   *, identity_version: str) -> tuple[int, list[str]]
```

It returns the number of chunks stamped, the clusters it **refused** to carry,
and the incoming labels it found **unmapped**. A cluster whose stored records
disagree — the same `speaker` label mapping to two different people — is
refused. This is the design's only self-checking read of its own past output,
and it costs one comparison.

Two things that disagreement can mean, and the report must not claim the
first: usually the stamps were corrupt, but a **same-day unnumbered triple
collision** (still open in `docs/corpus-integrity.md`) makes two distinct
episodes share `episode_id_prefix` and interleave, at which point one label
legitimately carries two people. Refusing is right either way; calling it
stamp corruption is not. The report says the label carries conflicting names
and names the triple, so an id-scheme fact is not filed as a data fault.

**Where a label is stamped on some records and absent on others, present
beats absent** — that is the §1.1 row 3 shape, where the absent records are
simply the ones written before identification ran. Written down because it is
the common case and the first draft left it to the implementer.

**Only `SPEAKER_*` keys are carried.** A chunk's `speaker` is not always a
diarisation cluster: `corpus/chunking.py:71,88` substitutes the literal
`UNKNOWN` for a segment carrying no speaker, and **53 of the 439 transcripts
hold such a bucket** (60 turns, 4 of them over the 12s crop minimum, longest
30.0s — measured 2026-09-05). That bucket is audio belonging to nobody in
particular, and naming it would be the silent over-merge this line of work
treats as the error that matters most.

It is currently unreachable rather than merely unlikely:
`corpus/embedding.py::cluster_crop_plan` filters clusters to
`startswith("SPEAKER_")` by rule, so `UNKNOWN` never reaches `assign`, never
enters `identified`, and therefore never acquires a `speaker_id` for
`carry_identity` to find. **The guarantee is real but it is inherited from
another module.** `carry_identity` reads stored records rather than
`identify_clusters`' output, so it must not depend on a filter three modules
away staying where it is; it applies the same rule itself, where a reader of
this function meets it.

`identity_version` is written on **every** chunk whenever the map is carried,
including chunks of clusters that matched nobody, which is what makes §3.1's
uniformity hold by construction rather than by hope.

**The map is keyed on every stored label, valued person-or-`None` — attempt,
not outcome.** Keying it only on *named* clusters cannot work: 27% of clusters
legitimately match nobody, so a label with no name would be indistinguishable
from a label that has gone missing. The map records which clusters were
*considered*. This is the same rule `identity_version` enforces one level up,
applied one level down.

**Carry only when the incoming label set is a subset of the stored one.
Otherwise carry nothing and withhold `identity_version`**, sending the episode
to `IDENTIFY_ONLY` to re-earn from audio. One rule, two failure modes it
closes, both of which a first draft of this section got wrong:

1. **The invariance claim is false on exactly the path `RECHUNK` exists for.**
   `corpus/identity.py`'s own comment says `RULES_VERSION` is *"bumped
   whenever anything that derives chunk CONTENT from a transcript changes —
   today the speaker labels"*, and the chunk document is literally
   `f"[{speaker}] ..."`. So a rules bump is precisely the event that may
   change how `speaker` is derived. The transcript's labels are invariant; the
   *derivation* is not, and the first draft asserted the former and used the
   latter. Under §3.4 the source records are deleted first, so a silent join
   miss would destroy the only copy of the map.
2. **A cluster can be missing from the map with no rules bump at all.** A tear
   removes a contiguous tail of positions, so a cluster appearing only in that
   tail has no stored record and no map entry. Writing `identity_version`
   anyway would send the episode to `SKIP` with that speaker permanently
   unnamed and nothing signalling it. This one is live in phase 1.

The subset test catches both without a new version constant, and it fails
safe: toward re-earning from audio, never toward certifying a gap.

Consequences, which are most of why this design got smaller rather than
larger:

- A `RULES_VERSION` bump that leaves label derivation alone re-carries every
  name in the same pass. **No corpus-wide nameless window, and no
  archive-sized audio queue.** One that changes label derivation fails the
  subset test and falls back to audio — correctly, and loudly.
- A torn episode's new positions are stamped from the map **when every
  incoming label is accounted for**; when one is not, the episode is left
  queued rather than certified.
- Audio is needed only for episodes never identified, whose map was refused as
  self-contradictory, or whose label set no longer matches.

**Carrying applies only where the transcript is unchanged.** A re-transcribe
produces a fresh diarisation whose labels mean something different; the old
map is invalid there and must not be carried. That path deletes and
re-identifies from audio, exactly as `transcribe.py` already does.

### 3.3 `stamp_chunks` writes the version, except when the join is broken

```python
def stamp_chunks(chunks, identified, *, identity_version: str) -> int
```

Writes `speaker_id` on matched clusters exactly as today, and
`identity_version` on **every** chunk — unless `identified` is non-empty and
zero chunks were stamped.

That exclusion reuses the existing `STAMP JOIN BROKEN` predicate. Clusters
named and not one chunk taking a value can only mean the `speaker` metadata
key drifted between `chunking.py` and this function. Stamping the version
there would record "identification ran" for an episode that stored no
identity, retiring it from repair forever and converting a loud nightly
failure into a silent permanent one. One predicate, two consumers; the alarm
in `transcribe.py` is unchanged.

Not writing the version is also how §1.2's other two producers self-heal: an
enrolment failure and an `identify_clusters` exception both skip this call, so
the episode is stored unstamped and is queued for repair. No new code for
either.

### 3.4 `upsert_then_prune` gains `replace_identity`

```python
def upsert_then_prune(..., replace_identity: bool = False, ...) -> dict
```

When true, it deletes the episode's existing records **before** the upsert,
then upserts, then prunes as today. The result dict gains `deleted`.

**The delete set is `existing - non_episode`** — the union of the triple arm
and the guid arm scoped by show, minus non-episode records, **including the
ids about to be rewritten**. It is *not* `to_prune`. `to_prune` is that set
after `stale_ids` subtracts the incoming ids, which excludes precisely the
records carrying the stale stamp; reusing it would ship a fix that deletes
nothing while every record-count test stays green.

The non-episode guard applies to the delete unchanged — `upload_book.py`'s 191
records can share a triple or a guid by construction and must be as
un-deletable here as they are un-prunable.

Set by the caller in the two cases that move boundaries:

- `TRANSCRIBE` — unconditionally true. A new episode's delete matches nothing,
  so no branch is needed and none is written.
- `RECHUNK` — the new action of §3.5, meaning the stored `rules_version`
  differs from the current one. Its chunks arrive carrying the carried-forward
  map from §3.2, so the delete removes stale positional stamps and the upsert
  immediately restores correct label-joined ones.

`EMBED_ONLY` leaves it false: same transcript, same rules, chunk *i* lands
where it did. It carries the map forward to cover the new positions, and needs
no delete to do so.

**Rejected: deriving the flag from the data.** Hashing the transcript, or
comparing stored against incoming `identity_version`, can both false-positive
(an encoding difference between the string written and the string read back;
an `EMBED_ONLY` write legitimately carrying no stamp), and a false positive
here is a **delete**. An explicit flag can only be wrong in the direction of
not firing, which leaves today's behaviour.

### 3.5 `decide_action` — two new members, and a reordering

```
excluded                                    -> EXCLUDE
not transcript_exists                       -> TRANSCRIBE
not parses_to_chunks                        -> UNPARSEABLE
stored_rules_version != RULES_VERSION       -> RECHUNK           (new)
not complete_in_chroma                      -> EMBED_ONLY
stored_identity_version != IDENTITY_VERSION -> IDENTIFY_ONLY     (new)
                                            -> SKIP
```

`stored_identity_version` is passed as a value, not a pre-computed boolean,
for symmetry with `stored_rules_version`.

**`RECHUNK` rather than a side channel.** `EMBED_ONLY` has two causes needing
different write behaviour, and the reason lives in `decide_action`. Threading
it to the worker as an extra dict key would put a load-bearing fact outside
the pure, total, millisecond-testable function whose purpose is to hold
exactly these facts — and `ShowPlan` documents that its episode dicts cross a
process boundary unchanged. `RECHUNK` routes to the same worker and differs
only in `replace_identity=True`.

**The rules check now precedes the completeness check.** Today both return
`EMBED_ONLY` so their order is unobservable; splitting them makes it
observable and it must be chosen. Rules-first is correct: an episode that is
both torn and rules-stale needs the stronger action, and `RECHUNK` does
everything `EMBED_ONLY` does.

**Two consequences, both real, neither a defect:**

1. An episode with a transcript and **zero** records has
   `stored_rules_version is None`, so it is `RECHUNK`, writing with
   `replace_identity=True` against an empty set. Harmless — the delete matches
   nothing — and it is the right treatment for pre-migration records, whose
   positional stamps are of unknown provenance.
2. That is also the state a delete-then-crash leaves behind, so the recovery
   path in Part 6 returns `RECHUNK`, not `EMBED_ONLY`. Stated here because a
   stop rule that reads "a non-zero `RECHUNK` count means stop" would
   otherwise halt a correctly self-healing system (see Part 5).

**`IDENTIFY_ONLY` is placed after both.** Identity is derived from chunks, so
chunks settle first. With §3.2 carrying the map, an episode reaching
`IDENTIFY_ONLY` is one that genuinely has no identity to carry — never merely
one whose boundaries moved.

`corpus/completeness.py::episode_state` returns `identity_version` from the
head-record read that already yields `n_chunks` and `rules_version`; no extra
round trip. `plan_episode` forwards it.

### 3.6 `plan_show` and `ShowPlan`

`ShowPlan` gains `to_identify: list[dict]`, and `has_work` includes it.
`RECHUNK` appends to `to_embed` alongside `EMBED_ONLY` — the worker is the
same — with the flag carried by the action, not by the dict.

The existing `else: raise RuntimeError(f"unhandled Action ...")` already
guarantees a new member cannot be silently dropped.

### 3.7 The repair runner

A new GPU function in `transcribe.py`'s app, `repair_identity(show_name,
to_identify) -> list[str]`, `timeout=7200` like its siblings, started only
when `to_identify` is non-empty. It does **not** reuse `process_episodes`,
which loads the BGE `SentenceTransformer` at container start before its
`to_embed` loop — a pure identity repair needs neither that nor whisper.

Per episode:

1. download the mp3 (`episode["audio_url"]`, from the feed entry the plan
   carried);
2. read the **saved transcript** from the volume and parse it with
   `corpus.chunking.parse_transcript_segments` — the same parser
   `build_chunks_from_text` feeds, so its `SPEAKER_XX` labels are the ones in
   each chunk's `speaker` metadata, correct by construction;
3. `identify_clusters(segments, audio_path, embedder, enrolment, ...)`;
4. metadata-only `collection.update()`: `speaker_id` for matched clusters
   joined on `speaker`, `identity_version` on every chunk of the episode;
5. delete the audio.

A mid-run kill is harmless: every write is a metadata-only `update()` on one
episode, and an episode that did not get its version stamped is simply still
queued.

**Never re-diarise.** The saved transcript's labels are what the stored
`speaker` metadata contains. Re-diarising produces different `SPEAKER_XX`
labels and the join would be to a different world.

It carries its own copy of §3.3's broken-join predicate, because it joins
against records in Chroma rather than chunk dicts on their way to a write: if
clusters were identified and **no stored record** carried a matching `speaker`
value, write nothing at all — not even the version — report the episode, and
leave it queued.

**It also refuses when a stored `speaker_id` contradicts what it just
identified**, and this is not belt-and-braces — it closes a hole Part 2
claimed was already closed. Part 2 says a stale name cannot survive into a
repair "because §1.1's delete clears the record first", but `IDENTIFY_ONLY`
**never deletes**; only `TRANSCRIBE` and `RECHUNK` do. A metadata-only
`update()` can add a name and never remove one, so an episode arriving here
with positionally-stale stamps would have its matched clusters corrected, its
declined clusters left wrong, and the run would report success — verbatim the
failure Part 2 describes.

Fixed at the source rather than papered over downstream. The path that
produces that state is a re-chunk with no delete and no stamping —
`transcribe.py::bulk_embed` calls `embed_and_store` directly, outside the
planner. **`bulk_embed` is brought under §3.2's rule**: it re-chunks from the
same transcript, which is exactly the `EMBED_ONLY` shape, so it carries the
map and writes `identity_version` like every other re-chunk. The refusal above
is then the detector for anything else that reaches the same state, including
`migration/reid.py` and any future direct caller.

Two extractions, so neither exists twice to drift:

- the enrolment/embedder block inside `process_episodes` becomes
  `_load_enrolment(hf_token, show_name) -> (enrolment, embedder, failures)`,
  keeping its `else`-not-`if` reporting distinction;
- the audio download block becomes `_download_audio(audio_url, dest)`.

`transcribe()` calls `repair_identity.remote` after `process_episodes.remote`,
so chunk work settles before identity work.

### 3.8 Bounds: a rate cap, and a gate on the event

```python
# corpus/planning.py
IDENTITY_REPAIR_CAP = 25
RECHUNK_CAP = 25

def cap_work(episodes: list[dict], cap: int) -> tuple[list[dict], int]:
    """The episodes to act on now, and how many are left over."""
```

**Both `IDENTIFY_ONLY` and `RECHUNK` are capped.** Capping only the former was
a defect in the first draft: `RECHUNK` deletes before writing, and an
uncapped 433-episode run under `timeout=7200` is the silent-kill case
invariant 2 was written about — now with delete-first writes under it.
Recovery works, but the hole stays open until the next run, and capping bounds
how many holes one timeout can open.

**After §3.2, `RECHUNK_CAP`'s justification is blast radius alone — not
bandwidth.** A rules bump no longer fetches audio, so the cap is there purely
to bound how many episodes one silent timeout can leave holed. Saying so
matters: a bound defended by a reason that no longer applies is the deferred
minor this project has already been bitten by twice.

**A secret-based bulk gate was designed and dropped.** The first draft
required an operator value in the Modal secret (`IDENTITY_BULK_OK`) before a
backlog above the cap could be repaired, reasoning that a cap bounds the rate
but does not gate the event. Three problems, any one fatal: `scheduled_job`
invokes `transcribe` once per show, so one secret value cannot match three
per-show backlogs; an exact-match comparison would refuse on night 2 of a
drain (433 → 408), and the at-least comparison that fixes it becomes ambient
state that stays set and silently permits the *next* bulk run; and Part 5
refuses env vars for the caps on the grounds that a bandwidth bound
changeable in the Modal dashboard with no code diff is the shape
`resolve_collection_name` exists to refuse — putting the larger bound in
exactly that shape contradicts it. Unlike `--expect`, it is not supplied per
invocation.

**What gates the event instead:** after §3.2 the only trigger for an
archive-sized audio spend is an `IDENTITY_RULES_VERSION` edit in source, and
Part 5's caps-at-0-then-raise already makes that two reviewed commits. The
gate is the source edit, which is the thing `--expect` was always standing in
for.

**Caps are per show, per run.** `scheduled_job` calls `transcribe.remote` once
per show behind its own `try`, so there is no cross-show accumulator, and
threading a budget through a call that raises on failure would lose it.

Bandwidth, stated as the total it is rather than as a sustained rate:

| | episodes | bandwidth |
|---|---|---|
| Steady state | 0–1 per night | negligible |
| First night after the stamp pass | the episodes published since the archive boundary (Part 5) | small, bounded, pre-registered |
| A `RULES_VERSION` bump, label derivation unchanged | **0 audio** — §3.2 carries the names | none |
| A `RULES_VERSION` bump that changes label derivation | the episodes failing §3.2's subset test | bounded by the cap, drained nightly |
| An `IDENTITY_RULES_VERSION` bump | 433 over ~14 nights, gated by the source edit | 56 GB total, ~9.7 GB on the busiest nights only |

The busiest-night figure holds for the 3–4 nights until Geopolitical Cousins
(76/25 = 4 nights) and Observing Japan (7/25 = 1 night) drain; after that only
Jacob Shapiro consumes slots. `25` is the one number here chosen rather than
measured.

### 3.9 What the operator sees

A backlog reported once per show as a bare number is indistinguishable, on
night 6 of a drain, from a queue that has been stuck since night 2 — unless
someone diffs yesterday's log. `transcribe.py`'s own `STAMP JOIN BROKEN`
comment makes exactly this argument about a cheerful nightly line that means
nothing.

So the run reports the absolute backlog, and separately verifies the work it
just did:

```
identity backlog 312
repaired 25, verified current 25
```

and identity-repair failures are reported under their own heading with the
episode triple and the exception, not folded into the undifferentiated
failure text — so a repeatedly-failing episode is recognisable at a glance
rather than by reading identical prose each night.

**The second number is re-measured, not subtracted.** After
`repair_identity.remote` returns, the run re-plans **only the episodes it just
repaired** — at most `IDENTITY_REPAIR_CAP` of them — and reports how many are
now identity-current. Arithmetic (`337 - 25`) would be blind to precisely the
failure a metadata-only `update()` invites: a repair that reports success and
writes nothing. A re-measurement catches it the same night, and it costs a
bounded number of round trips rather than a second full plan pass.

`repaired 25, verified current 25` is a measurement.
`312, was 337 (-25)` is a restatement of what the run already believed.

**This is a within-run check, not a night-over-night trend.** The run has no
memory of yesterday, and none is added: the absolute backlog printed each
night is what a human compares across nights. Saying which of the two it is
matters, because §3.9's motivating case — night 6 of a drain versus stuck
since night 2 — is answered by the absolute number moving, while the verified
count answers a different and equally silent question: whether tonight's
repairs actually landed.

### 3.10 A failed repair

Writes nothing. The episode stays `IDENTIFY_ONLY`, is retried next run, and is
reported under §3.9's heading until it succeeds or is acted on. A transient
failure heals with nothing to remember; a permanent one costs one capped slot
per night and announces itself.

**Known limitation, named rather than guarded.** `parse_all_episodes` only
emits entries carrying an enclosure URL, so an episode aged off its feed never
enters a plan and can never be queued for repair. This mirrors
reconciliation's `feed_unreachable` class: a fact about the feed, not a fault
in the corpus. It also bounds the poison-pill risk — a permanently failing
repair needs an episode still *in* the feed whose download keeps failing.

---

## Part 4 — The one-time stamp

The signal is worthless while `identity_version` is absent everywhere, and
treating absent as "needs identity" would queue all 433 archive episodes on
night one to recompute names that are already correct.

**The stamp set is the 433 triples in `speakers/assignments.json`** — the
archive pass's own record of what it processed. Measured 2026-09-05, and
independently re-measured during review: 970 cluster records over **433
distinct episodes**, 704 named and 266 not, every one carrying the full
`(show, episode_number, date)` triple, 0 records missing any part of it.

Provenance, not outcome. "Has a `speaker_id`" would have selected the same 433
here — 0 of the 433 have zero named clusters, so the selectors coincide
exactly — but it reads the wrong field, and it would silently exclude any
episode the pass processed and matched nobody in.

A **new script**, `stamp_identity_version.py`, writing `identity_version` to
**every chunk** of those 433. It needs its own planner: `plan_updates` filters
by cluster and skips `person: None`, so it would leave the 266 unmatched
clusters' chunks unstamped — precisely the uniformity §3.1 depends on. Its
per-episode triple query must apply `is_non_episode`, as the write path does,
or it will stamp book records sharing a triple.

It imports `_require_expect` from `backfill_speaker_id.py` rather than
restating the gate, so the dry-run → `--expect` flow is the same one and not a
lookalike. **Verified by reading back** from the collection, not from the
writer's return value — the pattern `tests/test_backfill.py` already
establishes.

### The six-episode gap, resolved

439 transcripts exist on the volume; 433 were identified. The gap is
`The Jacob Shapiro Podcast` episodes 1–6, dated 2022-05-02 to 2022-05-27.
`speakers/unavailable.json` holds `unresolved` clips spanning exactly those
six and `failed: []`, and `probe_audio_availability.py` defines that file as
six ABSENT, zero DEAD, where ABSENT means the key is not in the feed at all.

**They are the same six**, checked rather than assumed. The consequence
matters for Part 5: because they are absent from the feed,
`parse_all_episodes` never emits them and they can never be classified
`IDENTIFY_ONLY`. Their contribution to the first-night count is **0**.

---

## Part 5 — Landing order and pre-registered numbers

Within each phase of Part 7:

1. Land the code with the caps at `0` — the planner classifies and reports and
   acts on nothing. One night of real classification against production with
   no bandwidth spent. The caps are constants edited by a one-line follow-up
   commit, deliberately not env vars: a bandwidth bound changeable in the
   Modal dashboard with no code diff is the shape `resolve_collection_name`
   exists to refuse.
2. Run the stamp pass: dry run, pre-register, apply with `--expect`, verify by
   reading back.
3. Confirm the next night's plan reports the expected counts.
4. Raise the caps to 25.

Deploying the signal before the stamp inverts this: night one queues 433
episodes.

### The numbers, written down before running

| Quantity | Expected | Source |
|---|---|---|
| `IDENTIFY_ONLY` on the step-1 night, **before** the stamp | near 439 minus exclusions — essentially every episode | nothing carries the key yet; a *small* number here means the planner is not reading it |
| Episodes stamped by the one-time pass | **433** | `speakers/assignments.json`, counted twice |
| Distinct shows in that set | **3** (350 / 76 / 7) | same |
| Records updated by the pass | *from the dry run* | must be carried to `--apply --expect` |
| `IDENTIFY_ONLY` on the first post-stamp night | **the episodes published since each show's archive boundary** — 2026-08-31, 2026-08-22, 2026-06-19 — enumerated from the feeds *before* the run | not zero, and not the six aged-off episodes, which contribute 0 |
| `RECHUNK` on the first night | **0** | `RULES_VERSION` has not moved, and `migration/reid.py` stamped it onto every migrated record |

The first-night `IDENTIFY_ONLY` population is the archive pass's date boundary,
not the six-episode gap. Live-ingest identity landed 2026-09-04 (`1f9bb83`),
after the archive pass, so episodes published in between are outside the
433-triple stamp set and are legitimately repaired — including a few that
already carry a correct `speaker_id` and will be re-downloaded to re-derive
it. Bounded by days, enumerated in advance, and accepted rather than special-
cased: the alternative is selecting part of the stamp set by outcome, which is
the rule §4 exists to refuse.

**Stop rules.** A first-night `IDENTIFY_ONLY` of 433 means the stamp did not
land. A non-zero `RECHUNK` count means one of three things: `RULES_VERSION`
moved, the head-record read is not returning what it did, **or an episode has
a transcript and no records at all** — the third is self-healing and expected
after a crash, so investigate the cause before stopping rather than stopping
on the count alone.

---

## Part 6 — The amendment to invariant 2

`docs/corpus-integrity.md` invariant 2 says writes are upsert-then-prune,
never delete-then-upsert, because a Modal timeout kills the container without
raising anything an `except` can see, and delete-first would leave "a deleted
and never rewritten episode with no log line."

**That rationale predates invariant 3.** It was written when the planner asked
*presence*, and a hole with one surviving record answered "present" and was
never repaired. Completeness-not-presence changed the outcome: zero or partial
records now fail `is_complete`. Verified against the code rather than
asserted: with zero records `episode_state` returns `([], None, None)`, so
`stored_rules_version is None`, and under §3.5's ordering the episode is
planned as **`RECHUNK`** — which rewrites it from a transcript already
committed to the volume, with no GPU and no download. On the `TRANSCRIBE` path
`volume.commit()` runs before `build_chunks`, so the input is durable before
any Chroma write happens.

So the window this design opens does not risk permanent loss **of chunks**. It
risks one night of missing records, self-repaired. §3.8's cap bounds how many
episodes a single timeout can leave in that state.

### Invariant 3 does not cover identity, and that gap is not free

The paragraph above is the argument the first two drafts made, and on its own
it licenses the wrong thing. **Invariant 3 guarantees recovery of CHUNKS from
the committed transcript. It says nothing about identity.** A delete followed
by a timeout destroys the only copy of the cluster→person map; `RECHUNK`
rebuilds the chunks with no identity, and re-earning identity needs the audio,
which needs the episode to still be in its feed. §3.10 already concedes an
aged-off episode can never be queued for repair — and JSP episodes 1–6 are the
standing proof that these feeds do age episodes off. For an episode that ages
off inside that window, identity loss is **permanent and silent**.

Cheap mitigation, adopted: **write the `speaker → speaker_id` map to the
transcripts volume immediately before any delete**, and have the repair path
read it before falling back to audio. The volume is already durable and
already committed at that point in both write paths, the artifact is a few
hundred bytes per episode, and it converts the delete from a one-way door into
a two-way one for the only thing invariant 3 could not recover. Without it
this design would be trading a wrong name for an unrecoverable missing one,
which is the trade the whole spec argues against making.

The invariant is narrowed, not deleted, with the narrowing and its condition
written into the page: *upsert-then-prune remains the default; a delete is
permitted only where the write replaces identity, and only while
completeness-not-presence guarantees the repair.* If invariant 3 is ever
weakened, this permission goes with it.

---

## Part 7 — Two phases

Phase 1 ships §1.2. Phase 2 ships §1.1, after phase 1 has run against
production for a week.

**Phase 1:** `identity_version` and the composite; `stamp_chunks` writing it
with the broken-join exclusion; `carry_identity` with the subset rule, on the
`EMBED_ONLY` path **and on `bulk_embed`**; the `TRANSCRIBE` guard above;
`IDENTIFY_ONLY`; `repair_identity` with the contradiction refusal; `cap_work`;
the absolute backlog plus the verified-repair count;
`stamp_identity_version.py`; Part 5's landing order.

**Phase 2:** `replace_identity`; `RECHUNK` and the `decide_action` reorder;
`carry_identity` on the `RECHUNK` path; `RECHUNK_CAP`; **the map persisted to
the volume before any delete, and the repair reading it before falling back to
audio**; the invariant 2 amendment; the inverted write-path test.

**Why this order and not the reverse.** §1.2 has a live population today;
§1.1's requires a re-transcribe or a rules bump, neither of which has
happened. And Part 2's coupling is directional: the delete is affordable
*because* the repair exists to re-earn a cleared name. Landing §1.1 first
would open a window where a cleared `speaker_id` has nothing to restore it.

**Phase 1 is not purely additive, and an earlier draft claimed it was.** The
claim was that phase 1 "leaves §1.1 exactly as found". It does not.
`stamp_chunks` ships in phase 1 and runs on the `TRANSCRIBE` path, which in
phase 1 has no delete. So a re-transcribe during the phase-1 week would write
`identity_version` onto records still carrying positionally-stale
`speaker_id`s — **certifying** them. Today such an episode is merely wrong;
after an unguarded phase 1 it is wrong *and* marked identified, which retires
it from `IDENTIFY_ONLY` permanently. Phase 2 ships no rescan, so the only
recovery would be the 433-episode `IDENTITY_RULES_VERSION` bump. The same
shape applies to `carry_identity` on an `EMBED_ONLY` over surviving records
that already carry stale stamps.

**Phase-1 guard, in code and not only in a document:** on the `TRANSCRIBE`
path, if the episode already has stored records carrying any `speaker_id`,
withhold `identity_version` and report it. The episode stays queued for
`IDENTIFY_ONLY`, where §3.7's contradiction refusal reports it again rather
than half-repairing it. Loud and unresolved beats silent and certified for the
one week this applies. Phase 2's delete makes the guard redundant; it is
cheap enough to leave in place rather than remove.

**Two constraints for `docs/operations.md`, not one:**

1. Do not bump `RULES_VERSION` between phases — `EMBED_ONLY` would carry the
   map forward correctly, but without `RECHUNK`'s delete the old positional
   stamps survive on records the map does not cover.
2. Do not force a re-transcribe of an already-identified episode between
   phases. The guard above makes this safe rather than silent, but the
   episode still ends up needing an audio repair it would not otherwise have
   needed.

---

## Testing

A check that cannot fail is worse than no check; each of these can.

**`tests/test_planning.py`** — pure, total, milliseconds.
- `RECHUNK` when `stored_rules_version` differs, including `None`.
- `RECHUNK` wins over `EMBED_ONLY` when an episode is both torn and rules-stale.
- `IDENTIFY_ONLY` when the identity version differs, including `None`.
- `IDENTIFY_ONLY` does **not** fire while the episode is torn or rules-stale.
- `SKIP` only when complete, current, and identity-current.
- `cap_work`: under, at, over; the remainder count; `cap=0` acts on nothing and
  reports the full backlog.

**`tests/test_identify.py`**
- `carry_identity` joins on `speaker`, not position: a map applied to a chunk
  set whose boundaries moved lands every name on the right cluster. **Fixtures
  must have chunks at different positions than the stored records** — a
  same-position fixture would pass under a positional implementation and make
  this a check that cannot fail.
- A cluster whose stored records name two different people is refused,
  reported, and left unstamped.
- A stored record whose `speaker` is `UNKNOWN` contributes nothing to the map
  and receives no name, even when the fixture gives it a `speaker_id` — the
  fixture must supply one, or the test passes against an implementation with
  no such rule and becomes a check that cannot fail.
- **The subset rule**: an incoming label absent from the stored map causes the
  whole carry to be abandoned and `identity_version` to be withheld. Two
  fixtures, because they are different causes with one remedy — a label
  renamed (§3.2 case 1) and a label present only in a torn episode's missing
  tail (case 2).
- A label stamped on some stored records and absent on others carries the
  name: present beats absent.
- A cluster with conflicting names is reported as conflicting, naming the
  triple, and is **not** described as corruption.

**Repair path (`repair_identity`)**
- Refuses and reports when a stored `speaker_id` contradicts the freshly
  identified map, rather than correcting the matched clusters and leaving the
  declined ones wrong.
- `bulk_embed` carries the map and writes `identity_version`, so it can no
  longer produce the state that refusal exists to catch.
- The verified-repair count catches an `update()` that reports success and
  writes nothing. The fake must be able to *lie* — an always-succeeding stub
  makes this a check that cannot fail.

**Phase-1 guard**
- A `TRANSCRIBE` write over an episode whose stored records carry a
  `speaker_id` withholds `identity_version` and reports; the episode is still
  classified `IDENTIFY_ONLY` on the next plan.

**Map durability (phase 2)**
- The map is on the volume before the delete happens, not after — assert the
  ordering, since a test that only checks the file exists at the end passes
  against a write placed after the upsert.
- The repair reads a persisted map in preference to fetching audio.
- `identity_version` lands on every chunk, including chunks of clusters that
  matched nobody.
- It is **not** written when `identified` is non-empty and `stamped == 0`; it
  **is** written when `identified` is empty — nobody matched is a completed
  identification, and that episode must reach `SKIP` rather than loop.

**`tests/test_write_path.py`**
- `test_a_stale_speaker_id_survives_a_re_transcribe` is **inverted** into
  `..._is_cleared_by_a_re_transcribe`; same scenario, only the assertion moves.
- `replace_identity=False` preserves a backfilled `speaker_id` across a
  re-embed — the test that stops someone globalising the delete.
- The delete covers ids that are about to be rewritten, not merely the stale
  tail. Assert on a **document or metadata value**, not on a record count: a
  count-only assertion passes against the `to_prune` misreading of §3.4.
- Book records survive a `replace_identity=True` write at their own triple, and
  one reachable only via the guid arm.
- A crash after the delete and before the upsert leaves zero records, and
  `plan_episode` then returns **`RECHUNK`**. Part 6's recovery claim asserted,
  not narrated.
- `deleted` is reported, and deletes are batched at `BATCH` like the prune.

**`tests/test_showplan.py`** — `to_identify` populated; `has_work` true for an
identity-only plan; `RECHUNK` lands in `to_embed`.

**The stamp pass** — refuses without `--expect`; refuses when the planned count
disagrees; stamps chunks of unmatched clusters; skips non-episode records at a
shared triple; verified by reading back from the collection.

---

## Acceptance

1. An episode stored without identity is classified `IDENTIFY_ONLY` on the
   next plan and carries `speaker_id` and `identity_version` after one repair.
2. An episode where identification ran and matched nobody reaches `SKIP` and
   stays there. **The termination test** — plan twice, confirm the second run
   does not re-queue it.
3. A torn episode's `EMBED_ONLY` repair leaves every chunk carrying
   `identity_version`, including the newly created positions.
4. A simulated `RULES_VERSION` bump re-carries every name with no audio
   fetched and no episode entering `IDENTIFY_ONLY` (phase 2).
5. An already-identified episode, re-transcribed, ends with no stale
   `speaker_id` (phase 2).
6. An episode whose label set no longer matches its stored map is left queued
   rather than certified, and says so in the nightly output.
7. A repair that writes nothing is reported as unverified the same night,
   not counted as done.
8. The pre-registered counts in Part 5 are met, or the run is stopped.
9. `docs/corpus-integrity.md` invariant 2 carries the narrowing and its
   condition (phase 2).

---

## Out of scope

- **Item 3**, verifying the live ingest identity path on real data.
- **Item 4**, `HF_HOME` leaking into `DiarizationPipeline`.
- Same-day unnumbered id collisions, and index-shifted in-place repair. Both
  remain open in `docs/corpus-integrity.md`.
- Re-identification triggered by an enrolment change: bumping
  `IDENTITY_RULES_VERSION` already does it.

---

## Housekeeping

A corrected fact survives uncorrected in every sibling written at the same
time, so:

- `docs/corpus-integrity.md` — invariant 2 narrowed (Part 6); the still-open
  list.
- `docs/speaker-identity-decisions.md` — §3's "so no future backfill is
  needed" and §4's known exposure both resolve here.
- `corpus/writing.py` docstring — the merge paragraph; still true, now say
  which caller deletes and why.
- `corpus/identify.py::stamp_chunks` docstring — its KNOWN EXPOSURE block.
- `docs/speaker-identification.md` and `docs/operations.md` — the new actions,
  the new key, the caps, what a nightly `IDENTIFY_ONLY` line, an absolute
  backlog and a verified-repair count each mean, and **both** between-phases
  constraints from Part 7.
