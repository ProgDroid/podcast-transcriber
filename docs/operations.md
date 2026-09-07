# Operations

Runbook for the things that are only done occasionally, and the verification
habits that caught real bugs while doing them.

## Verifying the corpus

```bash
uv run modal run migration/reconcile_report.py
```

Read-only. Compares the transcript volume, the store and the live feeds.
A healthy run looks like:

```
volume=439 feed=433 records=29160
MISSING (0)  EXTRA (0)  NON_CONTIGUOUS (0)  SHARED_PREFIXES (0)
EXCLUDED_WITH_RECORDS (0)  INCOMPLETE (0)  FEED_UNREACHABLE (6)  CLEAN: True
```

`FEED_UNREACHABLE` is expected and is **not** a fault: the earliest Jacob
Shapiro episodes have aged off the front of their feed.

> **This number GROWS — it is a moving baseline, not a constant.** The feed
> keeps a fixed window, so each time its front rolls, one more episode on the
> volume stops being feed-reachable. It read `6` (episodes 1-6) when this was
> written; **measured 2026-09-07 directly from the feed, episode 7 has also
> rolled off — the oldest entry the feed now emits is episode 8 (2022-06-03),
> so expect `7`.** Treat a value one higher than last time as normal, and only
> the *shape* described below as a fault. `volume=439 feed=433` above moves
> the same way: `feed` is a window, `volume` only grows.

> **A transient feed failure invalidates that show's verdict.** If
> `FEED_UNREACHABLE` lists *every* episode of a show, its feed did not load —
> that is the signature, as opposed to a handful of old episodes ageing off.
> `MISSING`, `EXTRA` and `NON_CONTIGUOUS` for that show are then **UNKNOWN**,
> not clean, because the comparison had nothing to compare against. Re-run
> before accepting the result. The discriminating check is the `feed=` count:
> a transient failure shows up as a drop there.

## Repairing a show

> Repairing *chunks*. For repairing *identity* -- an episode stored
> without a `speaker_id` because it was re-embedded with no audio, or
> because enrolment failed that night -- see **Identity repair** below.
> They are different actions with different containers, and a re-embed
> does not fix identity.

```bash
uv run modal run transcribe.py::transcribe \
  --feed-url "<feed url>" --show-name "<show name>"
```

Idempotent. Plans every episode and acts only where needed. Planning runs on
CPU; a GPU container starts only if there is real work.

Before running against the corpus, **pre-register what you expect** — which
episodes, which action, how many chunks — and compare afterwards. Every
production bug found in this project was found this way: by a real run's
output disagreeing with a number written down beforehand, never by reading
the code.

The plan output names an action per episode:

| Action | Meaning |
|---|---|
| `SKIP` | complete and current |
| `EMBED_ONLY` | transcript on disk, records missing or incomplete |
| `TRANSCRIBE` | no transcript |
| `EXCLUDE` | on the human exclusion list |
| `UNPARSEABLE` | transcript exists but yields no chunks — terminal |

## Cutting over to a new collection

Both apps read `CHROMA_COLLECTION` from the `podcast-secrets` Modal secret. It
is deliberately **one shared variable**: pointing writer and reader at
different collections is the failure it exists to prevent, and a cutover is
then one edit rather than two that must agree.

1. Copy and validate with `migration/` (see its own README notes).
2. Edit `CHROMA_COLLECTION` in the secret.
3. Redeploy **both** apps.
4. Verify with the tells below.

Three things worth knowing:

- **A wrong name fails loudly.** The reader uses `get_collection`, not
  `get_or_create_collection` — a typo raises instead of silently creating an
  empty collection and serving zero results, which looks exactly like a broken
  query and is much harder to diagnose.
- **There is no in-band version tell.** This runbook used to say `n_chunks` on
  a search result was one. It is not: `n_chunks` is put into the searcher's
  result dict at `mcp_server.py:126` and never rendered into the string a
  caller receives, so no consumer has ever been able to see it. Verified
  2026-09-02 against the deployed endpoint. It was also only ever a
  discriminator by accident — v2 happened to add the field — so a future
  migration that adds nothing would leave the tell silently useless, which is
  worse than having none at all. Use the log line below instead.
- **A warm container must be cycled, not waited out.** A container that
  entered `load()` before the cutover holds a `Collection` handle bound to the
  old collection. That handle has no name to re-resolve, so nothing raises and
  `get_collection` does not help — it serves the old data correctly and
  silently for as long as the container lives. `modal app stop
  podcast-mcp-server`, then redeploy.
- **After cycling, the log line is authoritative.** `load()` prints
  `Reading from collection: <name>` (`mcp_server.py:62`). That is the direct
  answer to what the reader is bound to, rather than an inference from a schema
  artefact. It is only unambiguous *after* a stop-and-redeploy, because that is
  what guarantees a single generation of containers — which is the same step
  the bullet above already requires, so the in-band tell was redundant with it.

If a future migration needs a probe readable from a response without stopping
production, **design one deliberately** — a small status tool returning the
collection name and count — rather than inheriting whatever field the new
schema happens to carry. Note it would cold-start a T4, since the search class
holds a GPU.

**`modal secret create --force` replaces the secret wholesale**, so re-supply
every key or you will silently drop one. Losing `MCP_ALLOWED_HOST` returns the
MCP endpoint to `421` on every call.

## Deleting records

> **There is no second copy of the corpus.** The v1 collection
> `podcast_transcripts` (28,541 records) was deleted on 2026-09-02 after a clean
> reconciliation, so `podcast_transcripts_v2` in `geo-podcasts-us` is now the
> only copy that exists — as is already true of the region, since the EU
> database went in July. Any destructive operation here is unrecoverable.

Two deletions have been performed: two cross-post duplicate episodes, and the
v1 collection. The procedure is worth repeating because the ordering is
load-bearing:

1. **Add the exclusion first, and deploy it.** Otherwise the 09:00 UTC cron
   restores what you deleted the next morning.
2. **Prove the surviving copy exists.** For a cross-post, the question is not
   "does the copy exist" but "does the *original* still hold records". If it
   does not, deleting the copy destroys the only copy.
3. **Pre-register the counts and assert them in the script**, so a collection
   that moved since the probe causes a refusal rather than an unknown-size
   delete.
4. Re-run reconciliation. `MISSING (0)` afterwards is the end-to-end proof the
   exclusion works, and a stronger one than a planning run: the episodes are
   still in the reachable feed, so their absence from `MISSING` can only mean
   the feed-derived triple matches the exclusion triple exactly — using the
   same derivation the planner uses.

## Identity repair, and the phase gate

Design: [`docs/superpowers/specs/2026-09-05-identity-lifecycle-design.md`](superpowers/specs/2026-09-05-identity-lifecycle-design.md).
**Phase 1 is built. Phase 2 is not.** The two constraints at the bottom of
this section apply for as long as that is true.

### The key, and why it is not `speaker_id`

`identity_version` (`"1@1"` — `IDENTITY_RULES_VERSION@RULES_VERSION`) records
that identification **ran** over an episode's chunks under these rules. It
never carries an outcome.

The distinction is the whole point. `speaker_id` present means matched, absent
means unmatched, and **266 of the 970 archive clusters (27%) are legitimately
unmatched** — nobody enrolled speaks in them. A planner keyed on `speaker_id`
absence would re-download the audio of every one of those episodes every
night, forever. `identity_version` is what lets an episode where nobody
matched reach `SKIP` and stay there.

It is written on **every** chunk of an episode, including the chunks of
clusters that matched nobody, because `episode_state` reads metadata from one
arbitrary record. Non-uniformity is the one state that read cannot survive.

### The new action

`IDENTIFY_ONLY` — complete, rules-current, and `identity_version` missing or
stale. Ordered **last**, after the completeness and rules checks: identity is
derived from chunks, so chunks settle first.

`repair_identity` acts on it. It re-reads the **saved transcript**, identifies
from audio, and writes **metadata only** — it never re-diarises (labels would
not match the stored `speaker` values) and never re-transcribes (that moves
chunk boundaries). A mid-run kill is harmless.

### Reading the nightly output

```
identity backlog 312 (repairing 25 this run, 287 left)
repaired 25, verified current 25
```

- **`identity backlog N`** is ABSOLUTE, and it is the number to compare across
  nights. A backlog reported only as a delta is indistinguishable, on night 6
  of a drain, from a queue that has been stuck since night 2.
- **`verified current`** is RE-MEASURED, not `25 - 0`. After the repair the run
  re-plans only the episodes it just repaired and asks whether they are now
  identity-current. Arithmetic would be blind to precisely the failure a
  metadata-only `update()` invites: reporting success and writing nothing.
  `verified` below `repaired` raises.
- **`IDENTITY REPAIR REFUSED`** — the repair declined to half-fix an episode.
  Most often a stored `speaker_id` the fresh run does not confirm. Refusals
  write nothing at all, so the episode stays queued and says so again
  tomorrow.
- **`IDENTITY CONFLICT`** — one label maps to two people in stored records.
  **Do not read this as corruption.** A same-day unnumbered triple collision
  makes two distinct episodes share `episode_id_prefix` and interleave, at
  which point one label legitimately carries two people. Check the triple
  before concluding the stamps are bad.
- **`IDENTITY NOT CERTIFIED`** — the phase-1 guard fired (see below).

### The landing order — do these in order

1. **Deploy with `IDENTITY_REPAIR_CAP = 0`** (its committed value). The
   planner classifies and reports and acts on nothing. One night of real
   classification against production with no bandwidth spent.
   - Expected on that night: `IDENTIFY_ONLY` near 439 minus exclusions —
     essentially every episode, because nothing carries the key yet. **A small
     number here means the planner is not reading it.**
2. **Run the stamp pass:**
   ```bash
   uv run modal run stamp_identity_version.py                      # dry run
   uv run modal run stamp_identity_version.py --apply --expect N   # N from the dry run
   ```
   Expected: **433** episodes across **3** shows (350 / 76 / 7). It verifies by
   reading back and refuses to report success on a partial stamp.
3. **Confirm the next night's plan.** Expected `IDENTIFY_ONLY` is the episodes
   published since each show's archive boundary — 2026-08-31, 2026-08-22,
   2026-06-19 — **enumerated from the feeds before the run**. Not zero, and
   not the six aged-off Jacob Shapiro episodes, which contribute 0 because
   `parse_all_episodes` never emits them.
   - **A first-night `IDENTIFY_ONLY` of 433 means the stamp did not land.**
4. **Only then raise `IDENTITY_REPAIR_CAP` to 25**, as a one-line commit. It is
   a constant and not an env var on purpose: a bandwidth bound changeable in
   the Modal dashboard with no code diff is the shape `resolve_collection_name`
   exists to refuse.

Caps are **per show, per run** — `scheduled_job` calls `transcribe.remote`
once per show behind its own `try`, so there is no cross-show accumulator.

### ⚠️ Two constraints that apply until phase 2 lands

1. **Do not bump `RULES_VERSION`.** `EMBED_ONLY` would carry the identity map
   forward correctly, but without phase 2's delete the old positional
   `speaker_id`s survive on records the map does not cover.
2. **Do not force a re-transcribe of an already-identified episode.** The
   phase-1 guard makes this safe rather than silent — `transcribe.py` reads the
   episode's prior records first, and withholds `identity_version` when any
   name survives, printing `IDENTITY NOT CERTIFIED` and failing the run — but
   the episode then needs an audio repair it would not otherwise have needed.

Without that guard, such a write would stamp `identity_version` onto records
still carrying positionally-stale names, **certifying** them: the episode
would be wrong *and* retired from `IDENTIFY_ONLY` permanently, and phase 2
ships no rescan.

## Pinning images

Each Modal image is built once and cached, so **each froze whatever versions
were current on its own build date**. The transcriber and MCP images name the
same unpinned requirements and resolved differently — torch 2.8.0 vs
2.11.0+cu128, transformers 4.57.6 vs 5.8.0, huggingface-hub 0.36.2 vs 1.14.0.

**The two images are not meant to agree.** Pin each to versions probed from
*its own* running image; carrying one image's numbers across to the other
would downgrade a working system. A pin the running system contradicts is
worse than no pin.

To probe an image's versions, replicate its spec **verbatim** so Modal's
content-addressing reuses the cache. If the run prints build lines instead of
reusing, the spec has drifted and the numbers are not the deployed ones.

`.add_local_python_source("corpus")` must stay **last** in the transcriber's
image chain.

## Watching a long Modal run

The local client is killed at ten minutes by some harnesses while the remote
function keeps going. Use `modal run --detach` so the run survives, then:

> **`modal app logs <id>` is not a completion probe.** It returns a window and
> exits 0 having possibly matched nothing, while the app still holds a running
> task. A short or empty log there is **UNKNOWN**, not failure.
>
> Completion is `modal app list` showing the app `stopped` with 0 tasks, **and**
> the terminal `All done.` marker present in the log. Check for the marker, not
> for the absence of errors.
