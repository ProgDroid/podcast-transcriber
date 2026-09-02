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

`FEED_UNREACHABLE (6)` is expected and is **not** a fault: six early Jacob
Shapiro episodes have aged off the front of their feed.

> **A transient feed failure invalidates that show's verdict.** If
> `FEED_UNREACHABLE` lists *every* episode of a show, its feed did not load —
> that is the signature, as opposed to a handful of old episodes ageing off.
> `MISSING`, `EXTRA` and `NON_CONTIGUOUS` for that show are then **UNKNOWN**,
> not clean, because the comparison had nothing to compare against. Re-run
> before accepting the result. The discriminating check is the `feed=` count:
> a transient failure shows up as a drop there.

## Repairing a show

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
- **`n_chunks` on a search result is the version tell.** It is absent from
  pre-migration records and the reader passes it through with no default, so a
  result carrying it came from the new collection. This does not depend on
  trusting the deploy.
- **A warm container must be cycled, not waited out.** A container that
  entered `load()` before the cutover holds a `Collection` handle bound to the
  old collection. That handle has no name to re-resolve, so nothing raises and
  `get_collection` does not help — it serves the old data correctly and
  silently for as long as the container lives. `modal app stop
  podcast-mcp-server`, then redeploy.

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
