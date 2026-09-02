# podcast-transcriber

Serverless podcast transcription and semantic search on [Modal](https://modal.com).
Pulls episodes from RSS, transcribes them with WhisperX including speaker
diarisation, embeds the chunks with BGE, stores them in Chroma Cloud, and serves
the result as an **MCP server** so an LLM can search across everything that has
ever been said on the shows you follow.

It runs on a schedule, scales to zero between runs, and costs nothing while idle.

```
RSS feed ──> WhisperX (T4)      ──> chunks ──> BGE-large ──> Chroma Cloud
             transcribe+diarise                 embeddings        │
                                                                  ▼
                                          MCP server (streamable HTTP)
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                              Claude / IDE     Open WebUI      your own client
```

## Why this exists

Podcasts are a genuinely good source on slow-moving subjects, and they are almost
impossible to search. A phrase you half-remember from an episode eight months ago
is effectively gone. This makes the whole back catalogue queryable by meaning
rather than by keyword, and it returns the publication date with every hit so a
model can reason about whether a view is current or stale.

## What is in here

| File | What it does |
|---|---|
| `transcribe.py` | Modal app `podcast-transcriber`. Fetches RSS, transcribes with WhisperX on a T4, diarises speakers, chunks, embeds, upserts to Chroma. Includes a daily cron and a bulk backfill entrypoint. |
| `mcp_server.py` | Modal app `podcast-mcp-server`. Serves `search_podcasts` and `latest_on_topic` over MCP's streamable HTTP transport. |
| `upload_book.py` | One-off: chunk, embed and upload a PDF into the same collection. |
| `speaker_stats.py` | Local, no Modal. Recomputes the corpus's turn and speech-concentration figures from a transcript directory, so they can be reproduced rather than remembered. |
| `migration/` | Copy a Chroma Cloud collection between databases or regions, with validation. Chroma cannot move a database's region in place, so this exists to do it as a copy-validate-cutover. |
| `corpus/` | The pure logic: episode identity, planning, completeness, writes, exclusions, reconciliation. No Modal imports, so all of it is unit-tested. |

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/corpus-integrity.md`](docs/corpus-integrity.md) | How the corpus silently lost episodes, and the six invariants that now prevent it. **Read this before changing anything in `corpus/`.** |
| [`docs/chroma-cloud-behaviour.md`](docs/chroma-cloud-behaviour.md) | Chroma Cloud limits and semantics, all measured rather than documented — the 300-record cap and its silent form, `$eq`/`$ne` asymmetry, paging stability, metadata merge. |
| [`docs/operations.md`](docs/operations.md) | Runbook: verifying the corpus, repairing a show, cutting over a collection, deleting records safely, pinning images. |

## Prerequisites

- A [Modal](https://modal.com) account, with `pip install modal && modal setup`.
- A [Chroma Cloud](https://trychroma.com) database.
- A [Hugging Face](https://huggingface.co) token, for the pyannote diarisation
  models. **You must accept the model terms on the Hugging Face page for
  `pyannote/speaker-diarization` and `pyannote/segmentation` first**, or
  diarisation fails at runtime with an authorisation error rather than at setup.

## Setup

Create the Modal secret the apps expect. It is named `podcast-secrets` and every
value is read from the environment, so nothing sensitive is ever in the source:

```bash
modal secret create podcast-secrets \
  CHROMA_API_KEY=<your chroma api key> \
  CHROMA_TENANT=<your chroma tenant uuid> \
  CHROMA_DATABASE=<your chroma database name> \
  CHROMA_COLLECTION=<your collection name> \
  HF_TOKEN=<your huggingface token> \
  MCP_ALLOWED_HOST=<filled in after the first deploy, see below>
```

`CHROMA_COLLECTION` names the collection both apps use — the writer in
`transcribe.py` and the reader in `mcp_server.py`. It is a **shared** variable
on purpose: pointing the two halves at different collections is the failure it
exists to prevent, and a cutover is then a single edit rather than two edits
that must agree. It defaults to `podcast_transcripts` if unset.

`MCP_ALLOWED_HOST` is required and there is a deliberate chicken-and-egg to it:
it must equal the hostname clients actually connect to, which you only learn
once `modal deploy mcp_server.py` prints the endpoint URL. So deploy once, take
the hostname out of that URL, set the key, and deploy again.

**Get it wrong and every authenticated request returns `421 invalid host
header`**, which reads like a client bug rather than a config one. The MCP
transport uses this value for DNS-rebinding protection, so it is not merely a
bind address.

> ⚠️ `modal secret create --force` **overwrites the whole secret rather than
> merging into it**, and Modal never lets you read a secret's values back. So to
> add one key later you must re-supply every existing key in the same command,
> or you will silently destroy the ones you left out. Prefer a second, separate
> secret over amending this one.

Then deploy:

```bash
modal deploy transcribe.py
modal deploy mcp_server.py
```

`modal deploy mcp_server.py` prints the endpoint URL. Note it down; that is what
clients connect to.

### Region note

`chromadb.CloudClient` defaults to `aws-us-east-1`. The region is selected purely
by the presence or absence of a `cloud_host` argument, so to target a different
region you pass `cloud_host=<region host>` explicitly. There is no in-place
region change in Chroma Cloud; see `migration/` if you need to move one.

### Cutover

Moving to a new collection — after a migration, say — means changing
`CHROMA_COLLECTION` in the secret and redeploying both apps. Three things about
that are worth knowing before you do it.

**A wrong name fails loudly.** The reader calls `get_collection`, not
`get_or_create_collection`. A typo raises instead of silently creating an empty
collection and serving zero results, which is the same symptom as a broken
query and considerably harder to diagnose.

**There is no in-band version tell.** An earlier version of this section said
`n_chunks` in a search result was one. It is not: the field is put into the
searcher's result dict and never rendered into the string a caller receives, so
no consumer can see it. What tells you which collection is serving is the
`Reading from collection: <name>` line `load()` prints — authoritative once you
have cycled the app, per the next paragraph, since that leaves one generation of
containers.

**A warm container must be cycled, not waited out.** A container that entered
`load()` before the cutover holds a `Collection` handle bound to the old
collection. That handle has no name to re-resolve, so nothing raises and
`get_collection` does not help: it keeps serving the old data correctly and
silently, for as long as the container lives. Stop the app (`modal app stop
podcast-mcp-server`) and redeploy rather than assuming traffic has drained.

## Authentication

The MCP endpoint is protected with **Modal proxy auth**, so unauthenticated
requests are rejected at Modal's edge before a container starts. That matters
here beyond security: the search app holds a GPU, so an unauthenticated endpoint
would let anyone trigger a paid cold start.

Create a token pair. **Which route you take depends on your client version**, and
the two halves of this feature shipped separately: `requires_proxy_auth` has been
supported on `asgi_app` since at least 1.4.x, but the token-management CLI
arrived later. On 1.4.x, `modal workspace` does not exist.

```bash
# modal >= 1.5
modal workspace proxy-tokens create

# modal 1.4.x and earlier: no `workspace` command. Create the token in the
# dashboard instead (Settings, proxy auth tokens). Tokens are server-side, so
# this works regardless of client version.
modal dashboard
```

Either route returns a `wk-…` token ID and a `ws-…` secret. **The secret is
shown only once and cannot be retrieved afterwards**, so capture it immediately.

Then send its ID and secret on every request:

```bash
curl -H "Modal-Key: $TOKEN_ID" \
     -H "Modal-Secret: $TOKEN_SECRET" \
     -H "Content-Type: application/json" \
     -H "Accept: application/json, text/event-stream" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
          "params":{"name":"search_podcasts","arguments":{"query":"Taiwan"}}}' \
     https://<your-endpoint>.modal.run/mcp
```

The `Accept` header is not optional. MCP's streamable HTTP transport requires the
client to advertise both `application/json` and `text/event-stream`; omitting it
returns `406 Not Acceptable`.

## Usage

Transcribe a single feed:

```bash
modal run transcribe.py --feed-url "https://example.com/feed.xml" --show-name "Some Show"
```

Backfill an already-transcribed show into Chroma:

```bash
modal run transcribe.py::bulk_upload --show-name "Some Show"
```

The daily cron lives in `scheduled_job` in `transcribe.py`, running at 09:00 UTC.
Edit the feed list there to follow your own shows.

## Tools the MCP server exposes

- **`search_podcasts(query, n_results, show, after_date, before_date)`** —
  semantic search over every transcript chunk. Returns the text with speaker
  attribution, show, episode number and title, timestamp, and a relevance score.
  Results are sorted most-recent first so recency is always visible.
- **`latest_on_topic(topic, n_results)`** — the most recent things said about a
  topic, for when you want the current view rather than the best match.

## Cost

Both apps scale to zero. You pay for GPU seconds during transcription and during
search cold starts, plus Chroma Cloud storage. Transcription is the expensive
part and it is the part that runs on a schedule rather than on demand, so the
steady-state cost of *querying* is small. The proxy auth above is part of keeping
it that way.

## Licence

MIT. See [LICENSE](LICENSE).
