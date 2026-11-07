# Self-hosted Chroma: measured behaviour

Measured 2026-11-07 against `chromadb/chroma:1.5.9` in local Docker (it reports
API version `1.0.0`), driven by the repo's pinned client `chromadb==1.5.9`.
These facts were measured on throwaway collections with synthetic data, not
on the production corpus. This is the counterpart to
[chroma-cloud-behaviour.md](chroma-cloud-behaviour.md), written for the move
off Chroma Cloud. Chroma Cloud goes read-only on 2026-11-22 and stops serving
reads on 2026-12-22.

## Same as Cloud

| Behaviour | Result |
|---|---|
| Upsert that **omits** a metadata key | key survives (merge), same as Cloud |
| `$eq` on a key some records lack | matches only records carrying the key |
| `$ne` on a key some records lack | **includes** records lacking the key |
| Paged `get` under the `episode_where` `$and`-of-`$eq` shape | 600/600 unique ids, identical order across two sweeps |
| `delete(where=...)` | removes 400 in one call |

## Different from Cloud

- **No 300-record request cap.** `get(limit=301)`, an unlimited `get()`
  (it returned all 400), a 301-record upsert, a 400-record add and a
  301-record `delete(ids=...)` all succeed. The repo's paging at 250 stays
  correct and harmless. `FakeCollection` enforcing the Cloud cap is now
  stricter than the backend, which is the safe direction.
- **A `None` metadata value DELETES the key** on upsert. It is accepted, not
  rejected. `corpus/chunking.py` used to say "Chroma rejects a None metadata
  value"; that comment (and its copy in `tests/test_chunking.py`) was
  corrected on 2026-11-07. Its behaviour was always right: omit the key,
  never pass `None`, which is exactly what keeps merges safe. Never pass
  `None` to mean "leave unchanged".
- **No built-in authentication.** The server takes legacy
  `CHROMA_SERVER_AUTHN_PROVIDER` / `CHROMA_SERVER_AUTHN_CREDENTIALS`
  silently and enforces nothing. With them set, requests with no token, with
  a wrong token, and an anonymous collection create all returned HTTP 200.
  Chroma's docs agree: v1.0.0+ ships no auth implementation. **Any internet-
  reachable deployment needs a token-checking reverse proxy in front.**

## Sizing (29,000 records, 1024-d, ~400-word documents, 11 metadata keys)

| Measurement | Value |
|---|---|
| Idle, empty | 65 MiB |
| Steady state after restart + queries | 157-182 MiB |
| Peak during a full 29k bulk load, uncapped | ~505-545 MiB (sampled) |
| Peak during a full 29k bulk load, capped at 512 MiB | 398 MiB sampled, no OOM kill |
| Disk (`/data`) | 517 MB |
| Bulk load time (batches of 250) | 63 s uncapped, 87 s capped |
| Restart to heartbeat | ~1.0-1.2 s |
| Query latency (n=10) | 7 ms unfiltered; 25-35 ms single `$eq`; ~90-105 ms `$and` + `date_ts` range |

The first query after a restart was as fast as a warm one, so a cold start
costs ~1 s of boot plus the host's own machine start.

Sampling was `docker stats` at roughly 2 s intervals, so the true bulk-load
peak may sit above the sampled one. A 512 MB Fly machine also gives up some
memory to its own kernel and init. **Run the migration's bulk copy on a
temporarily larger machine (1 GB) and scale down afterwards.** Steady-state
nightly ingest fits 512 MB comfortably.

## Schema and distance space (measured on the real Cloud schema)

- Chroma Cloud stores the space in the collection **schema**
  (`keys.#embedding.float_list.vector_index.config.space = cosine`, with a
  SPANN index), and `collection.metadata` is `None`.
- That exact schema, passed to `create_collection(schema=...)` on the OSS
  server, is **accepted**: the result is cosine with an HNSW index. So
  `migration/chroma_migrate.py::create_dest_collection`'s schema branch
  works, and `validate_collection`'s "schema mismatch" (spann vs hnsw) is
  expected, not a failure.
- `create_collection(metadata={"hnsw:space": "cosine"})` also gives cosine.
- `create_collection` with **no** config gives **l2**, silently. Distances
  change scale (orthogonal unit vectors: 2.0 instead of 1.0) while rankings
  look right. See [backups.md](backups.md) for the restore rule this produced.

The real corpus (30,483 records) restored from the first export into a local
1.5.9 server reproduces exact cosine search; see [backups.md](backups.md).

## The Fly deployment and the copy from Cloud (2026-11-07)

`deploy/chroma/` is live on Fly (1 GB machine, auto-stop, `iad`).

Measured against the live app:
- heartbeat 200 without a token
- no token, a wrong token, or an empty `Bearer ` all get 401; the right token gets 200
- plain http redirects 301 to https
- addresses: one shared IPv4 and the free IPv6, no dedicated IPv4
- the machine was stopped between requests, so auto-stop engaged
- `/api/v2/version` returns `"1.0.0"`. The same pinned 1.5.9 image does that locally too: it is the API version, not the image tag.

The copy ran on Modal (`migration/modal_run.py`), because the only live Cloud key is in `podcast-secrets`:
- smoke: source `podcast_transcripts_v2` had 30,483 records; the destination was empty
- copy: 122 batches, 30,483/30,483; the source count was unchanged at the end, so the freeze held
- validate: PASS with 0 problems over 30,483 records (document, metadata, embedding at atol 1e-4)
  - the schema differs by backend (SPANN becomes HNSW), but the distance space matches: cosine on both
  - the self-query recovered its record

Between backends `validate.py` uses `schema_policy="space"`: only the distance space has to match, and a space it cannot read counts as a failure. The self-query alone cannot tell l2 from cosine.

## Cutover facts (measured 2026-11-07)

- On an existing collection, `get_or_create_collection(name, metadata=...)` ignores `metadata` entirely. It does not raise, and the schema and space are unchanged, even when the metadata says `l2` against a cosine collection. Measured on a local 1.5.9 server with a collection built from the Fly collection's real schema. So the writers' `metadata={"hnsw:space": "cosine"}` matters only when the collection is brand new, where it prevents l2.
- Auto-stop: Fly's proxy stopped the machine about 6.5 minutes after the last request (start at 20:20:17, stop at 20:26:51, from machine events). A cold start plus a count took 4.3 s from a local client.
- **Refused requests also wake the machine**, because the token check runs inside it (Caddy). Only a request at least every ~6 minutes around the clock would keep it always on (~$3.69/mo at 512 MB). That is why the app's name is kept out of this public repo: `fly.toml` has no `app =` line.
