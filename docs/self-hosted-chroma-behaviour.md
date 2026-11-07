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
