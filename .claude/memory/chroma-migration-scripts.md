---
name: chroma-migration-scripts
description: "Reusable env-driven Chroma DB-to-DB copy+validate scripts in migration/, plus chromadb 1.5.4 and Modal gotchas"
metadata: 
  node_type: memory
  type: reference
  originSessionId: d5442bb9-4860-4afe-86c0-a2c2303e6860
  modified: 2026-11-07T18:04:31.311Z
---

Reusable Chroma Cloud DB→DB migration tooling lives in `G:\audio-transcription\via-modal\migration\`:
- `chroma_migrate.py` — shared logic (create dest from copied schema, paged copy, full validation).
- `migrate.py` / `validate.py` — env-driven CLIs (`SRC_KEY/TENANT/DB/HOST`, `DST_KEY/TENANT/DB[/HOST]`;
  `SMOKE_ONLY=1` for a read-only creds/empty-dest check; `ALLOW_NONEMPTY_DST=1` to resume).
- `selftest.py` — dress-rehearses copy+validate against two local `PersistentClient` dirs (>300 records).
- Copies embeddings verbatim (`.tolist()`), idempotent upserts, re-checks source count at end to prove the
  freeze held, then validates counts/schema/per-record docs+meta+uris+embeddings (`allclose` 1e-4) + self-query.

Non-obvious gotchas hit and worth remembering (validated against `chromadb==1.5.4`):
- **`create_collection` rejects `schema=` and `metadata=` together** → `Cannot set both collection config and
  schema simultaneously`. The deserialized schema already carries the distance-space config; pass **schema only**.
  A metadata-built collection also serializes to a *different* schema than a schema-copied one (validation catches it).
- Chroma Cloud **`get()` caps at 300 records/request** — page at ≤250.
- **Default region is `aws-us-east-1`**; other regions require `cloud_host` (or the `CHROMA_HOST` env var).
  `CloudClient` has no `region` arg — region == which host you hit.
- Chroma rejects an **all-`None` column** — pass `None` for the whole arg, not `[None, …]`.
- **`modal app stop <app> --yes`** is required in non-interactive shells (this session's PowerShell/Bash tools);
  without `--yes` it aborts on the confirm prompt.

- **`modal volume get <vol> <dir> <local>` into a local path that does not exist yet** wrote every file to that
  one path, each overwriting the last, so only the final file survived. It exited 0 and printed "Finished
  downloading". Fetch each file to its own name (measured 2026-11-07, see `docs/backups.md`).
- **Self-hosted Chroma 1.5.9 differs from Cloud**: it has no request cap, no built-in auth, a `None` metadata
  value deletes the key, and a collection created with no config is **l2**. Cloud keeps the space in the schema,
  not in metadata. See `docs/self-hosted-chroma-behaviour.md`; `corpus/backup.py::distance_space` reads it.

Used 2026-07-09 for the EU→US move — see [[chroma-us-region-migration]].
- **`migrate.py`/`validate.py` take either end as self-hosted** (`<P>_URL` + `<P>_TOKEN`); `modal_run.py` runs them on Modal against the live Cloud key. Across backends, validate uses `schema_policy="space"`, because Cloud's SPANN becomes HNSW and the exact schemas never match. On 1.5.9, `get_or_create_collection` on an EXISTING collection ignores `metadata`, even a conflicting space (measured 2026-11-07, `docs/self-hosted-chroma-behaviour.md`).
