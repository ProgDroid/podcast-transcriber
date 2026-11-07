# Backups

The corpus has one live copy: the Chroma collection named by
`CHROMA_COLLECTION` in the `podcast-secrets` Modal secret. `backup.py`
exports it to the `chroma-backups` Modal Volume, a different provider from
wherever Chroma runs.

It can also be regenerated from the transcripts and audio, given time and
Modal spend. The backup exists so that a lost database costs a restore, not a
re-run of transcription, embedding and speaker identification.

## Taking one

```bash
uv run modal run backup.py            # export, verify, print a summary
uv run modal run backup.py --keep 8   # same, then prune to the newest 8
```

Run it when nothing is writing; the nightly cron is 09:00 UTC. A write
during the export is detected (count before vs after) and refused, never
silently captured half-way. Each export is checked from disk
(`verify_export`) before it is moved into place, so a directory without the
`.partial-` prefix is a verified export.

Layout: `/backups/<collection>/<UTC stamp>/` holding

| File | Content |
|---|---|
| `records.jsonl.gz` | one JSON object per record: `id`, `document`, `metadata` |
| `embeddings.npy` | float32, row *i* belongs to line *i* of the records file |
| `manifest.json` | count, dimension, distance space, source schema, sha256 of both files |

Plain JSON and `.npy` so a backup is readable without this repo or chromadb.

## Restoring

```python
from corpus.backup import distance_space, restore_into
import json, chromadb
m = json.load(open(f"{d}/manifest.json"))
col = client.create_collection(m["collection"],
                               metadata={"hnsw:space": distance_space(m)})
restore_into(col, d)   # refuses a non-empty collection; verifies first
```

**Always create the target with the manifest's space.** Chroma Cloud keeps
the space in the schema and reports `collection.metadata` as `None`. A
collection created with no config silently gets `l2`: rankings are the
same for normalized vectors, but every reported distance changes scale. This
happened in the first restore drill. `distance_space` raises rather than
guess.

To fetch an export, download each file separately:
`MSYS_NO_PATHCONV=1 uv run modal volume get chroma-backups <collection>/<stamp>/<file> <file>`.
Fetching the directory into a path that doesn't exist yet wrote all three
files over one another into a single file, leaving only the last.

## First export (2026-11-07, from Chroma Cloud)

- `podcast_transcripts_v2/2026-11-07T1753Z`: **30,483 records** (30,292
  podcast, 191 book), dimension 1024, space `cosine`. 23,469 records carry
  `speaker_id` and 29,967 carry `episode_guid`.
- Sizes: `embeddings.npy` 124,858,496 B (exactly 30,483 x 1024 x 4 + 128-byte
  header), `records.jsonl.gz` 11,154,414 B.
- Downloaded and verified locally (sha256, counts, unique ids, finite unit-norm
  embeddings), then restored into a local `chromadb/chroma:1.5.9`: 30,483
  restored in 65 s. 50/50 sampled self-queries matched id, document and
  metadata exactly. 30 perturbed queries had top-10 recall 1.000 against exact
  numpy cosine search, with reported distances within 5e-7 of `1 - cos`.

## Still to do

- Point `backup.py` at the self-hosted Chroma at the cutover.
- Schedule it weekly, then (`modal.Cron`, outside the 09:00 UTC ingest window).
