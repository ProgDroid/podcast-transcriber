"""Export the live Chroma collection to the `chroma-backups` Modal Volume.

    modal deploy backup.py            # weekly export (Sun 10:00 UTC), keeps 4
    modal run backup.py               # export once now, print the manifest
    modal run backup.py --keep 8      # also prune to the newest 8 exports

READ-ONLY against Chroma: it calls get_collection (never get_or_create), and
only `get`/`count`. Each export lands in
/backups/<collection>/<UTC stamp>/ only after corpus.backup.verify_export has
re-read it from disk; a failed run leaves at most a `.partial-*` directory,
which the next run removes.

Run it when nothing is writing (the nightly cron is 09:00 UTC). A write
during the export is detected and refused, not silently captured half-way.

Reads the self-hosted server (corpus/chroma_client.py, `chroma-selfhosted`
secret). Exports taken before 2026-11 came from Chroma Cloud; their manifests
say so in `source.backend`, and they restore the same way.
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb==1.5.9")
    .add_local_python_source("corpus")
)

app = modal.App("podcast-chroma-backup", image=image)
backups = modal.Volume.from_name("chroma-backups", create_if_missing=True)
ROOT = "/backups"


# Weekly, an hour after the nightly ingest (09:00 UTC), so a backup never
# races a writer; export refuses if the count moves mid-read anyway.
WEEKLY = modal.Cron("0 10 * * 0")
# Scheduled runs keep the newest 4 complete exports (~125 MB each at 30k
# records). Manual `modal run backup.py` prunes nothing unless --keep is given.
KEEP_WEEKLY = 4


@app.function(
    secrets=[
        modal.Secret.from_name("podcast-secrets"),
        modal.Secret.from_name("chroma-selfhosted"),
    ],
    volumes={ROOT: backups},
    timeout=3600,
    memory=2048,
    schedule=WEEKLY,
)
def export(keep: int = KEEP_WEEKLY) -> dict:
    import os
    import shutil
    from datetime import UTC, datetime
    from pathlib import Path

    import chromadb

    from corpus.backup import MANIFEST, export_collection, verify_export
    from corpus.chroma_client import make_client

    name = os.environ.get("CHROMA_COLLECTION", "podcast_transcripts")
    client = make_client()
    collection = client.get_collection(name=name)

    base = Path(ROOT) / name
    base.mkdir(parents=True, exist_ok=True)
    for stale in base.glob(".partial-*"):
        shutil.rmtree(stale)

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%MZ")
    partial = base / f".partial-{stamp}"
    final = base / stamp

    try:
        schema = collection.schema.serialize_to_json()
    except Exception as e:  # noqa: BLE001 - best effort, recorded either way
        schema = f"unavailable: {type(e).__name__}: {e}"

    manifest = export_collection(
        collection,
        partial,
        source={
            "backend": "chroma-selfhosted",
            "url": os.environ["CHROMA_URL"],
            "chromadb_client": chromadb.__version__,
            "schema": schema,
        },
    )
    verify_export(partial)
    partial.rename(final)

    pruned = []
    if keep > 0:
        complete = sorted(p for p in base.iterdir() if (p / MANIFEST).exists())
        for old in complete[:-keep]:
            shutil.rmtree(old)
            pruned.append(old.name)

    backups.commit()
    sizes = {p.name: p.stat().st_size for p in final.iterdir()}
    return {
        "path": str(final),
        "count": manifest["count"],
        "dimension": manifest["dimension"],
        "sizes_bytes": sizes,
        "pruned": pruned,
    }


@app.local_entrypoint()
def main(keep: int = 0) -> None:
    print(export.remote(keep=keep))
