"""Export a whole collection to plain files, verify them, and restore them.

Written against a duck-typed collection (`get`, `count`, `upsert`), like the
rest of corpus/, so it is testable against tests/conftest.py's fake and works
unchanged against Chroma Cloud or a self-hosted Chroma.

An export is a directory of three files:

  records.jsonl.gz  one JSON object per record: id, document, metadata
  embeddings.npy    float32 array, row i belongs to line i of records
  manifest.json     count, dimension, source, sha256 of the two files above

Plain JSON and .npy on purpose: a backup must be readable without this
code, without chromadb, and years from now.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from corpus.store import BATCH, PAGE, batched

FORMAT_VERSION = 1
RECORDS = "records.jsonl.gz"
EMBEDDINGS = "embeddings.npy"
MANIFEST = "manifest.json"


class ExportError(Exception):
    """The export or its verification found something that must not ship."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_all(collection) -> tuple[list[str], list[Any], list[Any], np.ndarray]:
    """Page every record. The unpaged get silently stops at 300 on Cloud."""
    ids: list[str] = []
    docs: list[Any] = []
    metas: list[Any] = []
    pages: list[np.ndarray] = []
    offset = 0
    while True:
        page = collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=PAGE,
            offset=offset,
        )
        n = len(page["ids"])
        if n == 0:
            break
        ids.extend(page["ids"])
        docs.extend(page["documents"])
        metas.extend(page["metadatas"])
        emb = np.asarray(page["embeddings"], dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != n:
            raise ExportError(f"page at offset {offset}: embeddings shape {emb.shape}")
        pages.append(emb)
        offset += n
    embs = np.concatenate(pages) if pages else np.zeros((0, 0), dtype=np.float32)
    return ids, docs, metas, embs


def export_collection(collection, out_dir: Path, *, source: dict) -> dict:
    """Write every record of `collection` into `out_dir` and return the manifest.

    Refuses (raises ExportError, after writing nothing it vouches for) when
    the collection changed while it was being read, or when paging returned
    duplicates or misses. Either would produce a backup that looks complete
    and is not.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count_before = collection.count()
    ids, docs, metas, embs = _read_all(collection)
    count_after = collection.count()

    if count_before != count_after:
        raise ExportError(
            f"collection changed during export ({count_before} -> {count_after} "
            f"records); re-run when nothing is writing"
        )
    if len(ids) != count_after:
        raise ExportError(
            f"paged {len(ids)} records but the collection holds {count_after}"
        )
    if len(set(ids)) != len(ids):
        raise ExportError(f"paging returned {len(ids) - len(set(ids))} duplicate ids")
    dims = {embs.shape[1]} if len(ids) else set()
    if len(dims) > 1:
        raise ExportError(f"mixed embedding dimensions: {dims}")

    with gzip.open(out_dir / RECORDS, "wt", encoding="utf-8") as fh:
        for _id, doc, meta in zip(ids, docs, metas, strict=True):
            fh.write(
                json.dumps(
                    {"id": _id, "document": doc, "metadata": meta}, ensure_ascii=False
                )
                + "\n"
            )
    np.save(out_dir / EMBEDDINGS, embs, allow_pickle=False)

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "collection": collection.name,
        "collection_metadata": collection.metadata,
        "source": source,
    }
    manifest["space"] = distance_space(manifest)
    manifest |= {
        "count": len(ids),
        "dimension": int(embs.shape[1]) if len(ids) else 0,
        "sha256": {
            RECORDS: _sha256(out_dir / RECORDS),
            EMBEDDINGS: _sha256(out_dir / EMBEDDINGS),
        },
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def distance_space(manifest: dict) -> str:
    """The distance space a restore target must be created with.

    Chroma Cloud keeps the space in the collection SCHEMA and reports
    `collection.metadata` as None; a self-hosted collection created from
    metadata keeps it in `hnsw:space`. Measured 2026-11-07: creating a
    collection with neither gives l2, silently, so this refuses rather than
    defaults. Create the target with metadata={"hnsw:space": <this>}.
    """
    if manifest.get("space"):
        return manifest["space"]
    schema = (manifest.get("source") or {}).get("schema")
    if isinstance(schema, dict):
        try:
            config = schema["keys"]["#embedding"]["float_list"]["vector_index"]
            if config["config"].get("space"):
                return config["config"]["space"]
        except (KeyError, TypeError):
            pass
    meta = manifest.get("collection_metadata") or {}
    if meta.get("hnsw:space"):
        return meta["hnsw:space"]
    raise ExportError("cannot determine the collection's distance space")


def load_export(out_dir: Path) -> tuple[list[str], list[Any], list[Any], np.ndarray]:
    """Read an export back: ids, documents, metadatas, embeddings (row-aligned)."""
    out_dir = Path(out_dir)
    ids: list[str] = []
    docs: list[Any] = []
    metas: list[Any] = []
    with gzip.open(out_dir / RECORDS, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            ids.append(rec["id"])
            docs.append(rec["document"])
            metas.append(rec["metadata"])
    embs = np.load(out_dir / EMBEDDINGS, allow_pickle=False)
    return ids, docs, metas, embs


def verify_export(out_dir: Path) -> dict:
    """Re-read an export from disk and check it against its own manifest."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / MANIFEST).read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        actual = _sha256(out_dir / name)
        if actual != expected:
            raise ExportError(f"{name}: sha256 {actual} != manifest {expected}")
    ids, _, _, embs = load_export(out_dir)
    n = manifest["count"]
    if len(ids) != n or len(set(ids)) != n:
        raise ExportError(
            f"records: {len(ids)} rows / {len(set(ids))} unique, manifest {n}"
        )
    if n and embs.shape != (n, manifest["dimension"]):
        raise ExportError(
            f"embeddings shape {embs.shape} != ({n}, {manifest['dimension']})"
        )
    if not np.isfinite(embs).all():
        raise ExportError("embeddings contain NaN or inf")
    return manifest


def restore_into(collection, out_dir: Path) -> int:
    """Load a verified export into an EMPTY collection. Returns records written.

    Refuses a non-empty destination: upsert merges metadata, so restoring
    over existing records would silently keep keys the backup does not have.
    """
    if collection.count():
        raise ExportError(
            f"destination {collection.name!r} is not empty "
            f"({collection.count()} records)"
        )
    verify_export(out_dir)
    ids, docs, metas, embs = load_export(out_dir)
    for batch in batched(list(range(len(ids))), BATCH):
        collection.upsert(
            ids=[ids[i] for i in batch],
            embeddings=embs[batch[0] : batch[-1] + 1],
            documents=[docs[i] for i in batch],
            # Chroma rejects an empty metadata dict; None means "no metadata".
            metadatas=[metas[i] or None for i in batch],
        )
    if collection.count() != len(ids):
        raise ExportError(f"restored {collection.count()} of {len(ids)} records")
    return len(ids)
