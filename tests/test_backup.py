"""Tests for corpus/backup.py: export a whole collection to files and back."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from corpus.backup import (
    ExportError,
    distance_space,
    export_collection,
    load_export,
    restore_into,
    verify_export,
)
from tests.conftest import MAX_REQUEST, FakeCollection

DIM = 8


class EmbeddingFake(FakeCollection):
    """FakeCollection that also keeps embeddings and returns them on include.

    Real chromadb 1.5.9 returns `embeddings` as a 2-D numpy array, which is
    what this returns too, so the exporter is exercised on the real shape.
    """

    def __init__(self, name: str = "podcast_transcripts_v2") -> None:
        super().__init__(name)
        self._emb: dict[str, list[float]] = {}

    def upsert(self, ids, embeddings=None, documents=None, metadatas=None):
        super().upsert(ids, embeddings, documents, metadatas)
        if embeddings is not None:
            for k, _id in enumerate(ids):
                self._emb[_id] = [float(x) for x in embeddings[k]]

    def get(self, ids=None, where=None, limit=None, offset=0, include=None):
        include = list(include or [])
        out = super().get(ids, where, limit, offset, include)
        if "embeddings" in include:
            out["embeddings"] = np.array(
                [self._emb[i] for i in out["ids"]], dtype=np.float32
            ).reshape(len(out["ids"]), DIM)
        return out


def seed(col: EmbeddingFake, n: int) -> None:
    rng = np.random.default_rng(0)
    for start in range(0, n, 250):
        ids = [f"Show-1-2026-01-01-{i}" for i in range(start, min(n, start + 250))]
        col.upsert(
            ids=ids,
            embeddings=rng.standard_normal((len(ids), DIM)).astype(np.float32),
            documents=[f"[A] text {i}" for i in ids],
            metadatas=[
                {"show": "Show", "date_ts": 20260101, "start_time": 1.5, "i": k}
                for k, _ in enumerate(ids)
            ],
        )


SOURCE = {"backend": "test", "database": "db"}


def test_round_trip_pages_past_the_request_cap(tmp_path):
    col = EmbeddingFake()
    seed(col, 700)  # > MAX_REQUEST: an unpaged get would see only 300

    manifest = export_collection(col, tmp_path, source=SOURCE)
    assert manifest["count"] == 700
    assert manifest["collection"] == "podcast_transcripts_v2"
    assert manifest["dimension"] == DIM
    assert verify_export(tmp_path)["count"] == 700

    ids, docs, metas, embs = load_export(tmp_path)
    assert len(set(ids)) == 700
    for j in (0, 299, 300, 699):
        _id = ids[j]
        assert docs[j] == col._docs[_id]
        assert metas[j] == col._meta[_id]
        assert np.array_equal(embs[j], np.float32(col._emb[_id]))


def test_metadata_types_survive(tmp_path):
    col = EmbeddingFake()
    seed(col, 3)
    export_collection(col, tmp_path, source=SOURCE)
    _, _, metas, _ = load_export(tmp_path)
    m = metas[0]
    assert isinstance(m["date_ts"], int) and not isinstance(m["date_ts"], bool)
    assert isinstance(m["start_time"], float)


def test_no_request_exceeds_the_cap(tmp_path):
    col = EmbeddingFake()
    seed(col, 400)
    seen = []
    real_get = col.get

    def spy(*a, **kw):
        seen.append(kw.get("limit"))
        return real_get(*a, **kw)

    col.get = spy
    export_collection(col, tmp_path, source=SOURCE)
    assert seen and all(lim is not None and lim <= MAX_REQUEST for lim in seen)


def test_a_write_during_export_is_refused(tmp_path):
    col = EmbeddingFake()
    seed(col, 400)
    real_get = col.get
    state = {"done": False}

    def get_then_write(*a, **kw):
        out = real_get(*a, **kw)
        if not state["done"]:
            state["done"] = True
            col.upsert(
                ids=["late-arrival"],
                embeddings=np.zeros((1, DIM), dtype=np.float32),
                documents=["late"],
                metadatas=[{"show": "Show"}],
            )
        return out

    col.get = get_then_write
    with pytest.raises(ExportError, match="changed during export"):
        export_collection(col, tmp_path, source=SOURCE)


def test_unstable_paging_is_refused(tmp_path):
    col = EmbeddingFake()
    seed(col, 400)
    real_get = col.get

    def repeat_first_page(*a, **kw):
        kw = dict(kw)
        if kw.get("offset", 0) == 250:
            kw["offset"] = 0  # serve page one twice: duplicates, and a miss
        return real_get(*a, **kw)

    col.get = repeat_first_page
    with pytest.raises(ExportError):
        export_collection(col, tmp_path, source=SOURCE)


def test_one_duplicate_plus_one_miss_is_refused(tmp_path):
    """The count-preserving shape: the total matches, so only the id check sees it."""
    col = EmbeddingFake()
    seed(col, 400)
    real_get = col.get

    def overlap_by_one(*a, **kw):
        if kw.get("offset", 0) != 250:
            return real_get(*a, **kw)
        # Same page length as the honest page, window shifted back by one:
        # record 249 is served twice and record 399 never.
        n = len(real_get(*a, **kw)["ids"])
        shifted = real_get(*a, **{**kw, "offset": 249})
        return {k: v[:n] for k, v in shifted.items()}

    col.get = overlap_by_one
    with pytest.raises(ExportError, match="duplicate"):
        export_collection(col, tmp_path, source=SOURCE)


def test_verify_catches_a_corrupted_file(tmp_path):
    col = EmbeddingFake()
    seed(col, 10)
    export_collection(col, tmp_path, source=SOURCE)
    with gzip.open(tmp_path / "records.jsonl.gz", "at", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "x", "document": "", "metadata": {}}) + "\n")
    with pytest.raises(ExportError, match="sha256"):
        verify_export(tmp_path)


def test_restore_reproduces_the_collection(tmp_path):
    src = EmbeddingFake()
    seed(src, 520)
    export_collection(src, tmp_path, source=SOURCE)

    dst = EmbeddingFake()
    assert restore_into(dst, tmp_path) == 520
    assert dst._docs == src._docs
    assert dst._meta == src._meta
    assert all(np.allclose(dst._emb[i], src._emb[i]) for i in src._emb)


# The shape Chroma Cloud actually records (trimmed from the 2026-11-07 export):
# the space lives in the schema, and collection.metadata is None.
CLOUD_SCHEMA = {
    "keys": {
        "#embedding": {
            "float_list": {
                "vector_index": {
                    "enabled": True,
                    "config": {"space": "cosine", "spann": {"ef_search": 200}},
                }
            }
        }
    }
}


def test_space_is_read_from_a_cloud_schema():
    manifest = {"collection_metadata": None, "source": {"schema": CLOUD_SCHEMA}}
    assert distance_space(manifest) == "cosine"


def test_space_falls_back_to_collection_metadata():
    manifest = {"collection_metadata": {"hnsw:space": "ip"}, "source": {}}
    assert distance_space(manifest) == "ip"


def test_space_prefers_the_recorded_field():
    manifest = {"space": "l2", "collection_metadata": {"hnsw:space": "cosine"}}
    assert distance_space(manifest) == "l2"


def test_an_unknown_space_is_refused_not_defaulted():
    # Creating a collection with no space silently gives l2: distances change
    # scale with no error. Guessing here would reproduce exactly that.
    with pytest.raises(ExportError, match="distance space"):
        distance_space({"collection_metadata": None, "source": {"schema": "n/a"}})


def test_export_records_the_space(tmp_path):
    col = EmbeddingFake()  # FakeCollection.metadata carries hnsw:space cosine
    seed(col, 3)
    manifest = export_collection(col, tmp_path, source=SOURCE)
    assert manifest["space"] == "cosine"


def test_export_records_the_space_from_a_schema(tmp_path):
    col = EmbeddingFake()
    col.metadata = None
    seed(col, 3)
    manifest = export_collection(
        col, tmp_path, source={**SOURCE, "schema": CLOUD_SCHEMA}
    )
    assert manifest["space"] == "cosine"


def test_restore_refuses_a_non_empty_destination(tmp_path):
    src = EmbeddingFake()
    seed(src, 5)
    export_collection(src, tmp_path, source=SOURCE)
    dst = EmbeddingFake()
    seed(dst, 1)
    with pytest.raises(ExportError, match="not empty"):
        restore_into(dst, tmp_path)
