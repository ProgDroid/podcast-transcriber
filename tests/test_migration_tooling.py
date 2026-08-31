from migration.chroma_migrate import validate_collection

from tests.conftest import FakeCollection


def _seed(col, ids, meta_extra=None):
    for _id in ids:
        col.upsert(
            ids=[_id],
            documents=[f"doc-{_id}"],
            metadatas=[dict({"show": "S"}, **(meta_extra or {}))],
        )


def test_validation_follows_the_id_map():
    src, dst = FakeCollection(), FakeCollection()
    _seed(src, ["S-ep1-0", "S-ep1-1"])
    id_map = {"S-ep1-0": "S-ep1-2025-01-01-0", "S-ep1-1": "S-ep1-2025-01-01-1"}
    for old, new in id_map.items():
        dst.upsert(ids=[new], documents=[f"doc-{old}"], metadatas=[{"show": "S"}])
    assert validate_collection(src, dst, id_map=id_map) == []


def test_validation_without_an_id_map_reports_every_record_missing():
    src, dst = FakeCollection(), FakeCollection()
    _seed(src, ["S-ep1-0"])
    dst.upsert(
        ids=["S-ep1-2025-01-01-0"], documents=["doc-S-ep1-0"], metadatas=[{"show": "S"}]
    )
    problems = validate_collection(src, dst)
    assert any("missing id in dst" in p for p in problems)


def test_declared_new_keys_are_allowed_but_others_are_not():
    src, dst = FakeCollection(), FakeCollection()
    _seed(src, ["S-ep1-0"])
    id_map = {"S-ep1-0": "S-ep1-2025-01-01-0"}
    dst.upsert(
        ids=["S-ep1-2025-01-01-0"],
        documents=["doc-S-ep1-0"],
        metadatas=[{"show": "S", "n_chunks": 1, "rules_version": "1"}],
    )
    assert (
        validate_collection(
            src,
            dst,
            id_map=id_map,
            allowed_new_keys=frozenset({"n_chunks", "rules_version"}),
        )
        == []
    )
    problems = validate_collection(src, dst, id_map=id_map)
    assert any("metadata mismatch" in p for p in problems)


def test_changing_an_existing_key_is_still_a_mismatch():
    src, dst = FakeCollection(), FakeCollection()
    _seed(src, ["S-ep1-0"])
    dst.upsert(
        ids=["S-ep1-2025-01-01-0"],
        documents=["doc-S-ep1-0"],
        metadatas=[{"show": "DIFFERENT", "n_chunks": 1}],
    )
    problems = validate_collection(
        src,
        dst,
        id_map={"S-ep1-0": "S-ep1-2025-01-01-0"},
        allowed_new_keys=frozenset({"n_chunks"}),
    )
    assert any("metadata mismatch" in p for p in problems)
