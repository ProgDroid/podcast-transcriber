import chromadb
from migration.chroma_migrate import (
    _self_query_ok,
    create_dest_collection,
    get_collection_readonly,
    validate_collection,
)

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


def test_allowed_new_keys_does_not_license_a_changed_value_on_that_same_key():
    # allowed_new_keys whitelists keys the DESTINATION may carry that the
    # SOURCE lacks (extra keys). It must not also license a changed VALUE on
    # a key present in both sides, even when that key's name happens to be
    # in allowed_new_keys. Declaring "show" allowed must not paper over
    # src show=S vs dst show=DIFFERENT.
    src, dst = FakeCollection(), FakeCollection()
    _seed(src, ["S-ep1-0"])
    dst.upsert(
        ids=["S-ep1-0"],
        documents=["doc-S-ep1-0"],
        metadatas=[{"show": "DIFFERENT"}],
    )
    problems = validate_collection(src, dst, allowed_new_keys=frozenset({"show"}))
    assert any("metadata mismatch" in p for p in problems)


def test_self_query_ok_maps_expected_id_through_id_map():
    assert _self_query_ok("src-1", ["dst-1", "dst-2"], id_map={"src-1": "dst-1"})


def test_self_query_ok_without_id_map_falls_back_to_bare_id():
    assert _self_query_ok("id-1", ["id-1", "id-2"])


def test_self_query_ok_false_when_id_absent_from_hits():
    assert not _self_query_ok("id-1", ["id-2", "id-3"])


def test_self_query_ok_id_missing_from_id_map_falls_back_to_bare_id():
    assert _self_query_ok("id-1", ["id-1"], id_map={"other": "mapped-other"})


def test_create_dest_collection_dest_name_does_a_same_database_re_id(tmp_path):
    # Rehearses the SAME-DATABASE re-ID the cutover will actually perform:
    # one client, source and destination both live in it, destination
    # created under an explicit dest_name that differs from the source's.
    # migrate.py/selftest.py only ever exercise DB-to-DB copies (two
    # separate clients), which never touches this branch.
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection(name="src-col", metadata={"hnsw:space": "cosine"})
    src_col = get_collection_readonly(client, "src-col")
    src_col.add(
        ids=["a"],
        embeddings=[[0.1, 0.2]],
        documents=["doc-a"],
        metadatas=[{"show": "S"}],
    )

    dst_col = create_dest_collection(client, src_col, dest_name="dst-col")

    names = {c.name for c in client.list_collections()}
    assert names == {"src-col", "dst-col"}
    assert dst_col.name == "dst-col"
    assert dst_col.count() == 0
    assert src_col.count() == 1
