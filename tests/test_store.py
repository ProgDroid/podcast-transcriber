import pytest

from corpus.store import (
    MAX_REQUEST as STORE_MAX_REQUEST,
)
from corpus.store import (
    batched,
    episode_where,
    guid_where,
    is_complete,
    paged_get,
    paged_get_ids,
    stale_ids,
)
from tests.conftest import MAX_REQUEST, ChromaQuotaError

TRIPLE = ("Geopolitical Cousins", "73", "2026-07-29")


def _seed(collection, prefix, n, **meta):
    base = {
        "show": TRIPLE[0],
        "episode_number": TRIPLE[1],
        "date": TRIPLE[2],
        "n_chunks": n,
    }
    base.update(meta)
    for batch in batched(list(range(n))):
        collection.upsert(
            ids=[f"{prefix}-{i}" for i in batch],
            documents=[f"doc-{i}" for i in batch],
            metadatas=[dict(base) for _ in batch],
        )


def test_batched_never_exceeds_the_request_cap():
    chunks = list(batched(list(range(431))))
    assert [len(c) for c in chunks] == [250, 181]
    assert all(len(c) <= MAX_REQUEST for c in chunks)


def test_batched_handles_an_empty_list():
    assert list(batched([])) == []


def test_unbatched_upsert_of_a_real_episode_would_raise(collection):
    # Geopolitical Cousins 73 is 431 chunks. This is the production failure.
    with pytest.raises(ChromaQuotaError):
        collection.upsert(
            ids=[f"x-{i}" for i in range(431)],
            documents=["d"] * 431,
            metadatas=[{}] * 431,
        )


def test_batched_upsert_of_the_same_episode_succeeds(collection):
    _seed(collection, "gc73", 431)
    assert collection.count() == 431


def test_paged_get_returns_everything_past_the_300_cap(collection):
    _seed(collection, "gc73", 431)
    ids = paged_get_ids(collection, episode_where(*TRIPLE))
    assert len(ids) == 431


def test_unpaged_get_silently_truncates(collection):
    # The trap: this returns 300 and raises nothing.
    _seed(collection, "gc73", 431)
    assert len(collection.get(where=episode_where(*TRIPLE))["ids"]) == 300


def test_is_complete_uses_the_expected_count_not_mere_presence(collection):
    _seed(collection, "gc73", 431)
    ids = paged_get_ids(collection, episode_where(*TRIPLE))
    assert is_complete(ids, 431)
    assert not is_complete(ids, 432)


def test_a_single_orphan_chunk_is_not_complete():
    # After a collision clobber an episode retained one chunk. A boolean
    # "does any chunk exist" check called that healthy.
    assert not is_complete(["gc73-2"], 431)


def test_no_records_is_not_complete():
    assert not is_complete([], 431)


def test_absent_expected_count_is_not_complete():
    # Pre-migration records carry no n_chunks. Treating that as satisfied
    # would mean old episodes are never completeness-checked at all.
    assert not is_complete(["a-0"], None)


def test_stale_ids_is_the_set_difference():
    assert stale_ids(["a-0", "a-1", "a-2"], ["a-0", "a-1"]) == ["a-2"]


def test_stale_ids_is_empty_when_the_episode_grew():
    assert stale_ids(["a-0"], ["a-0", "a-1"]) == []


def test_stale_ids_spans_old_and_new_id_schemes():
    old = ["Show-ep1-0", "Show-ep1-1"]
    new = ["Show-ep1-2025-01-01-0"]
    assert sorted(stale_ids(old + new, new)) == old


def test_guid_where_and_episode_where_select_the_same_records(collection):
    _seed(collection, "gc73", 10, episode_guid="abc-123")
    by_triple = paged_get_ids(collection, episode_where(*TRIPLE))
    by_guid = paged_get_ids(collection, guid_where("abc-123"))
    assert by_triple == by_guid


def test_delete_with_a_non_matching_filter_is_a_no_op(collection):
    _seed(collection, "gc73", 10)
    collection.delete(where=episode_where("Other Show", "1", "2020-01-01"))
    assert collection.count() == 10


def test_upsert_on_existing_id_merges_metadata(collection):
    # Verified on Cloud: an upsert on an existing id replaces the document
    # but MERGES the metadata -- a key absent from the new write survives.
    collection.upsert(ids=["a-0"], documents=["v1"], metadatas=[{"a": 1, "b": 2}])
    collection.upsert(ids=["a-0"], documents=["v2"], metadatas=[{"a": 9}])
    result = collection.get(ids=["a-0"], include=["documents", "metadatas"])
    assert result["documents"] == ["v2"]
    assert result["metadatas"][0]["a"] == 9
    assert result["metadatas"][0]["b"] == 2


def test_get_limit_above_cap_raises(collection):
    with pytest.raises(ChromaQuotaError):
        collection.get(limit=301)


def test_get_limit_at_cap_does_not_raise(collection):
    assert collection.get(limit=300)["ids"] == []


def test_paged_get_returns_documents_and_metadatas_aligned_to_ids(collection):
    n = 431
    ids = [f"gc73-{i}" for i in range(n)]
    documents = [f"doc-{i}" for i in range(n)]
    metadatas = [
        {
            "show": TRIPLE[0],
            "episode_number": TRIPLE[1],
            "date": TRIPLE[2],
            "n_chunks": n,
            "index": i,
        }
        for i in range(n)
    ]
    for id_batch, doc_batch, meta_batch in zip(
        batched(ids), batched(documents), batched(metadatas), strict=True
    ):
        collection.upsert(ids=id_batch, documents=doc_batch, metadatas=meta_batch)

    result = paged_get(
        collection, episode_where(*TRIPLE), include=["documents", "metadatas"]
    )
    assert len(result["ids"]) == n
    assert len(result["documents"]) == n
    assert len(result["metadatas"]) == n
    for _id, doc, meta in zip(
        result["ids"], result["documents"], result["metadatas"], strict=True
    ):
        index = int(_id.split("-")[-1])
        assert doc == f"doc-{index}"
        assert meta["index"] == index


def test_unimplemented_operator_raises(collection):
    # $ne, $in etc. are not modelled. Defaulting to True would silently
    # misreport a filter the fake does not actually understand.
    _seed(collection, "gc73", 5)
    with pytest.raises(ValueError):
        collection.get(where={"show": {"$ne": "Other Show"}})


def test_get_with_empty_ids_list_returns_nothing(collection):
    # ids=[] is falsy but not absent -- must not fall back to selecting
    # everything, the way `ids or ...` would.
    _seed(collection, "gc73", 5)
    assert collection.get(ids=[])["ids"] == []


def test_episode_where_does_not_match_a_different_date(collection):
    # Same show and episode_number, different date -- the date clause is
    # what stops this from colliding with another episode of the same show.
    _seed(collection, "gc73", 5)
    _seed(collection, "other", 3, date="2020-01-01")
    ids = paged_get_ids(collection, episode_where(*TRIPLE))
    assert len(ids) == 5
    assert all(i.startswith("gc73-") for i in ids)


def test_batched_rejects_a_size_above_the_request_cap():
    with pytest.raises(ValueError):
        list(batched([1, 2, 3], size=STORE_MAX_REQUEST + 1))


def test_batched_accepts_a_size_at_the_request_cap():
    assert list(batched([1, 2], size=STORE_MAX_REQUEST)) == [[1, 2]]
