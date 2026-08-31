import pytest

from corpus.store import (
    batched,
    episode_where,
    guid_where,
    is_complete,
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
