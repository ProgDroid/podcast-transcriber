"""The backfill writes ONE new metadata key and nothing else.

Chroma merges metadata on upsert (measured on Cloud, see conftest), so a key
this writes cannot later be removed by rewriting the episode -- only by
deleting the record. That makes "add, never overwrite" a correctness
requirement rather than a style preference.
"""

from __future__ import annotations

import pytest
from backfill_speaker_id import _require_expect, apply_updates, plan_updates

SHOW = "The Jacob Shapiro Podcast"
EP1 = f"{SHOW} - Episode 1 - 2022-05-02.txt"
EP2 = f"{SHOW} - Episode 2 - 2022-05-06.txt"


@pytest.fixture
def seeded(collection):
    # The metadata written by corpus/chunking.py:66-80. There is NO "episode"
    # key -- an episode is the (show, episode_number, date) triple, which is
    # what corpus/store.py:53 episode_where selects on.
    collection.upsert(
        ids=["ep1::0", "ep1::1", "ep2::0"],
        documents=["[SPEAKER_00] a", "[SPEAKER_01] b", "[SPEAKER_00] c"],
        metadatas=[
            {
                "show": SHOW,
                "episode_number": "1",
                "date": "2022-05-02",
                "speaker": "SPEAKER_00",
            },
            {
                "show": SHOW,
                "episode_number": "1",
                "date": "2022-05-02",
                "speaker": "SPEAKER_01",
            },
            {
                "show": SHOW,
                "episode_number": "2",
                "date": "2022-05-06",
                "speaker": "SPEAKER_00",
            },
        ],
    )
    return collection


def _record(episode, number, date, cluster, person):
    return {
        "episode": episode,
        "show": SHOW,
        "episode_number": number,
        "date": date,
        "cluster": cluster,
        "person": person,
    }


ASSIGNMENTS = [
    _record(EP1, "1", "2022-05-02", "SPEAKER_00", "Jacob Shapiro"),
    _record(EP1, "1", "2022-05-02", "SPEAKER_01", None),
    _record(EP2, "2", "2022-05-06", "SPEAKER_00", "Rob Larity"),
]


def test_an_assignment_without_the_triple_raises():
    # The defect this test exists for, and the one that broke the first draft
    # of this task: a filter that matches nothing plans zero updates and
    # reports success. Refuse instead.
    from backfill_speaker_id import plan_updates as plan

    with pytest.raises(ValueError, match="missing"):
        plan(None, [{"episode": "x.txt", "cluster": "SPEAKER_00", "person": "X"}])


def test_plans_one_update_per_matching_chunk(seeded):
    updates = plan_updates(seeded, ASSIGNMENTS)
    assert {chunk_id for chunk_id, _ in updates} == {"ep1::0", "ep2::0"}


def test_declined_clusters_get_no_key_at_all(seeded):
    # person: null is ground truth. Writing "unknown" would be a value a
    # filter could match, which is exactly the false confidence to avoid.
    updates = plan_updates(seeded, ASSIGNMENTS)
    assert "ep1::1" not in {chunk_id for chunk_id, _ in updates}


def test_the_update_carries_only_speaker_id(seeded):
    updates = plan_updates(seeded, ASSIGNMENTS)
    for _, metadata in updates:
        assert set(metadata) == {"speaker_id"}


def test_speaker_is_never_overwritten(seeded):
    apply_updates(seeded, plan_updates(seeded, ASSIGNMENTS), expected=2)
    got = seeded.get(ids=["ep1::0"], include=["metadatas"])
    assert got["metadatas"][0]["speaker"] == "SPEAKER_00"
    assert got["metadatas"][0]["speaker_id"] == "Jacob Shapiro"


def test_documents_are_not_rewritten(seeded):
    # HONEST ABOUT ITS OWN REACH. FakeCollection.upsert preserves a document
    # by construction when `documents` is None, so this pins the CALLER (it
    # passes no documents) and cannot prove Cloud behaves the same way. The
    # real behaviour of a metadata-only upsert against Cloud is spec UNKNOWN
    # 6.3 and is measured in Step 5 against a scratch collection, not here.
    before = seeded.get(ids=["ep1::0"], include=["documents"])["documents"][0]
    apply_updates(seeded, plan_updates(seeded, ASSIGNMENTS), expected=2)
    after = seeded.get(ids=["ep1::0"], include=["documents"])["documents"][0]
    assert before == after == "[SPEAKER_00] a"


def test_refuses_when_the_count_moved(seeded):
    # A collection that changed since the plan was made must cause a refusal,
    # not an unknown-size write. It must also write NOTHING -- not merely
    # raise -- so this checks the collection's metadata is untouched after.
    before = seeded.get(ids=["ep1::0", "ep2::0"], include=["metadatas"])["metadatas"]
    with pytest.raises(RuntimeError, match="expected 99"):
        apply_updates(seeded, plan_updates(seeded, ASSIGNMENTS), expected=99)
    after = seeded.get(ids=["ep1::0", "ep2::0"], include=["metadatas"])["metadatas"]
    assert before == after
    assert all("speaker_id" not in m for m in after)


def test_batches_stay_under_the_cloud_cap(seeded):
    for i in range(700):
        seeded.upsert(
            ids=[f"big::{i}"],
            documents=["[SPEAKER_00] x"],
            metadatas=[
                {
                    "show": SHOW,
                    "episode_number": "999",
                    "date": "2026-01-01",
                    "speaker": "SPEAKER_00",
                }
            ],
        )
    big = f"{SHOW} - Episode 999 - 2026-01-01.txt"
    many = plan_updates(
        seeded, [_record(big, "999", "2026-01-01", "SPEAKER_00", "Jacob Shapiro")]
    )
    # Both the paging in plan_updates and the batching in apply_updates are
    # under test: get() silently truncates at 300 and upsert() raises above it.
    assert len(many) == 700
    assert apply_updates(seeded, many, expected=700) == 700


def test_apply_refuses_without_expect():
    # `run_backfill`'s apply path calls `_require_expect(expect)` before ever
    # calling `apply_updates`. Without --expect (default 0), it must refuse --
    # falling back to the freshly planned count here would recreate the
    # original tautology (expected computed from the same call it guards).
    with pytest.raises(RuntimeError, match="requires --expect"):
        _require_expect(0)


def test_apply_accepts_a_supplied_expect():
    assert _require_expect(2) == 2


def test_the_count_guard_is_reachable_when_expect_is_independently_sourced(seeded):
    """Reproduces the original defect and proves the fix closes it.

    The old call site computed `expected=len(updates)` from the SAME
    `plan_updates` call it was compared against, so the guard could never
    fire: the two numbers were the same variable read twice. This test
    builds `expect` the way a dry-run actually would -- from an EARLIER,
    separate plan -- then changes the collection (the world moves between
    dry-run and apply), re-plans, and applies with the stale `expect`. The
    guard must fire, and nothing must be written.
    """
    expect = len(plan_updates(seeded, ASSIGNMENTS))  # what the dry-run prints

    # The world moves: a new chunk appears that the plan would also match.
    seeded.upsert(
        ids=["ep2::1"],
        documents=["[SPEAKER_00] extra"],
        metadatas=[
            {
                "show": SHOW,
                "episode_number": "2",
                "date": "2022-05-06",
                "speaker": "SPEAKER_00",
            }
        ],
    )

    fresh = plan_updates(seeded, ASSIGNMENTS)
    assert len(fresh) != expect  # sanity: the world really did move

    with pytest.raises(RuntimeError, match="expected"):
        apply_updates(seeded, fresh, expected=expect)

    got = seeded.get(ids=["ep1::0", "ep2::0", "ep2::1"], include=["metadatas"])
    assert all("speaker_id" not in m for m in got["metadatas"])
