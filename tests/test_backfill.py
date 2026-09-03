"""The backfill writes ONE new metadata key and nothing else.

Chroma merges metadata on upsert (measured on Cloud, see conftest), so a key
this writes cannot later be removed by rewriting the episode -- only by
deleting the record. That makes "add, never overwrite" a correctness
requirement rather than a style preference.
"""

from __future__ import annotations

import pytest
from backfill_speaker_id import (
    _require_expect,
    apply_updates,
    plan_updates,
    run_backfill,
)

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


@pytest.fixture
def wired(seeded, monkeypatch):
    """`run_backfill.local(...)` wired to the fake collection, not Cloud.

    `run_backfill` looks up its collection via `transcribe.get_chroma_collection`
    (imported inside the function body) and reads the Chroma credentials from
    `os.environ` before that call. Substituting both is what makes the Modal
    function's body -- not just `apply_updates` in isolation -- reachable from
    a unit test.
    """
    monkeypatch.setattr("transcribe.get_chroma_collection", lambda *a, **kw: seeded)
    monkeypatch.setenv("CHROMA_API_KEY", "x")
    monkeypatch.setenv("CHROMA_TENANT", "x")
    monkeypatch.setenv("CHROMA_DATABASE", "x")
    return seeded


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
    # -1 is the "not supplied" sentinel (the default for both `run_backfill`
    # and the local entrypoint). Falling back to the freshly planned count
    # here would recreate the original tautology (expected computed from the
    # same call it guards).
    with pytest.raises(RuntimeError, match="requires --expect"):
        _require_expect(-1)


def test_apply_accepts_a_supplied_expect():
    assert _require_expect(2) == 2


def test_apply_accepts_a_genuine_zero_count():
    # A plan can legitimately contain zero updates (every cluster in the
    # batch declined). 0 must be representable, not treated as "missing" --
    # the second review finding: `expect: int = 0` used to conflate the two.
    assert _require_expect(0) == 0


def test_apply_updates_refuses_when_expect_predates_a_moved_collection(seeded):
    """Models the dry-run/apply flow end to end, but only through `apply_updates`.

    NOTE ON REACH: this test calls `apply_updates` directly with two
    independently-computed ints. `apply_updates`'s own guard was always
    correct in isolation, both before and after the wiring fix -- the
    original defect lived purely in `run_backfill`'s call site, which built
    `expected` from the very `plan_updates` call it was compared against.
    This test cannot distinguish the fixed call site from the broken one; it
    is a duplicate of `test_refuses_when_the_count_moved` with a more
    elaborate setup, not a proof the wiring is correct. That proof is
    `test_run_backfill_apply_refuses_when_expect_predates_a_moved_collection`
    below, which calls `run_backfill.local(...)`.
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


# -- tests that actually exercise run_backfill's wiring, not just apply_updates --


def test_run_backfill_dry_run_writes_nothing_even_without_expect(wired):
    result = run_backfill.local(ASSIGNMENTS, False)
    assert result == {"planned": 2, "written": 0}
    got = wired.get(ids=["ep1::0", "ep2::0"], include=["metadatas"])
    assert all("speaker_id" not in m for m in got["metadatas"])


def test_run_backfill_apply_without_expect_refuses_and_writes_nothing(wired):
    before = wired.get(ids=["ep1::0", "ep2::0"], include=["metadatas"])["metadatas"]
    with pytest.raises(RuntimeError, match="requires --expect"):
        run_backfill.local(ASSIGNMENTS, True)
    after = wired.get(ids=["ep1::0", "ep2::0"], include=["metadatas"])["metadatas"]
    assert before == after


def test_run_backfill_apply_refuses_when_expect_predates_a_moved_collection(wired):
    """This is the test that discriminates the fixed call site from the broken one.

    Under the old code, `run_backfill` called
    `apply_updates(..., expected=len(updates))` using the SAME `plan_updates`
    result it was checking -- the two numbers could never disagree, so this
    scenario (a stale `expect` from an earlier plan, applied after the
    collection moved) could not be constructed as a call to `run_backfill`
    at all. It fires here because `expect` now genuinely originates outside
    the apply-time planning call.
    """
    stale_expect = len(
        plan_updates(wired, ASSIGNMENTS)
    )  # what an earlier dry-run printed

    # The world moves between dry-run and apply.
    wired.upsert(
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

    with pytest.raises(RuntimeError, match="expected"):
        run_backfill.local(ASSIGNMENTS, True, stale_expect)

    got = wired.get(ids=["ep1::0", "ep2::0", "ep2::1"], include=["metadatas"])
    assert all("speaker_id" not in m for m in got["metadatas"])


def test_run_backfill_apply_with_a_matching_expect_writes_speaker_id(wired):
    expect = len(plan_updates(wired, ASSIGNMENTS))

    result = run_backfill.local(ASSIGNMENTS, True, expect)

    assert result == {"planned": 2, "written": 2}
    got = wired.get(ids=["ep1::0", "ep2::0"], include=["metadatas"])
    speaker_ids = {m["speaker_id"] for m in got["metadatas"]}
    assert speaker_ids == {"Jacob Shapiro", "Rob Larity"}
