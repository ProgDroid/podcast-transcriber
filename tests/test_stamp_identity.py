"""The one-time pass that makes `identity_version` mean something.

The signal is worthless while the key is absent everywhere, and treating
absent as "needs identity" would queue all 433 archive episodes on night one
to recompute names that are already correct. This stamps the episodes the
archive pass actually processed, so the planner's first real night reports the
episodes published SINCE that pass rather than the whole corpus.

The stamp set is PROVENANCE, not outcome: the 433 triples in
speakers/assignments.json are the archive pass's own record of what it ran on.
"Has a speaker_id" would have selected the same 433 here -- 0 of them have
zero named clusters, so the selectors coincide exactly -- but it reads the
wrong field, and it would silently exclude any episode the pass processed and
matched nobody in.
"""

from __future__ import annotations

import pytest
from stamp_identity_version import (
    episode_triples,
    plan_stamp_updates,
    verify_stamped,
)

SHOW = "Geopolitical Cousins"
VERSION = "1@1"


def _assignment(ep, date, cluster, person, show=SHOW):
    return {
        "show": show,
        "episode_number": ep,
        "date": date,
        "cluster": cluster,
        "person": person,
    }


@pytest.fixture
def seeded(collection):
    collection.upsert(
        ids=["gc73::0", "gc73::1", "gc74::0"],
        documents=["[SPEAKER_00] a", "[SPEAKER_01] b", "[SPEAKER_00] c"],
        metadatas=[
            {
                "show": SHOW,
                "episode_number": "73",
                "date": "2026-07-29",
                "speaker": "SPEAKER_00",
                "speaker_id": "Alice",
            },
            {
                "show": SHOW,
                "episode_number": "73",
                "date": "2026-07-29",
                "speaker": "SPEAKER_01",
            },
            {
                "show": SHOW,
                "episode_number": "74",
                "date": "2026-07-31",
                "speaker": "SPEAKER_00",
            },
        ],
    )
    return collection


# --- the stamp set ----------------------------------------------------------


def test_one_triple_per_episode_not_one_per_cluster():
    # assignments.json holds 970 CLUSTER records over 433 distinct episodes.
    # Planning per cluster would query the same episode repeatedly and, worse,
    # invite the per-cluster filter this pass exists to avoid.
    triples = episode_triples(
        [
            _assignment("73", "2026-07-29", "SPEAKER_00", "Alice"),
            _assignment("73", "2026-07-29", "SPEAKER_01", None),
            _assignment("74", "2026-07-31", "SPEAKER_00", "Bob"),
        ]
    )

    assert triples == [
        (SHOW, "73", "2026-07-29"),
        (SHOW, "74", "2026-07-31"),
    ]


def test_an_episode_that_matched_nobody_is_still_in_the_stamp_set():
    # PROVENANCE, NOT OUTCOME. Selecting on `person` would silently exclude
    # any episode the archive pass processed and matched nobody in, and those
    # are exactly the episodes that must reach SKIP rather than loop.
    triples = episode_triples([_assignment("73", "2026-07-29", "SPEAKER_00", None)])

    assert triples == [(SHOW, "73", "2026-07-29")]


def test_a_missing_triple_part_is_refused_not_skipped():
    # A record without the full triple builds a filter that selects nothing,
    # plans zero updates, and reports success -- the failure mode with no
    # symptom.
    with pytest.raises(ValueError, match="episode_number"):
        episode_triples([{"show": SHOW, "date": "2026-07-29", "cluster": "SPEAKER_00"}])


def test_the_order_is_stable_so_two_dry_runs_agree():
    triples = episode_triples(
        [
            _assignment("74", "2026-07-31", "SPEAKER_00", "Bob"),
            _assignment("73", "2026-07-29", "SPEAKER_00", "Alice"),
            _assignment("74", "2026-07-31", "SPEAKER_01", None),
        ]
    )

    assert triples == [
        (SHOW, "73", "2026-07-29"),
        (SHOW, "74", "2026-07-31"),
    ]


# --- planning the writes ----------------------------------------------------


def test_every_chunk_of_a_stamped_episode_is_planned(seeded):
    # THE DIFFERENCE FROM backfill_speaker_id.plan_updates, which filters by
    # cluster and skips `person: None` -- it would leave the 266 unmatched
    # clusters' chunks unstamped, and uniformity is exactly what
    # episode_state's single-record read depends on.
    updates = plan_stamp_updates(
        seeded, [(SHOW, "73", "2026-07-29")], identity_version=VERSION
    )

    assert sorted(chunk_id for chunk_id, _ in updates) == ["gc73::0", "gc73::1"]
    assert all(meta == {"identity_version": VERSION} for _, meta in updates)


def test_the_chunk_of_an_unmatched_cluster_is_stamped(seeded):
    updates = plan_stamp_updates(
        seeded, [(SHOW, "73", "2026-07-29")], identity_version=VERSION
    )

    # gc73::1 carries no speaker_id at all. It is still stamped, because the
    # key records that identification RAN, never what it found.
    assert "gc73::1" in {chunk_id for chunk_id, _ in updates}


def test_only_identity_version_is_written(seeded):
    # Chroma merges metadata, so a key this writes cannot later be removed by
    # rewriting the episode -- only by deleting the record. "Add exactly one
    # key, overwrite nothing" is a correctness requirement here.
    updates = plan_stamp_updates(
        seeded, [(SHOW, "73", "2026-07-29")], identity_version=VERSION
    )

    for _, meta in updates:
        assert set(meta) == {"identity_version"}


def test_a_book_record_sharing_the_triple_is_not_stamped(collection):
    # upload_book.py's 191 records can share a triple by construction. The
    # write path already refuses to prune them; stamping one would claim a
    # book chunk was identified as a podcast episode.
    collection.upsert(
        ids=["gc73::0", "book::0"],
        documents=["[SPEAKER_00] a", "a page"],
        metadatas=[
            {
                "show": SHOW,
                "episode_number": "73",
                "date": "2026-07-29",
                "speaker": "SPEAKER_00",
            },
            {
                "show": SHOW,
                "episode_number": "73",
                "date": "2026-07-29",
                "source": "book",
            },
        ],
    )

    updates = plan_stamp_updates(
        collection, [(SHOW, "73", "2026-07-29")], identity_version=VERSION
    )

    assert [chunk_id for chunk_id, _ in updates] == ["gc73::0"]


def test_an_episode_with_no_records_plans_nothing_rather_than_raising(seeded):
    updates = plan_stamp_updates(
        seeded, [(SHOW, "999", "2030-01-01")], identity_version=VERSION
    )

    assert updates == []


def test_planning_is_paged(collection):
    # An unlimited get() SILENTLY returns 300. An unpaged plan against a
    # 431-chunk episode would stamp 300 records, leave 131 unstamped, and
    # report success -- and the episode would then be non-uniform, which is
    # the one state episode_state's head-record read cannot survive.
    n = 431
    for start in range(0, n, 250):
        ids = list(range(start, min(start + 250, n)))
        collection.upsert(
            ids=[f"gc73::{i}" for i in ids],
            documents=[f"d{i}" for i in ids],
            metadatas=[
                {
                    "show": SHOW,
                    "episode_number": "73",
                    "date": "2026-07-29",
                    "speaker": "SPEAKER_00",
                }
                for _ in ids
            ],
        )

    updates = plan_stamp_updates(
        collection, [(SHOW, "73", "2026-07-29")], identity_version=VERSION
    )

    assert len(updates) == n


# --- verification -----------------------------------------------------------


def test_verification_reads_back_rather_than_trusting_the_writer(seeded):
    # The pattern tests/test_backfill.py already establishes. A writer's
    # return value cannot detect an update() that reports success and writes
    # nothing, which is exactly what a metadata-only call invites.
    stamped, total = verify_stamped(
        seeded, [(SHOW, "73", "2026-07-29")], identity_version=VERSION
    )

    assert (stamped, total) == (0, 2)

    seeded.update(
        ids=["gc73::0", "gc73::1"],
        metadatas=[{"identity_version": VERSION}, {"identity_version": VERSION}],
    )

    assert verify_stamped(
        seeded, [(SHOW, "73", "2026-07-29")], identity_version=VERSION
    ) == (2, 2)


def test_verification_does_not_count_a_stale_version(seeded):
    seeded.update(ids=["gc73::0"], metadatas=[{"identity_version": "0@1"}])

    stamped, total = verify_stamped(
        seeded, [(SHOW, "73", "2026-07-29")], identity_version=VERSION
    )

    assert (stamped, total) == (0, 2)


def test_verification_ignores_book_records(collection):
    collection.upsert(
        ids=["gc73::0", "book::0"],
        documents=["[SPEAKER_00] a", "a page"],
        metadatas=[
            {
                "show": SHOW,
                "episode_number": "73",
                "date": "2026-07-29",
                "speaker": "SPEAKER_00",
                "identity_version": VERSION,
            },
            {
                "show": SHOW,
                "episode_number": "73",
                "date": "2026-07-29",
                "source": "book",
            },
        ],
    )

    # The book record is neither stamped nor counted, so a fully stamped
    # episode reads as fully stamped rather than as one short forever.
    assert verify_stamped(
        collection, [(SHOW, "73", "2026-07-29")], identity_version=VERSION
    ) == (1, 1)


# --- the gate ---------------------------------------------------------------


def test_the_expect_gate_is_the_same_function_not_a_lookalike():
    # Two copies of an operator gate drift, and the stale one keeps working
    # right up until it matters.
    import backfill_speaker_id
    import stamp_identity_version

    assert stamp_identity_version._require_expect is backfill_speaker_id._require_expect


def test_applying_without_expect_refuses():
    import stamp_identity_version

    with pytest.raises(RuntimeError, match="--expect"):
        stamp_identity_version._require_expect(-1)


def test_a_planned_count_of_zero_is_a_legitimate_plan():
    # 0 is representable and must not be read as "not supplied".
    import stamp_identity_version

    assert stamp_identity_version._require_expect(0) == 0
