import pytest

from corpus.reconcile import ReconcileReport, reconcile

VOL = {
    ("Geopolitical Cousins", "73", "2026-07-29"),
    ("Geopolitical Cousins", "74", "2026-07-31"),
    ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29"),
}
FEED = {
    ("Geopolitical Cousins", "73", "2026-07-29"),
    ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29"),
}


def _records(key, prefix, indices):
    show, ep, date = key
    return [
        (f"{prefix}-{i}", {"show": show, "episode_number": ep, "date": date})
        for i in indices
    ]


def test_clean_corpus_reports_nothing():
    records = _records(
        ("Geopolitical Cousins", "73", "2026-07-29"),
        "Geopolitical_Cousins-ep73-2026-07-29",
        range(3),
    ) + _records(
        ("Geopolitical Cousins", "74", "2026-07-31"),
        "Geopolitical_Cousins-ep74-2026-07-31",
        range(2),
    )
    report = reconcile(VOL, records, FEED)
    assert report.missing == []
    assert report.extra == []
    assert report.non_contiguous == []
    assert report.excluded_with_records == []


def test_missing_episode_is_reported():
    records = _records(
        ("Geopolitical Cousins", "73", "2026-07-29"),
        "Geopolitical_Cousins-ep73-2026-07-29",
        range(3),
    )
    report = reconcile(VOL, records, FEED)
    assert ("Geopolitical Cousins", "74", "2026-07-31") in report.missing


def test_records_with_no_volume_file_are_extra():
    records = _records(
        ("Ghost Show", "1", "2020-01-01"), "Ghost_Show-ep1-2020-01-01", range(2)
    )
    report = reconcile(VOL, records, FEED)
    assert ("Ghost Show", "1", "2020-01-01") in report.extra


def test_an_orphan_tail_shows_as_non_contiguous():
    # Indices 0,1,5 -- what a shrinking re-embed leaves behind.
    records = _records(
        ("Geopolitical Cousins", "73", "2026-07-29"),
        "Geopolitical_Cousins-ep73-2026-07-29",
        [0, 1, 5],
    )
    report = reconcile(VOL, records, FEED)
    assert ("Geopolitical Cousins", "73", "2026-07-29") in report.non_contiguous


def test_an_excluded_episode_holding_records_is_reported():
    records = _records(
        ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29"),
        "The_Jacob_Shapiro_Podcast-epUnknown-2026-07-29",
        range(2),
    )
    report = reconcile(VOL, records, FEED)
    assert (
        "The Jacob Shapiro Podcast",
        "Unknown",
        "2026-07-29",
    ) in report.excluded_with_records


def test_a_returned_exclusion_under_a_backfilled_number_is_not_mere_extra():
    # Same episode, backfilled episode_number, recorded guid. Without the guid
    # arm this lands in `extra` and the alarm names the wrong problem.
    records = [
        (
            "The_Jacob_Shapiro_Podcast-ep352-2026-07-29-0",
            {
                "show": "The Jacob Shapiro Podcast",
                "episode_number": "352",
                "date": "2026-07-29",
                "episode_guid": "1c45dbd9-0dc3-4d07-b2d1-758fe78405fe",
            },
        )
    ]
    report = reconcile(VOL, records, FEED)
    key = ("The Jacob Shapiro Podcast", "352", "2026-07-29")
    assert key in report.excluded_with_records
    assert key not in report.extra


def test_an_excluded_episode_is_never_reported_as_missing():
    report = reconcile(VOL, [], FEED)
    assert (
        "The Jacob Shapiro Podcast",
        "Unknown",
        "2026-07-29",
    ) not in report.missing


def test_a_torn_episode_with_a_contiguous_prefix_is_reported_incomplete():
    # Indices 0..299 of an expected 431 -- contiguous, so non_contiguous
    # stays silent, and something is present, so missing stays silent too.
    # incomplete is the only field whose job this is.
    key = ("Geopolitical Cousins", "73", "2026-07-29")
    records = [
        (
            f"Geopolitical_Cousins-ep73-2026-07-29-{i}",
            {
                "show": key[0],
                "episode_number": key[1],
                "date": key[2],
                "n_chunks": 431,
            },
        )
        for i in range(300)
    ]
    report = reconcile(VOL, records, FEED)
    assert key in report.incomplete
    assert key not in report.non_contiguous
    assert key not in report.missing
    assert report.is_clean() is False


def test_a_complete_episode_with_n_chunks_is_not_reported_incomplete():
    key = ("Geopolitical Cousins", "73", "2026-07-29")
    records = _records(key, "Geopolitical_Cousins-ep73-2026-07-29", range(3))
    records = [(rid, {**meta, "n_chunks": 3}) for rid, meta in records]
    report = reconcile(VOL, records, FEED)
    assert key not in report.incomplete


def test_a_record_with_no_n_chunks_is_never_reported_incomplete():
    # Pre-migration records carry no n_chunks at all. Absent must read as
    # "cannot verify", not as a mismatch -- these must never appear here,
    # no matter how few chunks are stored.
    key = ("Geopolitical Cousins", "73", "2026-07-29")
    records = _records(key, "Geopolitical_Cousins-ep73-2026-07-29", [0])
    report = reconcile(VOL, records, FEED)
    assert key not in report.incomplete


def test_a_shared_id_prefix_is_reported():
    records = [
        ("Show-ep1-0", {"show": "A", "episode_number": "1", "date": "2025-01-01"}),
        ("Show-ep1-1", {"show": "B", "episode_number": "1", "date": "2025-01-02"}),
    ]
    report = reconcile(set(), records, set())
    assert "Show-ep1" in report.shared_prefixes


def test_volume_files_with_no_feed_entry_are_their_own_class():
    report = reconcile(VOL, [], FEED)
    assert ("Geopolitical Cousins", "74", "2026-07-31") in report.feed_unreachable


def test_is_clean_is_false_when_anything_is_wrong():
    assert not reconcile(VOL, [], FEED).is_clean()


# One value per FAULT field is_clean() consults, typed to match the
# dataclass field it fills. Built directly against ReconcileReport rather
# than driven through reconcile() -- the point is pinning the predicate
# itself, not re-testing detection.
_FAULT_FIELD_VALUES = {
    "missing": [("A", "1", "2025-01-01")],
    "extra": [("A", "1", "2025-01-01")],
    "non_contiguous": [("A", "1", "2025-01-01")],
    "shared_prefixes": ["A-ep1"],
    "excluded_with_records": [("A", "1", "2025-01-01")],
    "incomplete": [("A", "1", "2025-01-01")],
}


@pytest.mark.parametrize("field_name", sorted(_FAULT_FIELD_VALUES))
def test_is_clean_is_false_for_each_fault_field(field_name):
    # A report with ONLY this one field populated must already be unclean --
    # this is what pins is_clean() to actually consult all five fault
    # fields, not just whichever one a given reconcile() call happens to hit.
    report = ReconcileReport(**{field_name: _FAULT_FIELD_VALUES[field_name]})
    assert report.is_clean() is False


def test_is_clean_is_true_with_only_informational_fields_set():
    # feed_unreachable and suspected_cross_posts are documented as
    # informational, not faults -- populating only those must stay clean.
    report = ReconcileReport(
        feed_unreachable=[("A", "1", "2025-01-01")],
        suspected_cross_posts=[
            (("A", "1", "2025-01-01"), ("B", "2", "2025-01-02")),
        ],
    )
    assert report.is_clean() is True


def test_book_records_are_not_reported_as_extra():
    # upload_book.py deliberately reuses episode-shaped metadata fields
    # (show="Geopolitical Alpha", episode_number="N/A", date="2021-01-01")
    # plus source="book", and ids like "{title}-p{page}-{i}" that match the
    # digit-suffix shape reconcile() otherwise groups on. Left unfiltered,
    # 191 book records would show up as `extra` (and feed shared_prefixes /
    # non_contiguous noise) on every single run -- the cry-wolf failure mode
    # that trains an operator to stop reading the report.
    records = [
        (
            f"Geopolitical Alpha Book-p1-{i}",
            {
                "show": "Geopolitical Alpha",
                "episode_number": "N/A",
                "date": "2021-01-01",
                "source": "book",
            },
        )
        for i in range(3)
    ]
    report = reconcile(set(), records, set())
    assert report.extra == []
    assert report.non_contiguous == []
    assert report.shared_prefixes == []
