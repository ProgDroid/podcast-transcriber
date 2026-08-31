# tests/test_remap.py
from corpus.remap import remap_id


def test_podcast_record_is_remapped_with_the_date_inserted():
    meta = {
        "show": "Geopolitical Cousins",
        "episode_number": "73",
        "date": "2026-07-29",
    }
    result = remap_id("Geopolitical_Cousins-ep73-17", meta)
    assert result.new_id == "Geopolitical_Cousins-ep73-2026-07-29-17"
    assert result.classification == "remapped"


def test_the_two_episode_243s_stop_colliding():
    a = remap_id(
        "The_Jacob_Shapiro_Podcast-ep243-0",
        {
            "show": "The Jacob Shapiro Podcast",
            "episode_number": "243",
            "date": "2024-11-07",
        },
    )
    b = remap_id(
        "The_Jacob_Shapiro_Podcast-ep243-0",
        {
            "show": "The Jacob Shapiro Podcast",
            "episode_number": "243",
            "date": "2024-11-08",
        },
    )
    assert a.new_id != b.new_id


def test_book_records_pass_through_untouched():
    # upload_book.py writes Geopolitical_Alpha-p{n}; these ids are already
    # unique and carry no episode concept. Applying the podcast scheme to
    # them would corrupt them.
    meta = {"show": "Geopolitical Alpha", "episode_number": "N/A", "date": "2021-01-01"}
    result = remap_id("Geopolitical_Alpha-p179", meta)
    assert result.new_id == "Geopolitical_Alpha-p179"
    assert result.classification == "passthrough_non_episode"


def test_an_id_whose_metadata_does_not_reconstruct_it_passes_through():
    # Never guess. If the id and the metadata disagree, leave it alone and
    # let reconciliation report it.
    meta = {"show": "Some Other Show", "episode_number": "9", "date": "2025-01-01"}
    result = remap_id("Geopolitical_Cousins-ep73-17", meta)
    assert result.new_id == "Geopolitical_Cousins-ep73-17"
    assert result.classification == "passthrough_unmatched"


def test_an_already_migrated_id_is_left_alone():
    meta = {
        "show": "Geopolitical Cousins",
        "episode_number": "73",
        "date": "2026-07-29",
    }
    new = "Geopolitical_Cousins-ep73-2026-07-29-17"
    result = remap_id(new, meta)
    assert result.new_id == new
    assert result.classification == "remapped"


def test_missing_metadata_keys_pass_through():
    assert remap_id("Whatever-ep1-0", {}).classification == "passthrough_unmatched"


def test_old_and_new_scheme_ids_can_never_collide():
    # Old ids end in an integer; new ids end in YYYY-MM-DD-{int}. This is what
    # makes a mixed-scheme corpus safe during the migration.
    meta = {"show": "Show", "episode_number": "1", "date": "2025-01-01"}
    old = {f"Show-ep1-{i}" for i in range(200)}
    new = {remap_id(f"Show-ep1-{i}", meta).new_id for i in range(200)}
    assert old & new == set()
