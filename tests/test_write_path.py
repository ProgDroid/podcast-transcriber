import math

from corpus.store import BATCH, episode_where, paged_get_ids
from corpus.writing import upsert_then_prune

SHOW, EP, DATE = "Geopolitical Cousins", "73", "2026-07-29"

# upload_book.py's own metadata, reused verbatim: it deliberately populates
# show/episode_number/date "so filters work consistently" with the podcast
# episodes -- the field values below are its BOOK_TITLE, its sentinel
# episode_number, and its publication-year date, not made up for this test.
BOOK_SHOW, BOOK_EP, BOOK_DATE = "Geopolitical Alpha", "N/A", "2021-01-01"


def _chunks_for(show, ep, date, n, speaker="SPEAKER_00", guid=None):
    out = []
    for i in range(n):
        meta = {
            "show": show,
            "episode_number": ep,
            "date": date,
            "speaker": speaker,
            "n_chunks": n,
        }
        if guid:
            meta["episode_guid"] = guid
        out.append({"text": f"[{speaker}] chunk {i}", "metadata": meta})
    return out


def _chunks(n, speaker="SPEAKER_00", guid=None):
    return _chunks_for(SHOW, EP, DATE, n, speaker=speaker, guid=guid)


def _book_chunks(n):
    out = []
    for i in range(n):
        out.append(
            {
                "text": f"book chunk {i}",
                "metadata": {
                    "source": "book",
                    "show": BOOK_SHOW,
                    "episode_number": BOOK_EP,
                    "date": BOOK_DATE,
                    "n_chunks": n,
                },
            }
        )
    return out


def _embeddings(n):
    return [[float(i), 0.0, 1.0] for i in range(n)]


def _write(collection, n, **kw):
    return upsert_then_prune(
        collection,
        _chunks(n, **kw),
        _embeddings(n),
        show=SHOW,
        episode_number=EP,
        date_str=DATE,
        episode_guid=kw.get("guid"),
    )


def _write_triple(collection, show, ep, date, n, guid=None):
    return upsert_then_prune(
        collection,
        _chunks_for(show, ep, date, n, guid=guid),
        _embeddings(n),
        show=show,
        episode_number=ep,
        date_str=date,
        episode_guid=guid,
    )


def test_writes_a_431_chunk_episode_without_hitting_the_request_cap(collection):
    result = _write(collection, 431)
    assert result["written"] == 431
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 431


def test_a_shrinking_re_embed_leaves_no_orphans(collection):
    _write(collection, 431)
    result = _write(collection, 300, speaker="Jacob Shapiro")
    assert result["pruned"] == 131
    ids = paged_get_ids(collection, episode_where(SHOW, EP, DATE))
    assert len(ids) == 300
    docs = collection.get(ids=ids, include=["documents"])["documents"]
    assert all("SPEAKER_00" not in d for d in docs)


def test_a_growing_re_embed_prunes_nothing(collection):
    _write(collection, 100)
    result = _write(collection, 150)
    assert result["pruned"] == 0
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 150


def test_a_crash_between_upsert_and_prune_leaves_a_superset_not_a_hole(collection):
    # Simulate: the upsert landed, the prune did not.
    _write(collection, 431)
    upsert_then_prune(
        collection,
        _chunks(300, speaker="Jacob Shapiro"),
        _embeddings(300),
        show=SHOW,
        episode_number=EP,
        date_str=DATE,
        episode_guid=None,
        _skip_prune=True,
    )
    ids = paged_get_ids(collection, episode_where(SHOW, EP, DATE))
    assert len(ids) == 431  # superset -- every chunk 0..299 rewritten, 300..430 stale
    # The next run converges.
    _write(collection, 300, speaker="Jacob Shapiro")
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 300


def test_prune_unions_guid_and_triple_so_old_records_are_not_stranded(collection):
    # Pre-migration records are triple-keyed with NO guid. A guid-only prune
    # would match nothing and leave the entire old set behind.
    _write(collection, 50)  # no guid
    result = _write(collection, 40, guid="guid-abc")
    assert result["pruned"] == 10
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 40


def test_another_episode_is_never_touched(collection):
    _write(collection, 20)
    upsert_then_prune(
        collection,
        [
            {
                "text": "other",
                "metadata": {
                    "show": SHOW,
                    "episode_number": "74",
                    "date": "2026-07-31",
                    "n_chunks": 1,
                },
            }
        ],
        [[0.0, 0.0, 1.0]],
        show=SHOW,
        episode_number="74",
        date_str="2026-07-31",
        episode_guid=None,
    )
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 20


def test_a_normal_episode_write_never_touches_the_book(collection):
    # upload_book.py reuses show/episode_number/date "so filters work
    # consistently" -- 191 real records at a triple no feed can ever produce.
    # A normal podcast write is at a completely different triple; confirm the
    # triple scoping alone is sufficient to leave the book alone, with no
    # source/id-shape check needed in upsert_then_prune.
    upsert_then_prune(
        collection,
        _book_chunks(191),
        _embeddings(191),
        show=BOOK_SHOW,
        episode_number=BOOK_EP,
        date_str=BOOK_DATE,
        episode_guid=None,
    )
    _write(collection, 20)
    book_ids = paged_get_ids(collection, episode_where(BOOK_SHOW, BOOK_EP, BOOK_DATE))
    assert len(book_ids) == 191
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 20


def test_same_show_and_episode_number_different_date_is_never_confused(collection):
    # The Jacob Shapiro Podcast Episode 243 exists on both 2024-11-07 and
    # 2024-11-08 -- distinct episodes. Dropping `date` from the prune's scope
    # would make each write wipe the other.
    show = "The Jacob Shapiro Podcast"
    _write_triple(collection, show, "243", "2024-11-07", 15)
    _write_triple(collection, show, "243", "2024-11-08", 12)
    assert (
        len(paged_get_ids(collection, episode_where(show, "243", "2024-11-07"))) == 15
    )
    assert (
        len(paged_get_ids(collection, episode_where(show, "243", "2024-11-08"))) == 12
    )


def test_prune_deletes_are_batched_at_the_store_batch_size(collection):
    _write(collection, 431)
    result = _write(collection, 100, speaker="Jacob Shapiro")
    assert result["pruned"] == 331

    delete_calls = [n for kind, n in collection.calls if kind == "delete"]
    assert delete_calls
    assert max(delete_calls) <= BATCH
    assert len(delete_calls) == math.ceil(331 / BATCH)


def test_prune_by_guid_reaches_records_backfilled_to_a_different_triple(collection):
    # Pre-backfill: episode_number was the sentinel "Unknown", guid already
    # known. The feed later backfills the true number -- the triple changes
    # but the guid doesn't move, so the guid arm must still find the old set
    # even though it now lives under a DIFFERENT triple than the new write.
    _write_triple(collection, SHOW, "Unknown", DATE, 20, guid="guid-xyz")
    result = _write_triple(collection, SHOW, EP, DATE, 20, guid="guid-xyz")
    assert result["pruned"] == 20
    assert len(paged_get_ids(collection, episode_where(SHOW, "Unknown", DATE))) == 0
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 20


def test_different_shows_same_episode_number_and_date_are_never_confused(collection):
    # Two feeds publishing on the same date is routine, and five episodes
    # carry the literal "Unknown" episode number -- episode_number and date
    # alone are not enough to key a prune; `show` must always be in it too.
    _write_triple(collection, "Geopolitical Cousins", "1", "2020-06-15", 10)
    _write_triple(collection, "The Jacob Shapiro Podcast", "1", "2020-06-15", 8)
    assert (
        len(
            paged_get_ids(
                collection, episode_where("Geopolitical Cousins", "1", "2020-06-15")
            )
        )
        == 10
    )
    assert (
        len(
            paged_get_ids(
                collection,
                episode_where("The Jacob Shapiro Podcast", "1", "2020-06-15"),
            )
        )
        == 8
    )


def test_a_write_at_the_books_own_triple_never_prunes_the_book(collection):
    # Reproduces the destroying case directly: something writes at the exact
    # triple upload_book.py's records occupy. Even though nothing in the
    # pipeline can produce that triple today (transcribe.py's sentinel is
    # "Unknown", never "N/A", and parse_transcript_filename's episode-number
    # group can't match "N/A"), the guard must hold regardless of triple --
    # "unguarded but unreachable" is exactly the shape this branch keeps
    # finding one task later.
    upsert_then_prune(
        collection,
        _book_chunks(191),
        _embeddings(191),
        show=BOOK_SHOW,
        episode_number=BOOK_EP,
        date_str=BOOK_DATE,
        episode_guid=None,
    )
    result = upsert_then_prune(
        collection,
        _chunks_for(BOOK_SHOW, BOOK_EP, BOOK_DATE, 1),
        _embeddings(1),
        show=BOOK_SHOW,
        episode_number=BOOK_EP,
        date_str=BOOK_DATE,
        episode_guid=None,
    )
    assert result["pruned"] == 0
    book_ids = paged_get_ids(collection, episode_where(BOOK_SHOW, BOOK_EP, BOOK_DATE))
    assert len(book_ids) == 191


def test_prune_by_guid_never_crosses_a_show_boundary(collection):
    # A guid is only unique WITHIN a feed. Geopolitical Cousins 73 was
    # cross-posted (different guid) but a shared guid on a different show is
    # exactly the shape a guid-only prune would wrongly cross.
    other_show = "The Jacob Shapiro Podcast"
    _write_triple(collection, other_show, "1", "2020-01-01", 30, guid="shared-guid")
    result = _write_triple(collection, SHOW, EP, DATE, 25, guid="shared-guid")
    assert result["pruned"] == 0
    assert (
        len(paged_get_ids(collection, episode_where(other_show, "1", "2020-01-01")))
        == 30
    )
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 25
