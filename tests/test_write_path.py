from corpus.store import episode_where, paged_get_ids
from corpus.writing import upsert_then_prune

SHOW, EP, DATE = "Geopolitical Cousins", "73", "2026-07-29"


def _chunks(n, speaker="SPEAKER_00", guid=None):
    out = []
    for i in range(n):
        meta = {
            "show": SHOW,
            "episode_number": EP,
            "date": DATE,
            "speaker": speaker,
            "n_chunks": n,
        }
        if guid:
            meta["episode_guid"] = guid
        out.append({"text": f"[{speaker}] chunk {i}", "metadata": meta})
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
