from corpus.completeness import episode_state, plan_episode
from corpus.planning import Action
from corpus.store import batched

SHOW, EP, DATE = "Geopolitical Cousins", "73", "2026-07-29"

TRANSCRIPT = """# Geopolitical Cousins - Episode 73

[SPEAKER_00] 1.0s - One.
[SPEAKER_01] 2.0s - Two.
"""


def _seed(collection, n, *, n_chunks, rules_version="1"):
    for batch in batched(list(range(n))):
        collection.upsert(
            ids=[f"Geopolitical_Cousins-ep73-2026-07-29-{i}" for i in batch],
            documents=[f"d{i}" for i in batch],
            metadatas=[
                {
                    "show": SHOW,
                    "episode_number": EP,
                    "date": DATE,
                    "n_chunks": n_chunks,
                    "rules_version": rules_version,
                }
                for _ in batch
            ],
        )


def test_state_of_an_absent_episode(collection):
    ids, n_chunks, rules = episode_state(collection, SHOW, EP, DATE)
    assert (ids, n_chunks, rules) == ([], None, None)


def test_state_reads_expected_count_and_version(collection):
    _seed(collection, 2, n_chunks=2)
    ids, n_chunks, rules = episode_state(collection, SHOW, EP, DATE)
    assert len(ids) == 2
    assert n_chunks == 2
    assert rules == "1"


def test_complete_episode_is_skipped(collection):
    _seed(collection, 2, n_chunks=2)
    assert (
        plan_episode(
            collection,
            show=SHOW,
            episode_number=EP,
            date_str=DATE,
            transcript_text=TRANSCRIPT,
        )
        is Action.SKIP
    )


def test_torn_episode_is_re_embedded(collection):
    # One surviving orphan out of two. A boolean check would call this healthy.
    _seed(collection, 1, n_chunks=2)
    assert (
        plan_episode(
            collection,
            show=SHOW,
            episode_number=EP,
            date_str=DATE,
            transcript_text=TRANSCRIPT,
        )
        is Action.EMBED_ONLY
    )


def test_stale_rules_version_is_re_embedded(collection):
    _seed(collection, 2, n_chunks=2, rules_version="0")
    assert (
        plan_episode(
            collection,
            show=SHOW,
            episode_number=EP,
            date_str=DATE,
            transcript_text=TRANSCRIPT,
        )
        is Action.EMBED_ONLY
    )


def test_absent_transcript_is_transcribed(collection):
    assert (
        plan_episode(
            collection,
            show=SHOW,
            episode_number=EP,
            date_str=DATE,
            transcript_text=None,
        )
        is Action.TRANSCRIBE
    )


def test_excluded_episode_is_excluded(collection):
    assert (
        plan_episode(
            collection,
            show="The Jacob Shapiro Podcast",
            episode_number="Unknown",
            date_str="2026-07-29",
            transcript_text=TRANSCRIPT,
        )
        is Action.EXCLUDE
    )


def test_exclusion_survives_a_backfilled_episode_number_via_the_guid(collection):
    # THE enforcement test. Captivate backfilling itunes_episode changes the
    # triple, so the triple arm stops matching -- and without the guid
    # threaded through plan_episode the whole guid arm is unreachable from the
    # cron path, present and documented and never firing.
    assert (
        plan_episode(
            collection,
            show="The Jacob Shapiro Podcast",
            episode_number="352",  # backfilled; the triple no longer matches
            date_str="2026-07-29",
            transcript_text=TRANSCRIPT,
            episode_guid="1c45dbd9-0dc3-4d07-b2d1-758fe78405fe",
        )
        is Action.EXCLUDE
    )


def test_a_normal_episode_with_a_guid_is_not_excluded(collection):
    assert (
        plan_episode(
            collection,
            show=SHOW,
            episode_number=EP,
            date_str=DATE,
            transcript_text=TRANSCRIPT,
            episode_guid="b4a9c88b-9dbf-46b7-9dc1-a7812a9bde65",
        )
        is not Action.EXCLUDE
    )


def test_a_431_chunk_episode_is_judged_complete_not_looping(collection):
    # Unpaged, the count would come back 300 against n_chunks=431 and this
    # episode would be re-embedded on every cron run forever.
    _seed(collection, 431, n_chunks=431)
    assert (
        plan_episode(
            collection,
            show=SHOW,
            episode_number=EP,
            date_str=DATE,
            transcript_text=TRANSCRIPT,
            expected_n_chunks=431,
        )
        is Action.SKIP
    )
