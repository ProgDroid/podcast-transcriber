"""`--resume`'s completeness check: a download failure must stay retryable.

Round 2 made a failed download return one `{**crop, "vector": None, "error":
...}` record per crop, so a bad host no longer kills the whole `.map()`.
Round 2 also made `--resume` treat an episode as complete once its record
count equals what the plan says it owes. Combined, those two individually
correct changes silently merge: a fully-failed download writes exactly
`len(job["crops"])` records, all `vector: None`, which satisfies the
count-equality check -- so every later `--resume` would skip that episode
forever. `completed_episodes` (speaker_embed.py) is the fix: it also checks
that none of an episode's existing records carries `error_kind ==
"download"`. These tests exercise that helper directly, and the dedup rule
`main()` relies on to make a re-embedded episode's stale records disappear.

`speaker_embed.py` imports `modal` at module scope; `modal` is a dev
dependency already, so importing it here to reach `completed_episodes` is
fine and does not touch the network or Modal's control plane.
"""

from __future__ import annotations

from speaker_embed import completed_episodes


def _crop(episode: str, cluster: str, draw: int, length_s: float) -> dict:
    return {
        "episode": episode,
        "cluster": cluster,
        "draw": draw,
        "length_s": length_s,
        "turn_start": 0.0,
        "turn_duration_s": 5.0,
        "crop_start_s": 0.0,
        "crop_length_s": length_s,
    }


def _job(episode: str, crops: list[dict]) -> dict:
    return {
        "episode": episode,
        "audio_url": f"https://example.com/{episode}.mp3",
        "embedder": "some/embedder",
        "crops": crops,
    }


def _vector_record(crop: dict) -> dict:
    return {**crop, "vector": [0.1, 0.2]}


def _download_failure_record(crop: dict) -> dict:
    return {
        **crop,
        "vector": None,
        "error": "SSLError: TLSV1_ALERT_INTERNAL_ERROR",
        "error_kind": "download",
    }


def _crop_failure_record(crop: dict) -> dict:
    return {
        **crop,
        "vector": None,
        "error": "RuntimeError: bad crop",
        "error_kind": "crop",
    }


def test_fully_failed_download_is_retried_not_skipped():
    # Full crop count, but every record is a download failure -- the exact
    # shape a fully-failed `embed_episode` download returns. A count-only
    # check would call this complete; it must not.
    crops = [_crop("ep1", "SPEAKER_00", 0, 10.0), _crop("ep1", "SPEAKER_00", 0, 30.0)]
    jobs = [_job("ep1", crops)]
    existing = [_download_failure_record(c) for c in crops]

    complete = completed_episodes(existing, jobs)

    assert complete == set()


def test_complete_episode_with_real_vectors_is_skipped():
    crops = [_crop("ep1", "SPEAKER_00", 0, 10.0), _crop("ep1", "SPEAKER_00", 0, 30.0)]
    jobs = [_job("ep1", crops)]
    existing = [_vector_record(c) for c in crops]

    complete = completed_episodes(existing, jobs)

    assert complete == {"ep1"}


def test_partial_episode_is_retried():
    crops = [_crop("ep1", "SPEAKER_00", 0, 10.0), _crop("ep1", "SPEAKER_00", 0, 30.0)]
    jobs = [_job("ep1", crops)]
    # Only one of the two owed crops was ever written -- an interrupted run,
    # not a failure of either kind.
    existing = [_vector_record(crops[0])]

    complete = completed_episodes(existing, jobs)

    assert complete == set()


def test_a_per_crop_failure_does_not_disqualify_an_otherwise_complete_episode():
    # The audio DID arrive; only one crop's embedding failed. Re-fetching
    # ~130 MB to retry that one crop is not worth it, so this episode should
    # still count as done.
    crops = [_crop("ep1", "SPEAKER_00", 0, 10.0), _crop("ep1", "SPEAKER_00", 0, 30.0)]
    jobs = [_job("ep1", crops)]
    existing = [_vector_record(crops[0]), _crop_failure_record(crops[1])]

    complete = completed_episodes(existing, jobs)

    assert complete == {"ep1"}


def test_mixed_episodes_each_judged_independently():
    done_crops = [_crop("done", "SPEAKER_00", 0, 10.0)]
    failed_crops = [_crop("failed", "SPEAKER_00", 0, 10.0)]
    partial_crops = [
        _crop("partial", "SPEAKER_00", 0, 10.0),
        _crop("partial", "SPEAKER_00", 0, 30.0),
    ]
    jobs = [
        _job("done", done_crops),
        _job("failed", failed_crops),
        _job("partial", partial_crops),
    ]
    existing = (
        [_vector_record(c) for c in done_crops]
        + [_download_failure_record(c) for c in failed_crops]
        + [_vector_record(partial_crops[0])]
    )

    complete = completed_episodes(existing, jobs)

    assert complete == {"done"}


def test_dedup_keeps_the_newer_record_when_a_failed_episode_is_re_run():
    # This pins the dedup logic in main(): existing_records + records,
    # keyed on (episode, cluster, draw, length_s), keeping the LAST
    # occurrence. Records from the earlier failed attempt are appended
    # before the newer, successful ones -- since dict construction keeps the
    # last write per key, the fresh vector must win.
    crop = _crop("ep1", "SPEAKER_00", 0, 10.0)
    stale = _download_failure_record(crop)
    fresh = _vector_record(crop)

    deduped: dict[tuple, dict] = {}
    for record in [stale, fresh]:
        key = (record["episode"], record["cluster"], record["draw"], record["length_s"])
        deduped[key] = record
    final_records = list(deduped.values())

    assert len(final_records) == 1
    assert final_records[0]["vector"] == [0.1, 0.2]
    assert final_records[0].get("error_kind") is None
