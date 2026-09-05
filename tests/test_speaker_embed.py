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

Round 4 closed the interaction between those two: the `.jsonl` is appended
to, so a successful retry leaves the failed attempt's records in the file.
Judging RAW records, an episode that was retried and succeeded shows 2N
records against a plan of N and still carries the stale download-failure
flag -- excluded from the skip set forever, re-downloading ~130 MB on every
later `--resume`. `completed_episodes` now judges `dedupe_records`, the same
helper the final `.json` is built with, so the two cannot drift apart.

`speaker_embed.py` imports `modal` at module scope; `modal` is a dev
dependency already, so importing it here to reach `completed_episodes` is
fine and does not touch the network or Modal's control plane.
"""

from __future__ import annotations

import inspect

import speaker_embed
from speaker_embed import completed_episodes, dedupe_records


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


def test_dedupe_records_keeps_the_last_occurrence_for_a_repeated_key():
    # The rule the final `.json` is built with: records from the earlier
    # failed attempt are appended before the newer, successful ones, so the
    # fresh vector must win.
    crop = _crop("ep1", "SPEAKER_00", 0, 10.0)
    stale = _download_failure_record(crop)
    fresh = _vector_record(crop)

    final_records = dedupe_records([stale, fresh])

    assert len(final_records) == 1
    assert final_records[0]["vector"] == [0.1, 0.2]
    assert final_records[0].get("error_kind") is None


def test_dedupe_records_keys_on_the_full_crop_identity():
    # Same episode and cluster, different draw and different window length --
    # four distinct crops, none of which may collapse into another.
    crops = [
        _crop("ep1", "SPEAKER_00", 0, 10.0),
        _crop("ep1", "SPEAKER_00", 0, 30.0),
        _crop("ep1", "SPEAKER_00", 1, 10.0),
        _crop("ep1", "SPEAKER_01", 0, 10.0),
    ]

    assert len(dedupe_records([_vector_record(c) for c in crops])) == 4


def test_a_retried_episode_is_complete_despite_its_stale_failure_records():
    # The three-run sequence `--resume` exists for.
    #   Run 1: the download fails -- N records, all `vector: None`,
    #          `error_kind == "download"`.
    #   Run 2: `--resume` correctly retries the episode and it succeeds --
    #          N good records are APPENDED, so the file now holds 2N.
    #   Run 3: this call. Judging raw records, `counts["ep1"]` is 2N (never
    #          equal to N) and the Run-1 download-failure flag is still set,
    #          so the episode would be excluded FOREVER and re-fetched on
    #          every later resume. De-duplicated, it holds N good records
    #          and no download failure, so it is done.
    crops = [_crop("ep1", "SPEAKER_00", 0, 10.0), _crop("ep1", "SPEAKER_00", 0, 30.0)]
    jobs = [_job("ep1", crops)]
    existing = [_download_failure_record(c) for c in crops] + [
        _vector_record(c) for c in crops
    ]
    assert len(existing) == 2 * len(crops)

    complete = completed_episodes(existing, jobs)

    assert complete == {"ep1"}


def test_a_retried_episode_that_failed_AGAIN_is_still_retried():
    # Same append-only shape, but the second attempt's download also failed.
    # Dedup must not paper over that -- the newest records still say
    # "download", so the episode stays in `remaining`.
    crops = [_crop("ep1", "SPEAKER_00", 0, 10.0), _crop("ep1", "SPEAKER_00", 0, 30.0)]
    jobs = [_job("ep1", crops)]
    existing = [_download_failure_record(c) for c in crops] * 2

    complete = completed_episodes(existing, jobs)

    assert complete == set()


def test_both_call_sites_go_through_the_one_dedup_helper():
    # By construction, not by coincidence: the skip set and the final `.json`
    # must collapse duplicate records with the SAME rule. Two hand-maintained
    # copies drifting apart is precisely the defect this round fixed, so pin
    # that neither call site re-implements it.
    judge = inspect.getsource(speaker_embed.completed_episodes)
    # `main` is wrapped into a Modal LocalEntrypoint, which `getsource`
    # cannot read, so slice it out of the module source instead.
    module_src = inspect.getsource(speaker_embed)
    writer = module_src[module_src.index("def main(") :]

    assert "dedupe_records(" in judge
    assert "dedupe_records(" in writer
    # No open-coded second copy of the keying rule at either site.
    assert 'record["cluster"]' not in judge
    assert 'record["cluster"]' not in writer


# --- embed_every_clip -------------------------------------------------------
#
# Filed as a minor while this output only fed a one-off bake-off: it raised
# only when NOTHING embedded, so 380 of 381 returned quietly. It stopped
# being minor on 2026-09-04, when speakers/clip_vectors_<embedder>.json
# became a MOUNTED INPUT to the nightly ingest -- the five enrolment
# centroids are rebuilt from it on every run, so a partial regeneration would
# narrow the evidence behind each voice and shift live assignments with
# nothing raising.


class _StubEmbedder:
    """Fails for any clip whose stem is in `bad`; no audio is touched."""

    def __init__(self, bad: set[str] | None = None) -> None:
        self.bad = bad or set()
        self.seen: list[str] = []

    def embed(self, path: str, start_s: float, length_s: float) -> list[float]:
        from pathlib import Path as _P

        stem = _P(path).stem
        self.seen.append(stem)
        if stem in self.bad:
            raise OSError("decode failed")
        return [0.1, 0.2]


def _paths(*stems: str):
    from pathlib import Path as _P

    return [_P(f"/clips/{s}.mp3") for s in stems]


def test_every_clip_embedding_returns_them_all():
    from speaker_embed import embed_every_clip

    out = embed_every_clip(_paths("a", "b", "c"), _StubEmbedder())

    assert sorted(out) == ["a", "b", "c"]


def test_one_failed_clip_raises_rather_than_returning_a_partial_basis():
    """The whole point: 2 of 3 must not come back looking like a result."""
    import pytest

    from speaker_embed import embed_every_clip

    with pytest.raises(RuntimeError, match="1 of 3 clips failed"):
        embed_every_clip(_paths("a", "b", "c"), _StubEmbedder(bad={"b"}))


def test_the_failing_clip_is_named():
    """A count alone does not tell an operator which file to look at."""
    import pytest

    from speaker_embed import embed_every_clip

    with pytest.raises(RuntimeError, match="b: OSError"):
        embed_every_clip(_paths("a", "b", "c"), _StubEmbedder(bad={"b"}))


def test_an_empty_clips_mount_raises():
    """clips=0 is a broken image, not an empty evaluation set."""
    import pytest

    from speaker_embed import embed_every_clip

    with pytest.raises(RuntimeError, match="broken image"):
        embed_every_clip([], _StubEmbedder())
