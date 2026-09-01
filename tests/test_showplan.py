"""The show planning loop, which had no test at all while it lived on the GPU.

Its failure-isolation behaviour was asserted only by a comment. These are the
tests that comment was standing in for.
"""

from corpus.showplan import plan_show
from corpus.store import batched

SHOW = "Geopolitical Cousins"

TRANSCRIPT = """# Geopolitical Cousins - Episode 73

[SPEAKER_00] 1.0s - One.
[SPEAKER_01] 2.0s - Two.
"""


def _episode(number, date, **extra):
    ep = {
        "episode_number": number,
        "date": date,
        "title": f"Episode {number}",
        "audio_url": f"https://example.invalid/{number}.mp3",
    }
    ep.update(extra)
    return ep


def _seed(collection, number, date, n, *, n_chunks, rules_version="1"):
    prefix = f"Geopolitical_Cousins-ep{number}-{date}"
    for batch in batched(list(range(n))):
        collection.upsert(
            ids=[f"{prefix}-{i}" for i in batch],
            documents=[f"d{i}" for i in batch],
            metadatas=[
                {
                    "show": SHOW,
                    "episode_number": number,
                    "date": date,
                    "n_chunks": n_chunks,
                    "rules_version": rules_version,
                }
                for _ in batch
            ],
        )


def _plan(collection, episodes, transcripts):
    """transcripts maps episode_number -> text, or omits it for "no file"."""
    return plan_show(
        collection,
        show=SHOW,
        episodes=episodes,
        read_transcript=lambda e: transcripts.get(e["episode_number"]),
        log=lambda _msg: None,
    )


def test_each_action_lands_in_the_right_bucket(collection):
    # 73 complete -> neither list. 74 torn -> re-embed. 75 no transcript ->
    # transcribe. All three in one pass, because the bug this guards against
    # is an episode being routed to the wrong container, not a wrong verdict.
    _seed(collection, "73", "2026-07-29", 2, n_chunks=2)
    _seed(collection, "74", "2026-07-31", 1, n_chunks=2)
    episodes = [
        _episode("73", "2026-07-29"),
        _episode("74", "2026-07-31"),
        _episode("75", "2026-08-02"),
    ]
    plan = _plan(collection, episodes, {"73": TRANSCRIPT, "74": TRANSCRIPT})

    assert [e["episode_number"] for e in plan.to_embed] == ["74"]
    assert [e["episode_number"] for e in plan.to_transcribe] == ["75"]
    assert plan.failures == []
    assert plan.has_work


def test_nothing_to_do_reports_no_work(collection):
    # The common nightly case, and the one the whole split exists to make
    # cheap: if this ever reports work, a GPU container starts every night
    # for nothing.
    _seed(collection, "73", "2026-07-29", 2, n_chunks=2)
    plan = _plan(collection, [_episode("73", "2026-07-29")], {"73": TRANSCRIPT})

    assert not plan.has_work
    assert plan.to_embed == [] and plan.to_transcribe == []


def test_one_episode_failing_does_not_abort_the_show(collection):
    # The regression that arrived with the feature: plan_episode can raise
    # where the os.path.exists it replaced could not, and the call sat
    # outside any try -- so one transient Chroma error dropped an entire
    # show's work before any of it started.
    _seed(collection, "75", "2026-08-02", 1, n_chunks=2)
    episodes = [
        _episode("73", "2026-07-29"),
        _episode("74", "2026-07-31"),
        _episode("75", "2026-08-02"),
    ]

    def read_transcript(episode):
        if episode["episode_number"] == "74":
            raise RuntimeError("volume read failed")
        return TRANSCRIPT

    plan = plan_show(
        collection,
        show=SHOW,
        episodes=episodes,
        read_transcript=read_transcript,
        log=lambda _msg: None,
    )

    # 73 and 75 were still planned; only 74 was lost.
    assert [e["episode_number"] for e in plan.to_embed] == ["73", "75"]
    assert len(plan.failures) == 1
    assert "ep74" in plan.failures[0]
    assert "volume read failed" in plan.failures[0]


def test_a_failure_is_reported_even_when_no_work_survives(collection):
    # A show whose every episode failed to plan must not look like a quiet
    # "nothing to do" night. has_work is False here, so the caller skips the
    # GPU -- but the failure list is what makes the run loudly unsuccessful.
    def read_transcript(_episode):
        raise RuntimeError("chroma unavailable")

    plan = plan_show(
        collection,
        show=SHOW,
        episodes=[_episode("73", "2026-07-29")],
        read_transcript=read_transcript,
        log=lambda _msg: None,
    )

    assert not plan.has_work
    assert len(plan.failures) == 1


def test_excluded_episodes_are_neither_transcribed_nor_embedded(collection):
    # A cross-post has a transcript on the volume and no records in Chroma,
    # which is the exact shape of an EMBED_ONLY. Only the exclusion keeps it
    # out, and getting this wrong re-embeds a deliberately deleted episode
    # every night.
    episode = _episode(
        "Unknown", "2026-07-29", guid="1c45dbd9-0dc3-4d07-b2d1-758fe78405fe"
    )
    plan = plan_show(
        collection,
        show="The Jacob Shapiro Podcast",
        episodes=[episode],
        read_transcript=lambda _e: TRANSCRIPT,
        log=lambda _msg: None,
    )

    assert not plan.has_work
    assert plan.failures == []


def test_the_guid_reaches_the_exclusion_check(collection):
    # The guid arm is unreachable unless the loop forwards episode["guid"].
    # Episode number 999 appears in no exclusion entry, so the TRIPLE arm
    # cannot match here -- this is the six-month threat model, an episode
    # whose number Captivate backfilled after the exclusion was written. Only
    # the guid still identifies it. Drop the forward and it plans as work.
    episode = _episode("999", "2025-04-04", guid="c4af95bf-cfbc-4c0a-b4d7-2c2df77d1fe6")
    plan = plan_show(
        collection,
        show="The Jacob Shapiro Podcast",
        episodes=[episode],
        read_transcript=lambda _e: TRANSCRIPT,
        log=lambda _msg: None,
    )

    assert not plan.has_work


def test_feed_order_is_preserved(collection):
    episodes = [_episode(str(n), f"2026-08-{n:02d}") for n in (5, 3, 9, 1)]
    plan = _plan(collection, episodes, {})

    assert [e["episode_number"] for e in plan.to_transcribe] == ["5", "3", "9", "1"]
