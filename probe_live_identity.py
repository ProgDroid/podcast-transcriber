"""THROWAWAY PROBE -- does the identity path work BENEATH `embedder.embed`?

Everything above that call is proven. A reviewer's 433-episode replay showed
the archive and ingest paths agree GIVEN identical vectors, and
`probe_ingest_mounts.py` showed the container can derive its centroids. What
has never executed on real data is the layer underneath:

  * real `torchaudio` on a real mp3. `tests/test_embed_crop_window.py`
    patches a FAKE torchaudio into `sys.modules`, so the frame arithmetic
    where the sample-rate bug actually lived has never met a real file.
  * `load_embedder` in the TRANSCRIBE container -- a different image from the
    archive pass's, on a T4 rather than an L4.

That gap matters more today than it did when it was written down: as of
2026-09-07 `repair_identity` calls the same `identify_clusters` ->
`embedder.embed` chain, so the unverified layer has a second caller.

THE CONTROL THAT MAKES THIS MEAN ANYTHING. Comparing vectors, or comparing
the names that fall out of them, only shows that two computations agree. It
cannot show the seek landed where it was sent -- the original bug produced
"a perfectly plausible vector" from the wrong moment of audio. So this wraps
`torchaudio.load` to RECORD the `frame_offset` it is actually called with,
lets the real call proceed, and asserts the offset equals `start_s * native`.
Under the bug the same crop seeks to `start_s * 16000`, which on this episode
is 0.3628 of the intended timestamp. The assertion can genuinely fail.

WHY THIS EPISODE. 44.1 kHz is the MINORITY rate -- 17% of the archive,
measured from the local clip headers -- and it is the one the bug
under-exercised. Episode 261 is the newest 44.1 kHz episode carrying two
named clusters, so it is the least likely to have aged off the feed. Its
third cluster matched nobody, which exercises the legitimate-unmatched path
in the same run.

WHY THE SAVED TRANSCRIPT, NOT A FRESH DIARISATION. Diarisation is not
deterministic. Re-diarising yields different SPEAKER_XX labels than the ones
in `speakers/assignments.json`, and you cannot both re-diarise and compare --
the comparison would be to a different world.

PRE-REGISTERED, before running:

  * native sample rate                 == 44100
  * clusters in the crop plan          == 3
  * crops                              == 14  (6 / 2 / 6 -- measured
                                          locally from the saved
                                          transcript, not estimated)
  * discriminating offsets             == 14 of 14 (no crop starts at 0,
                                          where both arithmetics agree)
  * identified                         == exactly 2: Jacob Shapiro, Rob Larity
  * the third cluster                  == ABSENT from identified
  * frame_offset on every crop         == int(start_s * 44100)
  * frame_offset under the old bug     == int(start_s * 16000)  (0.3628x)
  * records written to Chroma          == 0   (it holds no collection handle)

A result that merely says "2 clusters named correctly" WITHOUT the offset
assertion is NOT a pass -- that is the shape the original bug already
survived once.

    uv run modal run probe_live_identity.py

RESULT, 2026-09-07: PASS on every pre-registered number.

  native sample rate      44100          (as predicted -- the minority rate)
  episode duration        2080.2s        (crops at 927s / 1103s / 1324s /
                                          1619s are deep in the file, not
                                          clustered at the start)
  segments / turns        335 / 34       (0 dropped)
  crops                   14             6 / 2 / 6, matching the local
                                          pre-computation exactly
  embedder_loaded         True           load_embedder WORKS in the
                                          transcribe image on a T4
  identified              SPEAKER_00 -> Jacob Shapiro
                          SPEAKER_02 -> Rob Larity
                          SPEAKER_01 absent -- matched nobody, as recorded
  torchaudio.load calls   14 of 14 observed -- the control saw every seek
  offsets correct         14 / 14
  offsets matching bug     0 / 14
  discriminating offsets  14 / 14        no crop starts at 0, so every single
                                          one could have caught the bug
  image rebuild lines      0             the probed chain IS the deployed one

The layer beneath `embedder.embed` is verified: real torchaudio, a real
44.1 kHz mp3, a real seek, in the real transcribe container. Because
`repair_identity` reaches `embedder.embed` through the same
`identify_clusters`, this covers the repair path too.

WHAT IT DOES NOT COVER, stated so nobody reads it as more than it is: one
episode, at one rate, in one container. The 48 kHz majority path is not
re-verified here -- though the arithmetic is rate-generic and the archive
pass already exercised it -- and nothing here says anything about
whisper, diarisation, or the write path.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal
from transcribe import (
    ENROLMENT_LABELS,
    ENROLMENT_VECTORS,
    MODEL_PATH,
    VOLUME_PATH,
    image,
    model_cache,
    volume,
)

# The probe imports transcribe for the REAL image object, so the container
# needs that module too -- transcribe's own image mounts only `corpus`.
probe_image = image.add_local_python_source("transcribe")

app = modal.App("probe-live-identity")

SHOW = "The Jacob Shapiro Podcast"
FEED = "https://feeds.captivate.fm/jacob-shapiro/"
EPISODE = "261"
DATE = "2025-02-28"

EXPECT_RATE = 44100
EXPECT_CLUSTERS = 3
# MEASURED locally from the saved transcript before running, not estimated.
# The spec's "6 per cluster" is the CAP, not the count: select_crop_turns
# gates on turn duration, and SPEAKER_01 is a minor voice with only two
# turns long enough to crop -- consistent with it being the cluster that
# matched nobody.
EXPECT_CROPS = 14
EXPECT_CROPS_BY_CLUSTER = {"SPEAKER_00": 6, "SPEAKER_01": 2, "SPEAKER_02": 6}
BUGGY_RATE = 16000  # what the arithmetic used before 68231ba


@app.function(
    image=probe_image,
    gpu="T4",
    timeout=1800,
    volumes={VOLUME_PATH: volume, MODEL_PATH: model_cache},
    secrets=[modal.Secret.from_name("podcast-secrets")],
)
def probe(expected_people: list[str]) -> dict:
    import os

    from transcribe import _download_audio, _transcript_path, parse_all_episodes

    from corpus.chunking import parse_transcript_segments
    from corpus.embedding import cluster_crop_plan
    from corpus.enrolment import MARGIN, THRESHOLD, build_centroids
    from corpus.identify import identify_clusters, usable_segments
    from corpus.speakers import CLIP_MIN_TURN_S, GAP_CAP_S, merge_turns
    from corpus.voice import COMMUNITY, load_embedder

    out: dict = {"episode": f"{SHOW} - Episode {EPISODE} - {DATE}"}
    volume.reload()

    # 1. Is the episode still in the feed? Done HERE, on the real path, so a
    #    missing enclosure reads as the feed fact it is rather than as a probe
    #    failure.
    episodes = parse_all_episodes(FEED)
    out["feed_episodes"] = len(episodes)
    match = [
        e for e in episodes if e["episode_number"] == EPISODE and e["date"] == DATE
    ]
    if not match:
        out["status"] = "AGED_OFF_FEED"
        out["note"] = (
            "the episode is no longer in the feed, so there is no enclosure "
            "URL to download. This is a fact about the feed, not a result. "
            "Pick the next 44.1kHz candidate."
        )
        return out
    episode = match[0]

    # 2. The SAVED transcript -- its labels are the ones assignments.json used.
    path = _transcript_path(SHOW, episode)
    if not os.path.exists(path):
        out["status"] = "NO_TRANSCRIPT_ON_VOLUME"
        out["path"] = path
        return out
    with open(path, encoding="utf-8", errors="replace") as handle:
        segments = parse_transcript_segments(handle.read())
    out["segments"] = len(segments)

    # 3. The crop plan, computed the way identify_clusters computes it, so the
    #    recorded loads can be matched to the crops that caused them.
    kept, dropped = usable_segments(segments)
    turns = merge_turns(kept, gap_cap_s=GAP_CAP_S)
    crops = cluster_crop_plan(
        {"name": f"{SHOW} - Episode {EPISODE}", "turns": turns},
        min_turn_s=CLIP_MIN_TURN_S,
    )
    out["dropped_segments"] = dropped
    out["turns"] = len(turns)
    out["crops"] = len(crops)
    out["crops_by_cluster"] = {
        cluster: sum(1 for c in crops if c["cluster"] == cluster)
        for cluster in sorted({c["cluster"] for c in crops})
    }

    audio_path = "/tmp/probe_261.mp3"
    _download_audio(episode["audio_url"], audio_path)

    import torchaudio

    info = torchaudio.info(audio_path)
    native = int(info.sample_rate)
    out["native_sample_rate"] = native
    out["duration_s"] = round(info.num_frames / native, 1)

    # 4. THE CONTROL. Record what frame_offset the real load is called with,
    #    then let the real call proceed. `corpus/voice.py` does `import
    #    torchaudio` inside embed() and resolves `.load` at call time, so
    #    rebinding the attribute here is seen by the code under test.
    real_load = torchaudio.load
    seen: list[dict] = []

    def recording_load(path, frame_offset=0, num_frames=-1, **kwargs):
        seen.append({"frame_offset": frame_offset, "num_frames": num_frames})
        return real_load(
            path, frame_offset=frame_offset, num_frames=num_frames, **kwargs
        )

    torchaudio.load = recording_load
    try:
        labels = json.loads(Path(ENROLMENT_LABELS).read_text())
        vectors = json.loads(Path(ENROLMENT_VECTORS).read_text())
        enrolment = build_centroids(labels, vectors)
        out["enrolled"] = sorted(enrolment)

        embedder = load_embedder(
            COMMUNITY, os.environ["HF_TOKEN"], model_home=MODEL_PATH
        )
        out["embedder_loaded"] = True

        identified = identify_clusters(
            segments,
            audio_path,
            embedder,
            enrolment,
            threshold=THRESHOLD,
            margin=MARGIN,
            episode_name=f"{SHOW} - Episode {EPISODE}",
        )
    finally:
        torchaudio.load = real_load
        if os.path.exists(audio_path):
            os.remove(audio_path)

    out["identified"] = dict(sorted(identified.items()))
    out["people"] = sorted(set(identified.values()))
    out["loads_recorded"] = len(seen)

    # 5. The offsets, crop by crop. `identify_clusters` embeds in crop order,
    #    so seen[i] is the load caused by crops[i].
    checked = []
    mismatched = []
    would_have_matched_buggy = 0
    for crop, call in zip(crops, seen, strict=False):
        want = int(crop["crop_start_s"] * native)
        buggy = int(crop["crop_start_s"] * BUGGY_RATE)
        row = {
            "cluster": crop["cluster"],
            "start_s": round(crop["crop_start_s"], 2),
            "observed_offset": call["frame_offset"],
            "expected_offset": want,
            "buggy_offset": buggy,
            "ok": call["frame_offset"] == want,
        }
        if not row["ok"]:
            mismatched.append(row)
        if call["frame_offset"] == buggy and buggy != want:
            would_have_matched_buggy += 1
        checked.append(row)

    out["offsets_checked"] = len(checked)
    out["offsets_ok"] = sum(1 for r in checked if r["ok"])
    out["offsets_mismatched"] = mismatched[:5]
    out["offsets_matching_the_old_bug"] = would_have_matched_buggy
    # A crop at start_s == 0 seeks to 0 under BOTH the correct and the buggy
    # arithmetic, so it cannot discriminate. Count the ones that can.
    out["discriminating_offsets"] = sum(
        1 for r in checked if r["expected_offset"] != r["buggy_offset"] and r["ok"]
    )

    # 6. The verdict, computed here rather than eyeballed later.
    problems = []
    if native != EXPECT_RATE:
        problems.append(f"native rate {native} != {EXPECT_RATE}")
    if len(out["crops_by_cluster"]) != EXPECT_CLUSTERS:
        problems.append(
            f"{len(out['crops_by_cluster'])} clusters in the plan != {EXPECT_CLUSTERS}"
        )
    if out["crops_by_cluster"] != EXPECT_CROPS_BY_CLUSTER:
        problems.append(
            f"crop plan {out['crops_by_cluster']} != {EXPECT_CROPS_BY_CLUSTER}"
        )
    if out["crops"] != EXPECT_CROPS:
        problems.append(f"{out['crops']} crops != {EXPECT_CROPS}")
    if out["loads_recorded"] != EXPECT_CROPS:
        problems.append(
            f"{out['loads_recorded']} torchaudio.load calls recorded != "
            f"{EXPECT_CROPS} crops -- the control did not observe every seek"
        )
    if out["people"] != sorted(expected_people):
        problems.append(f"identified {out['people']} != {sorted(expected_people)}")
    if mismatched:
        problems.append(f"{len(mismatched)} crop(s) seeked to the wrong offset")
    if out["discriminating_offsets"] == 0:
        problems.append(
            "NO crop had an offset that distinguishes correct from buggy "
            "arithmetic -- this run proves nothing about the seek"
        )
    out["problems"] = problems
    out["status"] = "PASS" if not problems else "FAIL"
    return out


@app.local_entrypoint()
def main() -> None:
    rows = json.loads(Path("speakers/assignments.json").read_text(encoding="utf-8"))[
        "assignments"
    ]
    mine = [
        r
        for r in rows
        if r["show"] == SHOW and r["episode_number"] == EPISODE and r["date"] == DATE
    ]
    people = sorted({r["person"] for r in mine if r.get("person")})
    print(f"assignments.json records {len(mine)} cluster(s) for this episode")
    print(f"  named   : {people}")
    print(f"  unnamed : {sum(1 for r in mine if not r.get('person'))}")
    print("\n--- PRE-REGISTERED ---")
    print(f"  native rate      : {EXPECT_RATE}")
    print(f"  clusters in plan : {EXPECT_CLUSTERS}")
    print(f"  crops            : {EXPECT_CROPS} {EXPECT_CROPS_BY_CLUSTER}")
    print("  discriminating   : 14 of 14 (no crop starts at 0)")
    print(f"  identified       : {people}")
    print("  every crop offset: int(start_s * 44100), never int(start_s * 16000)")
    print("\nrunning...\n")

    result = probe.remote(people)
    print(json.dumps(result, indent=2))
    print(f"\nSTATUS: {result.get('status')}")
    for problem in result.get("problems", []):
        print(f"  PROBLEM: {problem}")
