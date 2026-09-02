"""Cut and label the clips the Tier 1 precision gate is measured on.

Three steps, deliberately separate files apart rather than one command:

    uv run python speaker_tool.py plan          # -> speakers/clip_plan.json
    uv run modal run speaker_tool.py::cut_clips # -> clips/
    uv run python speaker_tool.py label         # -> speakers/labels.json

**The plan is written before any audio is touched, and that is the point.**
Every draw, seed and clip window is in a file you can read, diff and re-run
before a single byte is downloaded. A selection that only exists inside the
run that used it cannot be audited afterwards, which is the failure this
whole line of work was opened by.

`plan` and `label` need no Modal account and no network. Only `cut_clips`
does, and it runs a minimal ffmpeg image rather than replicating the
transcription image: it needs no model stack, and the whisperx image would
be a large cold start for a job that only cuts audio. That is the opposite
of the advice in the spec's Part 7 §6, which applies to Tier 2 work that
must reuse the cached model layers -- this is not that.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - exercised by env, not tests
    # `plan` and `label` are local, offline steps, and CI is specified as
    # "no GPU, no secrets, no Modal". Importing this module must therefore
    # work without modal installed; only `cut_clips` actually needs it, and
    # calling it without modal fails loudly at the point of use rather than
    # making the whole tool unimportable.
    modal = None  # type: ignore[assignment]

from corpus.identity import parse_transcript_filename
from corpus.speakers import (
    CLIP_LEAD_IN_S,
    CLIP_LENGTH_S,
    CLIP_MIN_TURN_S,
    GAP_CAP_S,
    clip_window,
    evenly_spaced,
    pick_clip_turn,
    routes_to_tier2,
)
from corpus.transcripts import dominant_speaker, load_episodes

FEEDS = {
    "Geopolitical Cousins": "https://feeds.captivate.fm/geopolitical-cousins/",
    "The Jacob Shapiro Podcast": "https://feeds.captivate.fm/jacob-shapiro/",
    "The Observing Japan Podcast": (
        "https://api.substack.com/feed/podcast/868206/s/386602.rss"
    ),
}

# The 2026 stratum is taken whole (§3.3), so only the pre-2026 stratum has a
# target: it is sampled from a pool several times its size.
PRE_2026_TARGET = 150
ERA_BOUNDARY = "2026"

PLAN_PATH = Path("speakers/clip_plan.json")
LABELS_PATH = Path("speakers/labels.json")
CLIPS_DIR = Path("clips")


class _NoModal:
    """Stands in for the app when modal is absent, so decorators still bind.

    A decorated function keeps its plain-Python identity and simply is not
    remote. Calling one then fails where it tries to reach Modal, which is
    the honest place for it to fail.
    """

    def function(self, **_kwargs):
        return lambda fn: fn

    def local_entrypoint(self, **_kwargs):
        return lambda fn: fn


if modal is None:
    app = _NoModal()
    image = None
else:
    app = modal.App("speaker-clip-tool")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg")
        .pip_install("feedparser==6.0.11", "requests==2.32.3")
        # Stays last, so a change above does not invalidate it.
        .add_local_python_source("corpus")
    )


# ---------------------------------------------------------------- plan


def build_plan(directory: Path, *, campaign_seed: str) -> tuple[list[dict], dict]:
    """Choose the episodes and the exact clip windows, and explain nothing away.

    Tier 1 makes one assignment per episode -- the dominant cluster -- so the
    precision set is one clip per episode. Eligibility is §3.3's: single-host
    shows only, no co-host surname anywhere in the transcript, and a dominant
    cluster with a turn long enough to clip.
    """
    episodes = load_episodes(directory, cap=GAP_CAP_S)
    eligible: list[dict] = []
    routed = 0
    unclippable = 0
    for episode in episodes:
        if routes_to_tier2(episode["show"], episode["text"]):
            routed += 1
            continue
        speaker = dominant_speaker(episode)
        if speaker is None:
            continue
        seed = f"{campaign_seed}:{episode['name']}"
        turn = pick_clip_turn(
            episode["turns"], speaker, seed=seed, min_turn_s=CLIP_MIN_TURN_S
        )
        if turn is None:
            # Measured at 0 of 439 today. If it ever fires, the stratum's n
            # falls and §3.4's bound with it -- so it is dropped loudly.
            print(f"  no clippable turn, dropped: {episode['name']}")
            unclippable += 1
            continue
        eligible.append({"episode": episode, "speaker": speaker, "seed": seed})

    pre = sorted(
        (e for e in eligible if e["episode"]["year"] < ERA_BOUNDARY),
        key=lambda e: (e["episode"]["show"], e["episode"]["date"]),
    )
    recent = sorted(
        (e for e in eligible if e["episode"]["year"] >= ERA_BOUNDARY),
        key=lambda e: (e["episode"]["show"], e["episode"]["date"]),
    )
    selected_pre = evenly_spaced(pre, PRE_2026_TARGET)
    chosen = selected_pre + recent
    meta = {
        "pre_pool": len(pre),
        "pre_selected": len(selected_pre),
        # Recorded because §3.3 requires it: without the stride and the pool
        # size, "every nth by date" is not a reproducible instruction.
        "pre_stride": (len(pre) / len(selected_pre)) if selected_pre else 0.0,
        "recent_pool": len(recent),
        "recent_selected": len(recent),
        "routed_to_tier2": routed,
        "no_clippable_turn": unclippable,
        "era_boundary": ERA_BOUNDARY,
        "gap_cap_s": GAP_CAP_S,
        "clip_min_turn_s": CLIP_MIN_TURN_S,
    }

    plan = []
    for item in chosen:
        episode = item["episode"]
        turn = pick_clip_turn(
            episode["turns"],
            item["speaker"],
            seed=item["seed"],
            min_turn_s=CLIP_MIN_TURN_S,
        )
        start, length = clip_window(
            turn, lead_in_s=CLIP_LEAD_IN_S, length_s=CLIP_LENGTH_S
        )
        parsed = parse_transcript_filename(episode["name"])
        plan.append(
            {
                "clip_id": f"{Path(episode['name']).stem}--{item['speaker']}--d0",
                "episode": episode["name"],
                "show": episode["show"],
                "episode_number": parsed[1] if parsed else "Unknown",
                "date": episode["date"],
                "stratum": (
                    ERA_BOUNDARY
                    if episode["year"] >= ERA_BOUNDARY
                    else f"pre-{ERA_BOUNDARY}"
                ),
                "purpose": "precision",
                "cluster": item["speaker"],
                "seed": item["seed"],
                "draw": 0,
                "turn_start": turn["start"],
                "turn_duration_s": turn["duration_s"],
                "clip_start_s": start,
                "clip_length_s": length,
            }
        )
    return plan, meta


def cmd_plan(args: argparse.Namespace) -> None:
    plan, meta = build_plan(args.dir, campaign_seed=args.seed)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        PLAN_PATH, {"campaign_seed": args.seed, "sampling": meta, "clips": plan}
    )
    by_stratum: dict[str, int] = {}
    for entry in plan:
        by_stratum[entry["stratum"]] = by_stratum.get(entry["stratum"], 0) + 1
    print(f"\nwrote {PLAN_PATH} — {len(plan)} clips")
    for stratum, count in sorted(by_stratum.items()):
        print(f"  {stratum:<12} {count}")
    print(
        f"\n  pre-{ERA_BOUNDARY} pool {meta['pre_pool']} -> {meta['pre_selected']} "
        f"at stride {meta['pre_stride']:.3f}"
    )
    print(
        f"  {ERA_BOUNDARY} pool {meta['recent_pool']} -> {meta['recent_selected']} "
        f"(exhaustive, no sampling)"
    )
    print(f"  dropped, routed to Tier 2: {meta['routed_to_tier2']}")
    print(f"  dropped, no clippable turn: {meta['no_clippable_turn']}")


# ---------------------------------------------------------------- cut


@app.function(image=image, timeout=600)
def resolve_audio_urls(feed_url: str, show_name: str) -> dict[str, str]:
    """Map "<episode_number>|<date>" to its enclosure URL, from the feed.

    Keyed on the same (episode, date) the transcript filenames were built
    from, via the same `episode_number_of`, so a filename resolves back to
    its audio without a second derivation that could disagree.
    """
    import email.utils

    import feedparser

    from corpus.feed import episode_number_of

    parsed = feedparser.parse(feed_url)
    urls: dict[str, str] = {}
    for entry in parsed.entries:
        enclosures = entry.get("enclosures") or []
        if not enclosures:
            continue
        published = entry.get("published")
        if not published:
            continue
        try:
            date_str = email.utils.parsedate_to_datetime(published).strftime("%Y-%m-%d")
        except Exception:
            continue
        urls[f"{episode_number_of(entry)}|{date_str}"] = enclosures[0].get("url", "")
    print(f"{show_name}: {len(urls)} entries with audio")
    return urls


@app.function(image=image, timeout=1800)
def cut_one_episode(job: dict) -> list[dict]:
    """Download one episode's audio once and cut every clip it owes."""
    import requests

    results: list[dict] = []
    audio_path = "/tmp/episode.mp3"
    try:
        with requests.get(job["audio_url"], stream=True, timeout=300) as response:
            response.raise_for_status()
            with open(audio_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
    except Exception as exc:
        # Audio rot is expected: six episodes have aged off their feeds and
        # can never be clipped. A failure here reduces the stratum's n and
        # must be recorded as such, never silently skipped.
        return [
            {"clip_id": c["clip_id"], "error": f"{type(exc).__name__}: {exc}"}
            for c in job["clips"]
        ]

    for clip in job["clips"]:
        out = f"/tmp/{clip['clip_id']}.mp3"
        command = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(clip["clip_start_s"]),
            "-i",
            audio_path,
            "-t",
            str(clip["clip_length_s"]),
            "-ac",
            "1",
            out,
        ]
        proc = subprocess.run(command, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(out):
            results.append(
                {
                    "clip_id": clip["clip_id"],
                    "error": f"ffmpeg exit {proc.returncode}: "
                    f"{proc.stderr.decode('utf-8', 'replace')[:200]}",
                }
            )
            continue
        with open(out, "rb") as handle:
            results.append({"clip_id": clip["clip_id"], "audio": handle.read()})
    return results


@app.local_entrypoint()
def cut_clips(plan_path: str = str(PLAN_PATH), out_dir: str = str(CLIPS_DIR)) -> None:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    clips = plan["clips"]
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    shows = sorted({c["show"] for c in clips})
    url_maps = dict(
        zip(
            shows,
            resolve_audio_urls.starmap([(FEEDS[s], s) for s in shows]),
            strict=True,
        )
    )

    jobs: list[dict] = []
    unresolved: list[str] = []
    for clip in clips:
        if (destination / f"{clip['clip_id']}.mp3").exists():
            continue
        url = url_maps[clip["show"]].get(f"{clip['episode_number']}|{clip['date']}")
        if not url:
            unresolved.append(clip["clip_id"])
            continue
        jobs.append({"episode": clip["episode"], "audio_url": url, "clips": [clip]})

    if unresolved:
        print(f"{len(unresolved)} clips have no audio URL in their feed:")
        for clip_id in unresolved[:10]:
            print(f"  {clip_id}")

    written = 0
    failures: list[dict] = []
    for batch in cut_one_episode.map(jobs):
        for result in batch:
            if "error" in result:
                failures.append(result)
                continue
            (destination / f"{result['clip_id']}.mp3").write_bytes(result["audio"])
            written += 1

    print(f"\nwrote {written} clips to {destination}")
    if failures or unresolved:
        report = destination / "unavailable.json"
        _write_json(report, {"unresolved": unresolved, "failed": failures})
        print(
            f"{len(failures) + len(unresolved)} clips unavailable, recorded in "
            f"{report}. These REDUCE the stratum's n (§3.4); they are not "
            f"replaced silently."
        )


# ---------------------------------------------------------------- label


def _play(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 - a local file this tool just wrote
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.run([opener, str(path)], check=False)
    except Exception as exc:
        print(f"  (could not autoplay: {exc})")


def _write_json(path: Path, payload: object) -> None:
    """Write via a temp file and replace, so an interrupt cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def cmd_label(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    labels = (
        json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        if LABELS_PATH.exists()
        else {}
    )
    clips_dir = Path(args.clips)

    pending = [
        c
        for c in plan["clips"]
        if c["clip_id"] not in labels and (clips_dir / f"{c['clip_id']}.mp3").exists()
    ]
    print(f"{len(labels)} labelled, {len(pending)} to go.\n")
    print("Type a name, or a number from the list. Commands: ? skip, u undo, q quit.\n")

    history: list[str] = []
    for clip in pending:
        names = sorted({v["person"] for v in labels.values() if v.get("person")})
        path = clips_dir / f"{clip['clip_id']}.mp3"
        print(f"--- {clip['show']} — {clip['date']} ({clip['stratum']})")
        print(
            f"    cluster {clip['cluster']}, {clip['clip_length_s']:.0f}s from "
            f"{clip['clip_start_s']:.0f}s   draw {clip['draw']}"
        )
        print(f"    {path}")
        for index, name in enumerate(names, 1):
            print(f"      {index}. {name}")
        _play(path)

        answer = input("  who is this? ").strip()
        if answer == "q":
            break
        if answer == "u" and history:
            labels.pop(history.pop(), None)
            _write_json(LABELS_PATH, labels)
            print("  undone.\n")
            continue
        if answer in {"?", ""}:
            labels[clip["clip_id"]] = {"person": None, "basis": "skipped"}
        else:
            if answer.isdigit() and 1 <= int(answer) <= len(names):
                answer = names[int(answer) - 1]
            # Recorded because a labeller who reads the guest's name off an
            # episode title and types it produces ground truth that scores
            # the matcher CORRECT for the wrong reason.
            basis = input("  heard it (v) or inferred from context (c)? ").strip()
            labels[clip["clip_id"]] = {
                "person": answer,
                "basis": "voice" if basis != "c" else "context",
            }
        history.append(clip["clip_id"])
        _write_json(LABELS_PATH, labels)
        print()

    print(f"{len(labels)} labels in {LABELS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="choose episodes and clip windows")
    plan_parser.add_argument("--dir", default=Path("downloaded"), type=Path)
    plan_parser.add_argument("--seed", default="tier1-precision-2026-09-02")
    plan_parser.set_defaults(func=cmd_plan)

    label_parser = sub.add_parser("label", help="label the cut clips")
    label_parser.add_argument("--plan", default=str(PLAN_PATH))
    label_parser.add_argument("--clips", default=str(CLIPS_DIR))
    label_parser.set_defaults(func=cmd_label)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
