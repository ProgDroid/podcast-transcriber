"""Bring `speakers/assignments.json` onto the JSP feed's new episode numbers.

The 2026-11-15 renumber migration (`migration/renumber.py`) re-keyed 299 JSP
episodes in Chroma and on the volume, but `assignments.json` still named them
by their old numbers, so any (show, episode, date) join to the store missed
338 of its 350 JSP episodes.

An archived old file is remapped only if its body (all but the header line)
equals exactly one new file of the same date: same transcript, same
SPEAKER_XX labels, so the cluster names still point at the right voices. A
body with no identical new file would be a fresh diarisation whose labels are
different clusters, and its entries are dropped instead. The match is by
TEXT, never by the +72 shift.

MEASURED 2026-11-15: all 345 archived files matched, including the 46
episodes re-transcribed on 2026-11-14/15 before the guard landed -- each
re-transcription reproduced the old transcript byte for byte (same image, T4,
audio; pyannote disables TF32 for reproducibility). So 334 episodes were
remapped and none dropped. Do not generalise: a different image or GPU type
may not reproduce, which is what the text check is for.

Usage, with the volume downloaded (`modal volume get podcast-transcripts / D`):
  py migration/remap_assignments.py D            # dry run
  py migration/remap_assignments.py D --apply --expect-renamed 299 --expect-dropped 35
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus.identity import parse_transcript_filename, transcript_filename  # noqa: E402

SHOW = "The Jacob Shapiro Podcast"
ASSIGNMENTS = Path(__file__).resolve().parent.parent / "speakers" / "assignments.json"


def _body(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    return text.partition("\n")[2]


def classify(volume_dir: Path) -> tuple[dict[str, str], set[str], list[str]]:
    """old filename -> new filename, stale filenames, and problems."""
    archive = volume_dir / "renumbered"
    by_date: dict[str, list[Path]] = {}
    for path in volume_dir.iterdir():
        parsed = parse_transcript_filename(path.name)
        if parsed and parsed[0] == SHOW:
            by_date.setdefault(parsed[2], []).append(path)

    renamed: dict[str, str] = {}
    stale: set[str] = set()
    problems: list[str] = []
    for old in sorted(archive.iterdir()):
        parsed = parse_transcript_filename(old.name)
        if not parsed or parsed[0] != SHOW:
            continue
        body = _body(old)
        matches = [p for p in by_date.get(parsed[2], []) if _body(p) == body]
        if len(matches) == 1:
            renamed[old.name] = matches[0].name
        elif not matches:
            stale.add(old.name)
        else:
            problems.append(f"{old.name}: {len(matches)} identical new files")
    return renamed, stale, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("volume_dir", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--expect-renamed", type=int, default=-1)
    ap.add_argument("--expect-dropped", type=int, default=-1)
    args = ap.parse_args()

    renamed, stale, problems = classify(args.volume_dir)
    data = json.loads(ASSIGNMENTS.read_text(encoding="utf-8"))
    used = {a["episode"] for a in data["assignments"]}
    renamed_used = sorted(set(renamed) & used)
    stale_used = sorted(stale & used)
    print(
        f"archive: {len(renamed)} renamed, {len(stale)} re-transcribed, "
        f"{len(problems)} problems"
    )
    for line in problems:
        print(f"   {line}")
    print(
        f"assignments.json: {len(renamed_used)} episodes to remap, "
        f"{len(stale_used)} to drop"
    )

    assignments, dropped_records = [], 0
    for record in data["assignments"]:
        if record["episode"] in stale:
            dropped_records += 1
            continue
        if record["episode"] in renamed:
            _, number, date = parse_transcript_filename(renamed[record["episode"]])
            record = {
                **record,
                "episode": transcript_filename(SHOW, number, date),
                "episode_number": number,
            }
        assignments.append(record)
    conflicts = []
    for conflict in data["conflicts"]:
        if conflict["episode"] in stale:
            continue
        if conflict["episode"] in renamed:
            conflict = {**conflict, "episode": renamed[conflict["episode"]]}
        conflicts.append(conflict)
    print(
        f"records: {len(data['assignments'])} -> {len(assignments)} "
        f"({dropped_records} dropped); conflicts: {len(data['conflicts'])} -> "
        f"{len(conflicts)}"
    )
    print("REMAP PLAN COMPLETE")

    if not args.apply:
        print(
            f"DRY RUN -- nothing written. To apply: --apply --expect-renamed "
            f"{len(renamed_used)} --expect-dropped {len(stale_used)}"
        )
        return 0
    if problems or (len(renamed_used), len(stale_used)) != (
        args.expect_renamed,
        args.expect_dropped,
    ):
        print("REFUSED: counts differ from the expected ones, or problems exist.")
        return 1
    data["assignments"], data["conflicts"] = assignments, conflicts
    tmp = ASSIGNMENTS.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, ASSIGNMENTS)
    print("REMAP COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
