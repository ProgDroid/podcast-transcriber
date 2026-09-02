"""Recompute the corpus figures the speaker-identification spec depends on.

The spec's Part 2 quotes cluster counts, a segment/turn comparison and
speech-concentration percentages. Nothing in this repository produced them:
they were computed once, by hand, and only the results survived. A number
that cannot be recomputed is not a measurement, so this script is the method,
kept in the repo and driven by `corpus.speakers`.

Every run states its own inputs -- the directory, the file count and the gap
cap -- because the figures are only comparable between runs that agree on
all three.

    uv run python speaker_stats.py --dir downloaded --cap 30
    uv run python speaker_stats.py --dir some/snapshot --cap inf

`--cap inf` reproduces the naive "next start minus this start" derivation and
exists as a positive control: run it against the older snapshot and a figure
that still disagrees with the published one is a difference of METHOD, not of
population.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter
from pathlib import Path

from corpus.chunking import parse_transcript_segments
from corpus.identity import parse_transcript_filename
from corpus.speakers import (
    GAP_CAP_S,
    MIN_TURN_S,
    count_non_monotonic,
    merge_turns,
    speech_shares,
)

CONCENTRATION_FLOOR = 0.70


def show_of(name: str) -> str:
    return name.split(" - Episode ")[0]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[idx]


def raw_gaps(segments: list[dict]) -> list[float]:
    """Uncapped inter-segment gaps, for choosing a cap."""
    return [
        segments[i + 1]["start"] - segments[i]["start"]
        for i in range(len(segments) - 1)
        if segments[i + 1]["start"] >= segments[i]["start"]
    ]


def load(directory: Path, cap: float) -> list[dict]:
    episodes = []
    for path in sorted(directory.glob("*.txt")):
        segments = parse_transcript_segments(path.read_text(encoding="utf-8"))
        if not segments:
            continue
        turns = merge_turns(segments, gap_cap_s=cap)
        shares = speech_shares(turns, min_turn_s=MIN_TURN_S)
        # Reuse the tested parser rather than slicing the filename here: a
        # second date derivation is a second thing that can disagree.
        parsed = parse_transcript_filename(path.name)
        episodes.append(
            {
                "show": show_of(path.name),
                "name": path.name,
                "year": parsed[2][:4] if parsed else "unparsed",
                "segments": segments,
                "n_segments": len(segments),
                "turns": turns,
                "gaps": raw_gaps(segments),
                "shares": shares,
                "non_monotonic": count_non_monotonic(segments),
            }
        )
    return episodes


def report_gaps(episodes: list[dict]) -> None:
    gaps = [g for e in episodes for g in e["gaps"]]
    total = sum(gaps)
    print("\n## Inter-segment gap distribution (uncapped)")
    print(f"gaps: {len(gaps):,}   total: {total / 3600:,.1f} h")
    for q in (0.50, 0.90, 0.99, 0.999):
        print(f"  p{q * 100:<5g} {percentile(gaps, q):8.1f}s")
    print(f"  max    {max(gaps):8.1f}s" if gaps else "  max      n/a")
    print("\n  seconds removed by a candidate cap:")
    for cap in (10, 15, 20, 30, 60, 120):
        trimmed = sum(g - cap for g in gaps if g > cap)
        over = sum(1 for g in gaps if g > cap)
        print(
            f"    cap {cap:>4}s  {trimmed / 3600:7.1f} h "
            f"({trimmed / total:5.2%} of derived time, {over:,} gaps)"
        )


def report_clusters(episodes: list[dict]) -> None:
    counts = [len({t["speaker"] for t in e["turns"]}) for e in episodes]
    print("\n## Clusters per episode")
    print(
        f"median {statistics.median(counts):g}   "
        f"mean {statistics.mean(counts):.2f}   "
        f"range {min(counts)}-{max(counts)}"
    )
    histogram = Counter(counts)
    print(
        "  distribution: " + ", ".join(f"{k}x{histogram[k]}" for k in sorted(histogram))
    )
    for show in sorted({e["show"] for e in episodes}):
        per_show = [
            len({t["speaker"] for t in e["turns"]})
            for e in episodes
            if e["show"] == show
        ]
        print(
            f"  {show:<30} median {statistics.median(per_show):g} (n={len(per_show)})"
        )


def report_segments_vs_turns(episodes: list[dict], cap: float) -> None:
    seg_durations = [
        min(max(e["segments"][i + 1]["start"] - e["segments"][i]["start"], 0.0), cap)
        for e in episodes
        for i in range(len(e["segments"]) - 1)
    ]
    turn_durations = [t["duration_s"] for e in episodes for t in e["turns"]]
    n_segments = sum(e["n_segments"] for e in episodes)
    n_turns = sum(len(e["turns"]) for e in episodes)
    print("\n## Segments versus turns")
    print(f"{'':<18}{'count':>12}{'median duration':>18}")
    print(
        f"{'Whisper segments':<18}{n_segments:>12,}"
        f"{statistics.median(seg_durations):>17.1f}s"
    )
    print(
        f"{'Merged turns':<18}{n_turns:>12,}{statistics.median(turn_durations):>17.1f}s"
    )
    print(f"  ratio: {n_segments / n_turns:.1f}x fewer embeddings")


def report_concentration(episodes: list[dict]) -> None:
    print(f"\n## Speech concentration (turns under {MIN_TURN_S}s dropped)")
    header = (
        f"{'':<30}{'pooled top-1':>14}{'pooled top-2':>14}"
        f"{'median top-1':>14}{'top-1 < 70%':>14}"
    )
    print(header)
    for show in sorted({e["show"] for e in episodes}) + ["ALL"]:
        rows = [e for e in episodes if show == "ALL" or e["show"] == show]
        rows = [e for e in rows if e["shares"]["total_s"] > 0]
        if not rows:
            continue
        total = sum(e["shares"]["total_s"] for e in rows)
        top1 = sum(e["shares"]["top1_s_share"] * e["shares"]["total_s"] for e in rows)
        top2 = sum(e["shares"]["top2_s_share"] * e["shares"]["total_s"] for e in rows)
        medians = [e["shares"]["top1_s_share"] for e in rows]
        below = sum(1 for m in medians if m < CONCENTRATION_FLOOR)
        print(
            f"{show:<30}{top1 / total:>13.1%}{top2 / total:>14.1%}"
            f"{statistics.median(medians):>14.1%}"
            f"{f'{below} / {len(rows)}':>14}"
        )
    # The Tier 1 rule needs a dominant cluster to exist at all before it can
    # ask whether that cluster is the host.
    rows = [e for e in episodes if e["shares"]["total_s"] > 0]
    majority = sum(1 for e in rows if e["shares"]["top1_s_share"] > 0.50)
    medians = [e["shares"]["top1_s_share"] for e in rows]
    print(
        f"\n  {majority} of {len(rows)} episodes have a top-1 cluster over 50% "
        f"of speech, median share {statistics.median(medians):.1%}"
    )


def report_words_cross_check(episodes: list[dict]) -> None:
    """Seconds and words are computed independently; disagreement is signal."""
    print("\n## Seconds versus words (cross-check)")
    print(f"{'':<30}{'pooled top-1 s':>16}{'pooled top-1 w':>16}{'delta':>10}")
    for show in sorted({e["show"] for e in episodes}) + ["ALL"]:
        rows = [e for e in episodes if show == "ALL" or e["show"] == show]
        rows = [e for e in rows if e["shares"]["total_s"] > 0]
        if not rows:
            continue
        total_s = sum(e["shares"]["total_s"] for e in rows)
        total_w = sum(e["shares"]["total_words"] for e in rows)
        top1_s = sum(e["shares"]["top1_s_share"] * e["shares"]["total_s"] for e in rows)
        top1_w = sum(
            e["shares"]["top1_words_share"] * e["shares"]["total_words"] for e in rows
        )
        a, b = top1_s / total_s, top1_w / total_w
        print(f"{show:<30}{a:>15.1%}{b:>16.1%}{b - a:>+10.1%}")


def report_by_year(episodes: list[dict]) -> None:
    """Is a recent shift a trend or a small-sample blip?

    A shift measured only against "everything before it" cannot tell those
    apart: the same 15 episodes are both the signal and the whole of the
    recent bucket. Splitting the full history by year gives each bucket its
    own n and shows whether a value moved progressively or jumped once.
    """
    print("\n## Drift by year")
    print(
        f"{'show':<28}{'year':>6}{'n':>5}{'clusters':>10}"
        f"{'top-1':>8}{'top-2':>8}{'med turn':>10}"
    )
    for show in sorted({e["show"] for e in episodes}):
        in_show = [e for e in episodes if e["show"] == show]
        for year in sorted({e["year"] for e in in_show}):
            rows = [
                e for e in in_show if e["year"] == year and e["shares"]["total_s"] > 0
            ]
            if not rows:
                continue
            clusters = [len({t["speaker"] for t in e["turns"]}) for e in rows]
            total = sum(e["shares"]["total_s"] for e in rows)
            top1 = sum(
                e["shares"]["top1_s_share"] * e["shares"]["total_s"] for e in rows
            )
            top2 = sum(
                e["shares"]["top2_s_share"] * e["shares"]["total_s"] for e in rows
            )
            durations = [t["duration_s"] for e in rows for t in e["turns"]]
            print(
                f"{show:<28}{year:>6}{len(rows):>5}"
                f"{statistics.median(clusters):>10g}"
                f"{top1 / total:>8.1%}{top2 / total:>8.1%}"
                f"{statistics.median(durations):>9.1f}s"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="downloaded", type=Path)
    parser.add_argument("--cap", default=GAP_CAP_S, type=float)
    parser.add_argument("--by-year", action="store_true")
    args = parser.parse_args()

    episodes = load(args.dir, args.cap)
    if not episodes:
        raise SystemExit(f"No parseable transcripts in {args.dir}")

    cap_label = "inf (naive)" if math.isinf(args.cap) else f"{args.cap:g}s"
    print(f"# dir={args.dir}  files={len(episodes)}  gap_cap={cap_label}")
    faults = sum(e["non_monotonic"] for e in episodes)
    print(f"# non-monotonic segment starts: {faults}")

    report_gaps(episodes)
    report_clusters(episodes)
    report_segments_vs_turns(episodes, args.cap)
    report_concentration(episodes)
    report_words_cross_check(episodes)
    if args.by_year:
        report_by_year(episodes)


if __name__ == "__main__":
    main()
