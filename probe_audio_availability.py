"""THROWAWAY PROBE — how much of the archive can Tier 2 actually re-embed?

Tier 2 needs AUDIO, and audio is not retained: every episode must be re-fetched
from its feed. Part 7 §7 bounded that for one show (the Jacob Shapiro feed
carries 350 of 356) and left Geopolitical Cousins and Observing Japan
unmeasured. This settles the denominator for all three before the scope of
Tier 2 is argued rather than after.

Two distinct failure modes, kept separate because they mean different things:

  ABSENT   the (episode, date) key is not in the feed at all -- aged off.
  DEAD     the feed offers a URL and the URL does not serve bytes -- rot.

`speakers/unavailable.json` currently holds 6 ABSENT and 0 DEAD, so the second
mode is unobserved rather than known-impossible. This probe tests for it.

No downloads: HEAD, falling back to a one-byte ranged GET where a CDN rejects
HEAD. A 405 read as rot would be a false negative, which is the expensive
direction here -- it would shrink Tier 2's scope for no reason.

Run:  py probe_audio_availability.py
"""

import concurrent.futures
import json
import sys
from pathlib import Path

import requests

from corpus.identity import parse_transcript_filename
from speaker_tool import FEEDS, resolve_audio_urls

DIRECTORY = Path("downloaded")

# Someone else's server. The repo caps clip downloads at 8 containers for this
# reason; a HEAD is far cheaper than a 40MB GET but the courtesy is the same.
WORKERS = 8
TIMEOUT = 30


def check(url: str) -> tuple[str, str]:
    """(verdict, detail) for one URL. HEAD, then a ranged GET if HEAD is refused."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
        if r.status_code == 200:
            return "OK", f"HEAD 200 {r.headers.get('content-length', '?')}B"
        # Plenty of CDNs simply do not implement HEAD. Calling that rot would
        # under-count the archive, so ask for one byte instead before judging.
        if r.status_code in (403, 405, 501):
            g = requests.get(
                url, headers={"Range": "bytes=0-0"}, timeout=TIMEOUT, stream=True
            )
            g.close()
            if g.status_code in (200, 206):
                return "OK", f"HEAD {r.status_code}, ranged GET {g.status_code}"
            return "DEAD", f"HEAD {r.status_code}, ranged GET {g.status_code}"
        return "DEAD", f"HEAD {r.status_code}"
    except Exception as e:  # noqa: BLE001 - a network error is a verdict here
        return "UNKNOWN", f"{type(e).__name__}: {e}"[:160]


def main() -> int:
    files = sorted(DIRECTORY.glob("*.txt"))
    # State the inputs. A figure from this cache is only as current as the
    # cache, and nothing else records when it was taken.
    print(f"dir={DIRECTORY} files={len(files)}")

    by_show: dict[str, list[tuple[str, str, str]]] = {}
    unparsed = 0
    for path in files:
        parsed = parse_transcript_filename(path.name)
        if not parsed:
            unparsed += 1
            continue
        show, ep, date = parsed
        by_show.setdefault(show, []).append((path.name, ep, date))
    print(f"unparsed={unparsed}")
    for show, items in sorted(by_show.items()):
        print(f"  {show}: {len(items)}")
    print()

    rows = []
    for show, items in sorted(by_show.items()):
        if show not in FEEDS:
            print(f"  !! {show} has no feed configured; skipping")
            continue
        urls = resolve_audio_urls.local(FEEDS[show], show)

        # Same guard as cut_clips: a half-failed feed fetch is real and has
        # happened. Refuse rather than record a transient shortfall as rot.
        if len(urls) < len(items) * 0.9:
            print(
                f"REFUSING: {show} resolved {len(urls)} entries for "
                f"{len(items)} transcripts. That is a failed fetch, not "
                f"aged-off audio. Re-run before believing anything here."
            )
            return 2

        resolved, absent = [], []
        for name, ep, date in items:
            url = urls.get(f"{ep}|{date}")
            (resolved if url else absent).append((name, url))

        results: dict[str, tuple[str, str]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(check, u): n for n, u in resolved}
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()

        ok = [n for n, (v, _) in results.items() if v == "OK"]
        dead = [(n, d) for n, (v, d) in results.items() if v == "DEAD"]
        unk = [(n, d) for n, (v, d) in results.items() if v == "UNKNOWN"]
        rows.append((show, len(items), len(ok), len(absent), len(dead), len(unk)))

        print(f"\n{show}: {len(ok)} fetchable of {len(items)}")
        if absent:
            print(f"  ABSENT from feed ({len(absent)}):")
            for name, _ in sorted(absent):
                print(f"    {name}")
        for label, group in (("DEAD", dead), ("UNKNOWN", unk)):
            if group:
                print(f"  {label} ({len(group)}):")
                for name, detail in sorted(group):
                    print(f"    {name}  {detail}")

    print("\n" + "=" * 70)
    print(f"{'show':<32}{'total':>7}{'ok':>7}{'absent':>8}{'dead':>7}{'unk':>6}")
    for show, total, ok, absent, dead, unk in rows:
        print(f"{show:<32}{total:>7}{ok:>7}{absent:>8}{dead:>7}{unk:>6}")
    t = [sum(c) for c in zip(*[r[1:] for r in rows], strict=True)]
    print(f"{'TOTAL':<32}{t[0]:>7}{t[1]:>7}{t[2]:>8}{t[3]:>7}{t[4]:>6}")
    print(f"\nTier 2 denominator: {t[1]} of {t[0]} ({100 * t[1] / t[0]:.1f}%)")

    Path("speakers/audio_availability.json").write_text(
        json.dumps(
            {
                "dir": str(DIRECTORY),
                "files": len(files),
                "shows": [
                    {
                        "show": s,
                        "total": tot,
                        "ok": ok,
                        "absent": ab,
                        "dead": d,
                        "unknown": u,
                    }
                    for s, tot, ok, ab, d, u in rows
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("PROBE COMPLETE")  # terminal marker; absence means it did not finish
    return 0


if __name__ == "__main__":
    sys.exit(main())
