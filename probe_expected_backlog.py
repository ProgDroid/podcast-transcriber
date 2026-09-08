"""Pre-register the expected IDENTIFY_ONLY count, BEFORE a nightly run.

NOT throwaway. `docs/operations.md` § "Identity repair, and the phase gate"
requires the expected count to be "enumerated from the feeds BEFORE the run"
at two separate steps, so this is run again at each of them.

WHY IT IS NOT ARITHMETIC ON 433. Two populations that look interchangeable
are not:

  * The STAMP SET is selected by triple against the STORE. The feed cannot
    affect it. It is 433 and `--expect 433` stays right.
  * IDENTIFY_ONLY can only ever count episodes `parse_all_episodes` EMITS.
    The feed keeps a fixed window, so archive episodes fall out of reach over
    time and stop being countable while remaining stamped.

The gap between them grows on its own, silently, and in the wrong direction:
it makes the real count come in BELOW the predicted one, which reads exactly
like the runbook's "a small number means the planner is not reading the key".
Distinguishing a rolled-off episode from a broken planner is this probe's
whole job.

THE CONTROL IS THE POINT. Comparing totals proves nothing -- one episode
leaving the feed and one being published cancel exactly, which is precisely
what happened on 2026-09-07. So it asserts, per show, that the feed episodes
AT OR BEFORE the archive boundary equal the archive count, and on a mismatch
prints the multiset difference by episode number. That decomposition is what
turns "the numbers disagree" into "episode 7 rolled off, episode 354 is new".

MEASURED 2026-09-07, before the first classification night:

    JSP  archive=350  feed<=boundary=349  MISMATCH -> ep 7 rolled off
    GC   archive=76   feed<=boundary=76   MATCH
    OJ   archive=7    feed<=boundary=7    MATCH
    expected first night = 432  (349 / 76 / 7), NOT 433

Both post-boundary episodes published 12:30 UTC that day, AFTER the v7
deploy, so they are TRANSCRIBE and the ingest path stamps them -- they add 0.

RUN IT WITH SYSTEM PYTHON. The `uv` environment has no `feedparser`; the
system interpreter does. From the repo root, so `sys.path[0]` is the repo:

    py probe_expected_backlog.py
"""

from __future__ import annotations

import collections
import datetime
import email.utils
import json

from corpus.exclusions import is_excluded
from stamp_identity_version import episode_triples
from transcribe import parse_all_episodes

# The shows, copied from `scheduled_job`, whose list is inline in the function
# body and so cannot be imported.
SHOWS = [
    ("https://feeds.captivate.fm/geopolitical-cousins/", "Geopolitical Cousins"),
    ("https://feeds.captivate.fm/jacob-shapiro/", "The Jacob Shapiro Podcast"),
    (
        "https://api.substack.com/feed/podcast/868206/s/386602.rss",
        "The Observing Japan Podcast",
    ),
]

# When phase 1 went live. An episode INGESTED after this gets
# `identity_version` written by the ingest path itself, so it is never queued
# and contributes 0. One ingested before it lacks the key and contributes 1.
# Update this if the relevant deploy ever moves.
IDENTITY_DEPLOY_UTC = datetime.datetime(
    2026, 9, 7, 18, 21, tzinfo=datetime.timezone.utc
)

# `scheduled_job` runs on `modal.Cron("0 9 * * *")`.
CRON_UTC_HOUR = 9

ASSIGNMENTS = "speakers/assignments.json"


def published_utc(entry) -> datetime.datetime | None:
    """The entry's publication instant in UTC, or None if it has none."""
    raw = entry.get("published")
    if not raw:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def first_ingest_after(published: datetime.datetime) -> datetime.datetime:
    """The nightly cron instant that first SEES this episode.

    Publication time is the wrong thing to compare against a deploy. An
    episode published at 12:30 UTC is not ingested at 12:30 -- it waits for
    the next 09:00 UTC run, which may fall on the far side of a deploy that
    changes what ingest writes. Comparing publication directly counted two
    episodes as pre-deploy that are in fact ingested after it.
    """
    candidate = published.replace(hour=CRON_UTC_HOUR, minute=0, second=0, microsecond=0)
    if candidate <= published:
        candidate += datetime.timedelta(days=1)
    return candidate


def main() -> None:
    with open(ASSIGNMENTS) as handle:
        assignments = json.load(handle)["assignments"]

    archive = collections.defaultdict(list)
    for show, number, date in episode_triples(assignments):
        archive[show].append((number, date))

    total_expected = 0
    total_archive = 0
    control_ok = True

    for feed_url, show in SHOWS:
        entries = archive[show]
        boundary = max(date for _, date in entries)
        archive_n = len(entries)
        total_archive += archive_n

        episodes = parse_all_episodes(feed_url)
        at_or_before = [e for e in episodes if e["date"] <= boundary]
        after = [e for e in episodes if e["date"] > boundary]

        ok = len(at_or_before) == archive_n
        control_ok = control_ok and ok

        print(f"\n{show}")
        print(
            f"  archive={archive_n}  boundary={boundary}  feed_emitted={len(episodes)}"
        )
        print(
            f"  CONTROL feed<=boundary={len(at_or_before)} vs archive="
            f"{archive_n} -> {'MATCH' if ok else 'MISMATCH'}"
        )

        if not ok:
            # Decompose rather than report a bare disagreement. Totals can
            # cancel, so compare as MULTISETS by episode number.
            arch_c = collections.Counter(number for number, _ in entries)
            feed_c = collections.Counter(e["episode_number"] for e in episodes)
            for number in sorted(set(arch_c) | set(feed_c)):
                a, f = arch_c.get(number, 0), feed_c.get(number, 0)
                if a != f:
                    where = (
                        "archive only (rolled off feed)" if a > f else "feed only (new)"
                    )
                    print(f"    ep {number}: archive={a} feed={f}  <- {where}")

        # Archive episodes the feed still emits. These are the certain term:
        # stamped-but-unreachable episodes can never be classified.
        #
        # MINUS the deliberate exclusions. This is the term the probe missed
        # on 2026-09-07: it predicted 432 where the planner reported 428, and
        # the four-episode gap was `corpus/exclusions.py` in full. An excluded
        # episode is emitted BY THE FEED and present IN `assignments.json`, so
        # it survives the reachability filter and the archive count both --
        # and `decide_action` checks EXCLUDE first, so it can never reach
        # IDENTIFY_ONLY. The feed roll-off was subtracted and this was not.
        excluded_before = [e for e in at_or_before if _excluded(show, e)]
        reachable = len(at_or_before) - len(excluded_before)
        if excluded_before:
            print(
                f"  EXCLUDE (deliberate cross-posts, never IDENTIFY_ONLY) = "
                f"{len(excluded_before)}"
            )
            for episode in excluded_before:
                print(f"    ep {episode['episode_number']} ({episode['date']})")

        counted_post = []
        for episode in after:
            if _excluded(show, episode):
                print(
                    f"    post-boundary: {episode['date']} ep "
                    f"{episode['episode_number']} -> EXCLUDED, adds 0"
                )
                continue
            when = published_utc(
                next(
                    (
                        entry
                        for entry in _entries(feed_url)
                        if str(entry.get("title", "")) == episode["title"]
                    ),
                    {},
                )
            )
            # UNKNOWN publication is treated as counted: an episode that adds
            # 1 when it should add 0 makes the prediction too HIGH, and a real
            # count coming in under a too-high prediction is investigated. The
            # opposite error hides a genuine backlog entry.
            ingest = first_ingest_after(when) if when else None
            stamped_at_ingest = ingest is not None and ingest > IDENTITY_DEPLOY_UTC
            marker = "stamped at ingest, adds 0" if stamped_at_ingest else "ADDS 1"
            fmt = "%Y-%m-%d %H:%M"
            print(
                f"    post-boundary: {episode['date']} ep "
                f"{episode['episode_number']} published "
                f"{when.strftime(fmt) if when else '?'} UTC, first ingest "
                f"{ingest.strftime(fmt) if ingest else '?'} UTC -> {marker}"
            )
            if not stamped_at_ingest:
                counted_post.append(episode)

        expected = reachable + len(counted_post)
        total_expected += expected
        print(f"  EXPECTED IDENTIFY_ONLY = {expected}")

    print("\n" + "=" * 62)
    print(
        f"CONTROL across all shows: {'PASS' if control_ok else 'SEE DECOMPOSITION ABOVE'}"
    )
    print(f"stamp set (store-based, unaffected by the feed) = {total_archive}")
    print(f"EXPECTED IDENTIFY_ONLY total                    = {total_expected}")
    print(
        "\nThese two are DIFFERENT NUMBERS and neither is a typo for the other.\n"
        "The stamp set is every episode `assignments.json` records a run over,\n"
        "selected by triple against the STORE. The backlog is what the planner\n"
        "can reach FROM THE FEED and has not excluded. They differ by the\n"
        "rolled-off episodes plus the deliberate exclusions -- 433 vs 428 on\n"
        "2026-09-08. Carrying one into a check expecting the other is the\n"
        "mistake this block exists to prevent."
    )
    print(
        "\nA count BELOW this is not automatically a broken planner -- check the\n"
        "decomposition above for a newly rolled-off episode first. A count near\n"
        "zero, with the control passing, is the planner failing to read the key.\n"
        "\nTHIS PREDICTION ASSUMES NOTHING CARRIES identity_version YET. It models\n"
        "an UNSTAMPED store and has no way to read one, so it keeps printing the\n"
        "pre-stamp number forever. The stamp pass ran 2026-09-08; from that night\n"
        "on the planner should report 0 and a near-zero count is the SUCCESS\n"
        "case, not the key-not-read case above. Use this probe before a stamp,\n"
        "or after a RULES_VERSION bump re-opens the backlog -- not to grade a\n"
        "night that follows one."
    )


_FEED_CACHE: dict = {}


def _excluded(show: str, episode: dict) -> bool:
    """`decide_action`'s EXCLUDE arm, mirrored rather than re-derived.

    The guid is passed because `parse_all_episodes` already carries one and
    `is_excluded` documents that a caller holding one must supply it -- the
    triple is not durable across an `episode_number` backfill, which is the
    exact scenario the guid arm exists for.
    """
    return is_excluded(
        show, episode["episode_number"], episode["date"], episode.get("guid")
    )


def _entries(feed_url: str):
    """Raw feedparser entries, cached so each feed is fetched once."""
    if feed_url not in _FEED_CACHE:
        import feedparser

        _FEED_CACHE[feed_url] = feedparser.parse(feed_url).entries
    return _FEED_CACHE[feed_url]


if __name__ == "__main__":
    main()
