"""Episodes deliberately kept out of the corpus.

Four episodes are Geopolitical Cousins content republished on the Jacob
Shapiro feed. The transcripts are byte-identical apart from the header show
name, and each was downloaded and transcribed twice. They are embedded under
Geopolitical Cousins only.

This is a HUMAN-MAINTAINED LIST, deliberately not a heuristic. The enclosure
URLs differ across feeds (zero shared -- Captivate re-hosts), so the only
available signal is a fuzzy title-and-date match, and code that silently
discards an episode because a regex thought two titles matched is a failure
mode with no alarm on it. Reconciliation REPORTS suspected cross-posts;
exclusion is a decision recorded here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExcludedEpisode:
    show: str
    episode_number: str
    date: str
    guid: str | None
    reason: str

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.show, self.episode_number, self.date)


EXCLUDED: frozenset[ExcludedEpisode] = frozenset(
    {
        ExcludedEpisode(
            "The Jacob Shapiro Podcast",
            "Unknown",
            "2026-07-29",
            "1c45dbd9-0dc3-4d07-b2d1-758fe78405fe",
            "cross-post of Geopolitical Cousins 73, 'This Is The Way The World Ends'",
        ),
        ExcludedEpisode(
            "The Jacob Shapiro Podcast",
            "Unknown",
            "2026-07-31",
            "d738c6b4-cb9e-497e-995f-c106c42d9b1d",
            "cross-post of Geopolitical Cousins 74, 'Lessons Learned'",
        ),
        # The 2025 pair. Captivate publishes an itunes_episode for these two,
        # which is why they carry real numbers where the 2026 pair fall back to
        # the literal "Unknown". Their Chroma records were deleted by hand on
        # 2026-09-01 (175 and 50 records); these entries are what stops the
        # cron restoring them the next morning.
        ExcludedEpisode(
            "The Jacob Shapiro Podcast",
            "271",
            "2025-04-04",
            "c4af95bf-cfbc-4c0a-b4d7-2c2df77d1fe6",
            "cross-post of Geopolitical Cousins, 'Riding on the Hog of a Fiscal Orgy'",
        ),
        ExcludedEpisode(
            "The Jacob Shapiro Podcast",
            "273",
            "2025-04-08",
            "3a3c0a69-ea66-46b1-a54e-7ef1ea657505",
            "cross-post of Geopolitical Cousins, 'Let Them Drink Bleach'",
        ),
    }
)

# Derived, never hand-maintained. Two parallel hand-written lists would drift
# ASYMMETRICALLY: forgetting a guid still excludes correctly today and fails
# only in the future scenario the guid was added for, so the violation is
# invisible until exactly the moment it matters.
EXCLUDED_EPISODES: frozenset[tuple[str, str, str]] = frozenset(
    e.triple for e in EXCLUDED
)
EXCLUDED_GUIDS: frozenset[str] = frozenset(
    e.guid for e in EXCLUDED if e.guid is not None
)


def is_excluded(
    show: str,
    episode_number: str,
    date_str: str,
    episode_guid: str | None = None,
) -> bool:
    """Whether this episode is deliberately kept out of the corpus.

    Either arm matching is enough. The triple is NOT durable: both excluded
    episodes fall back to the literal "Unknown" precisely because Captivate
    publishes no itunes_episode for them, and the moment it backfills one --
    the spec's six-month threat model, and the reason episode_guid exists --
    the triple changes and the triple arm silently stops matching. The guid
    does not move with a metadata backfill.

    The triple arm still earns its place: 6 episodes have aged off the front
    of their feed and can never be assigned a guid at all.

    CALLERS MUST PASS episode_guid WHERE THEY HAVE ONE. An arm that no live
    path reaches is not protection, it is decoration.

    THE GUID ARM IS SCOPED BY SHOW, and that is not defensive tidiness -- an
    unscoped guid arm excluded the ORIGINALS. Every excluded episode here is a
    CROSS-POST, and a cross-post syndicated through Captivate MAY carry the
    same guid as the episode it copies. The 2026 pair do: `guid in
    EXCLUDED_GUIDS` alone matched Geopolitical Cousins 73 and 74 as well as
    the Jacob Shapiro copies of them, and the repair run planned EXCLUDE for
    the two episodes this entire project exists to fix. Caught only by reading
    a real run's plan output.

    The 2025 pair do NOT -- measured against the live feed on 2026-09-01,
    their guids differ from their Geopolitical Cousins originals. So guid
    sharing is a property of the individual cross-post, not of Captivate, and
    there is no way to tell which kind you have without checking the feed.
    That is precisely why the arm is scoped unconditionally rather than only
    where sharing is known: the safe version does not depend on knowing.

    Scoping by show keeps what the arm is for -- surviving an episode_number
    backfill WITHIN a show, where the triple moves and the guid does not --
    while making it structurally impossible to reach the other feed. This is
    the same correction Task 8 applied to the prune's guid arm, which was
    likewise unscoped and likewise crossed a show boundary.
    """
    for excluded in EXCLUDED:
        if (
            episode_guid is not None
            and excluded.guid == episode_guid
            and excluded.show == show
        ):
            return True
        if excluded.triple == (show, episode_number, date_str):
            return True
    return False
