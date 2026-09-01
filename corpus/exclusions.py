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
    CROSS-POST, and a cross-post syndicated through Captivate carries THE SAME
    GUID as the episode it copies. So `guid in EXCLUDED_GUIDS` alone matched
    Geopolitical Cousins 73 and 74 as well as the Jacob Shapiro copies of
    them, and the repair run planned EXCLUDE for the two episodes this entire
    project exists to fix. Caught only by reading a real run's plan output.

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
