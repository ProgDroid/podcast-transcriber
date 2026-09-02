"""Feed-entry facts that both the writer and the migration must agree on."""

from __future__ import annotations

import re


def entry_guid(entry: dict) -> str | None:
    """The entry's stable guid, or None when it does not have one.

    RSS `<guid isPermaLink>` DEFAULTS TO TRUE, so an absent `guidislink` key
    means permalink, not "plain guid". A link-derived id is a URL and this
    publisher rewrites URLs, so it is not an identity and must be refused.
    Returning None is correct and expected: 6 episodes have aged off the front
    of their feed and can never be assigned a guid at all.
    """
    gid = entry.get("id")
    if not gid or entry.get("guidislink", True):
        return None
    return str(gid)


def episode_number_of(entry: dict) -> str:
    """The entry's episode number as the pipeline files it, or "Unknown".

    Prefers the machine-readable `<itunes:episode>` and falls back to the
    title. The fallback matches "Ep 5", "Ep. 5" and "Ep5" but NOT the word
    "Episode 5" -- after `Ep` the pattern allows only an optional dot and
    whitespace before the digits. Three Observing Japan episodes carry their
    number plainly in the title and are filed "Unknown" because of it (5 of
    439 transcripts overall, verified 2026-09-02).

    DO NOT "fix" it. That publisher's two numbering signals disagree by one
    wherever both exist -- `<itunes:episode>7</itunes:episode>` on an item
    titled "Episode 6", and 4 on one titled "Episode 3", because they count
    the trailer as episode 1. Widening the regex would number the show
    7, 5, 4, 4, 3, 2, Unknown: two "Episode 4"s from two different sources.
    `corpus/identity.py` keys on the (show, episode, date) triple precisely
    because episode_number is a display attribute, so neither form is a
    correctness bug -- but "Unknown" is a visible gap where a duplicated
    number is an invisible collision.

    This lives here rather than in `transcribe.py` because the clip tool has
    to resolve a transcript filename back to its feed entry, and a second
    copy of this derivation is a second thing that can disagree with the
    filenames already on the volume.
    """
    number = entry.get("itunes_episode", None)
    if number is None:
        match = re.search(r"\b[Ee]p\.?\s*(\d+)", entry.get("title", ""))
        number = match.group(1) if match else "Unknown"
    return str(number)
