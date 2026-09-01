"""Feed-entry facts that both the writer and the migration must agree on."""

from __future__ import annotations


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
