"""Group id to display name. The ONE place a name is written down.

Derived, never duplicated: Chroma holds a stable `speaker_id` and this maps it
to something a person reads, so naming a group is an edit here and nothing
else -- no Chroma write, no re-embed, and it propagates across all 433
episodes at once.

Ship 1's identities are people, so today this is close to an identity map. It
exists now anyway because Ship 2 writes `grp_NNN` keys into the same table and
the render path must not change shape between ships.
"""

from __future__ import annotations

NAMES: dict[str, str] = {
    "Jacob Shapiro": "Jacob Shapiro",
    "Rob Larity": "Rob Larity",
    "Marko Papic": "Marko Papic",
    "Tobias Harris": "Tobias Harris",
    "Matt Gertken": "Matt Gertken",
}


def speaker_filter_error(speaker: str | None) -> str | None:
    """The message to show for an unfilterable `speaker`, or None if it is fine.

    ONE definition, because both call sites in `mcp_server.py` need this rule
    and they need it in different shapes -- the Modal method raises, the tool
    surface returns a string to the caller. Two copies of a membership test
    is how one of them ends up stale.

    Why this exists at all: Chroma's `$eq` does not match a record missing the
    key, and does not match a near-miss either. So `"Jacob"` or
    `"jacob shapiro"` matches ZERO rows, and a zero-row search renders as
    "No results found." -- byte-identical to "this person exists and never
    discussed this topic". A typo becomes a confident negative about
    somebody's views.

    The message NAMES the five rather than merely rejecting, because a caller
    forced to recall an exact proper noun is being asked the wrong question;
    one picking from five candidates is not.
    """
    if speaker is None or speaker in NAMES:
        return None
    return (
        f"Unknown speaker {speaker!r}. This filter matches an exact stored "
        f"name, so a near-miss returns nothing rather than erroring. Known "
        f"speakers: " + ", ".join(sorted(NAMES)) + "."
    )
