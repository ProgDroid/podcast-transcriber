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
