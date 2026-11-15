"""Re-key episodes a feed renumbered, instead of re-transcribing them.

WHY THIS EXISTS. Around 2026-11-12 the Jacob Shapiro feed shifted every
`<itunes:episode>` by +72 (three by +73). The planner finds a transcript by
its filename, `Show - Episode N - date.txt`, so all 301 archive episodes
looked untranscribed, and each night spent its 2h timeout re-transcribing
them. The transcripts and their records were fine; only the number had moved.

THE MATCH IS BY GUID, NEVER BY THE SHIFT. A feed entry with no transcript
under its own number is mapped to the triple its guid is stored under in
Chroma. Arithmetic would miss the +73 episodes and would happily pair two
different same-day episodes. Anything the guid does not settle uniquely is
reported, not guessed: a wrong rename moves one episode's transcript onto
another's feed entry, and nothing downstream would notice.

ORPHANS are transcript files whose triple is in neither the feed nor the
store: the old copies of episodes already re-transcribed under their new
number before the guard landed, whose records the guid prune in
`upsert_then_prune` already removed. They are harmless to the planner, which
enumerates from the feed, but `bulk_embed` enumerates the VOLUME and would
re-embed them as duplicates.

Pure: no Modal, no Chroma, no filesystem. `migration/renumber.py` does the I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from corpus.exclusions import is_excluded
from corpus.identity import chunk_id, episode_id_prefix

Triple = tuple[str, str, str]


@dataclass(frozen=True)
class Rename:
    show: str
    date: str
    guid: str
    old_number: str
    new_number: str


@dataclass
class RenumberPlan:
    renames: list[Rename] = field(default_factory=list)
    # Feed entries with no file and no stored guid: genuinely new episodes,
    # which the nightly run should transcribe. Counted, not acted on.
    new: int = 0
    unresolved: list[str] = field(default_factory=list)
    orphans: list[Triple] = field(default_factory=list)


def plan_renumber(
    show: str,
    feed_episodes: list[dict],
    *,
    volume_triples: set[Triple],
    stored_triples: set[Triple],
    stored_by_guid: dict[str, set[tuple[str, str]]],
) -> RenumberPlan:
    """Map each feed entry missing its transcript back to the one it has.

    `stored_by_guid` maps a guid to the (episode_number, date) pairs this
    show's records carry it under. `stored_triples` is every triple in the
    store for this show, guid or not.
    """
    plan = RenumberPlan()
    feed_triples = {(show, e["episode_number"], e["date"]) for e in feed_episodes}
    renamed_from: set[Triple] = set()

    for episode in feed_episodes:
        number, date, guid = episode["episode_number"], episode["date"], episode["guid"]
        if (show, number, date) in volume_triples:
            continue
        if is_excluded(show, number, date, guid):
            continue
        label = f"{show} ep{number} ({date})"
        if not guid:
            plan.unresolved.append(f"{label}: no transcript and no guid to match on")
            continue

        olds = {t for t in stored_by_guid.get(guid, set()) if t != (number, date)}
        if not olds:
            plan.new += 1
            continue
        if len(olds) > 1:
            plan.unresolved.append(
                f"{label}: guid {guid} is stored under {len(olds)} triples "
                f"{sorted(olds)}"
            )
            continue
        old_number, old_date = olds.pop()
        old = (show, old_number, old_date)
        if old_date != date:
            plan.unresolved.append(
                f"{label}: guid {guid} is stored under ep{old_number} dated "
                f"{old_date}; the date moved, not just the number"
            )
            continue
        if old not in volume_triples:
            plan.unresolved.append(
                f"{label}: guid {guid} is stored under ep{old_number}, but no "
                f"transcript file exists for that triple"
            )
            continue
        if old in feed_triples:
            plan.unresolved.append(
                f"{label}: old triple ep{old_number} ({date}) is itself still in "
                f"the feed; renaming would take its transcript away"
            )
            continue
        plan.renames.append(Rename(show, date, guid, old_number, number))
        renamed_from.add(old)

    for triple in sorted(volume_triples):
        if triple in feed_triples or triple in stored_triples or triple in renamed_from:
            continue
        _, number, date = triple
        if is_excluded(show, number, date):
            continue
        plan.orphans.append(triple)

    return plan


def rekey(
    ids: list[str],
    metadatas: list[dict],
    *,
    show: str,
    old_number: str,
    new_number: str,
    date: str,
) -> tuple[list[str], list[dict]]:
    """New ids and metadata for one episode's records, same chunk indices.

    Refuses rather than guesses on any record that is not plainly this
    episode's: an id outside the old prefix, or metadata naming another
    episode, means the selection was not what the caller believed.
    """
    old_prefix = episode_id_prefix(show, old_number, date)
    new_prefix = episode_id_prefix(show, new_number, date)
    new_ids: list[str] = []
    new_metas: list[dict] = []
    for record_id, meta in zip(ids, metadatas, strict=True):
        head, _, index = record_id.rpartition("-")
        if head != old_prefix or not index.isdigit():
            raise ValueError(f"{record_id!r} is not a chunk of {old_prefix!r}")
        if meta.get("episode_number") != old_number or meta.get("date") != date:
            raise ValueError(
                f"{record_id!r} metadata names ep{meta.get('episode_number')} "
                f"({meta.get('date')}), not ep{old_number} ({date})"
            )
        new_ids.append(chunk_id(new_prefix, int(index)))
        new_metas.append({**meta, "episode_number": new_number})
    return new_ids, new_metas


def rewrite_header(text: str, show: str, old_number: str, new_number: str) -> str:
    """Renumber the transcript's first header line; anything else is untouched.

    Cosmetic -- chunking ignores `#` lines -- but a file whose header and
    filename disagree is a trap for whoever reads it next.
    """
    old_line = f"# {show} - Episode {old_number}"
    first, sep, rest = text.partition("\n")
    if first.rstrip("\r") != old_line:
        return text
    return f"# {show} - Episode {new_number}" + first[len(old_line) :] + sep + rest
