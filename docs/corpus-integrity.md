# Corpus integrity: the invariants and what broke them

The corpus lost episodes. Not to a crash — quietly, to an identity bug, while
every log line said success. This page records what the failure actually was,
the invariants that now prevent it, and the reasoning behind each, so that a
later change does not helpfully undo one.

The design document is
[`superpowers/specs/2026-08-31-corpus-integrity-design.md`](superpowers/specs/2026-08-31-corpus-integrity-design.md).
This page is the durable summary.

## What went wrong

Chunk ids were built from **show + episode number** only:

```
{show}-ep{episode_number}-{chunk_index}
```

Nothing in that identity distinguishes two episodes that share an episode
number — and episodes routinely do. Feeds publish no `itunes_episode` for some
entries, and the pipeline falls back to the literal string `"Unknown"`, so
*every* unnumbered episode of a show collided into one id space. Writing the
second one overwrote the first. No error was raised, because an upsert onto an
existing id is a legitimate operation.

Three mechanisms compounded it:

1. **Id collision** — the overwrite above.
2. **The 300-record cap** — a 431-chunk episode's upsert failed the cap, so
   episodes were stored partially.
3. **Presence checks** — the code asked "does any chunk exist for this
   episode?" A collision-clobbered episode with one surviving chunk answered
   yes, so the self-healing path never repaired it, and reconciliation passed.

Any one of these alone is survivable. Together they made a corpus that was
missing content while reporting itself healthy.

## Invariant 1: identity includes the date

```
{show}-ep{episode_number}-{date}-{chunk_index}
```

The triple `(show, episode_number, date)` is the episode key everywhere —
ids, filters, reconciliation. `corpus/identity.py` owns its construction and
nothing else may build an id by hand.

**The triple is not durable, and that is a known limitation, not an
oversight.** Episodes numbered `"Unknown"` are only unknown until the feed
backfills a number, at which point the triple changes and anything keyed on
the old one stops matching. That is exactly why `episode_guid` exists as a
second, stable identifier — see invariant 4.

Two episodes of the same show published on the same day, both unnumbered,
would still collide. Nothing currently guards that, because the id scheme
cannot represent it; it would have to be fixed in the scheme itself.

## Invariant 2: write is upsert-then-prune, never delete-then-upsert

`corpus/writing.py::upsert_then_prune` writes every chunk, then removes the
ids the episode no longer occupies.

The order is the whole point. Under delete-first, a full-archive re-embed
opens one destructive window per healthy episode. And the failure that matters
is not an exception: the job runs under a 7200s timeout covering a whole show,
and a Modal timeout kills the container **without raising anything an `except`
can see**. Delete-first, that is a deleted-and-never-rewritten episode with no
log line. Upsert-first, a crash leaves a **superset** — never a hole — and the
next run prunes it.

Two scoping rules on the prune, both learned the hard way:

- **The guid arm is `$and`-scoped by show.** A cross-post can carry the same
  guid as the episode it copies, so an unscoped guid arm reaches into another
  show's records. Cost of the narrowing: a genuine show rename now strands old
  records instead of cleaning them up. Accepted — a recoverable orphan beats an
  unrecoverable cross-show delete.
- **Non-episode records are never pruned.** `upload_book.py`'s chunks reuse
  episode-shaped metadata and can share a triple by construction. Proved, not
  assumed: seeding 191 book records and writing the book's own triple pruned
  all 191. The guard costs permanent un-prunability for anything stamping a
  `source` other than `"podcast"` — accepted, because a stray record is
  recoverable by hand and a destroyed corpus is not.

## Invariant 3: completeness, never presence

`corpus/store.py::is_complete` requires the stored index set to equal
`range(n_chunks)`. Not "some chunk exists", and not merely a matching count.

Count alone was not enough. A count check and reconciliation's contiguity
check disagreed on exactly one shape — an episode whose ids are contiguous but
**start above zero**. The count matches, so the planner called it complete and
skipped it forever, while reconciliation faulted it forever. A permanent
disagreement between the repairer and the auditor is worse than either verdict
alone, because neither side converges and there is no repair path at all.

Over-count is also incomplete, deliberately: orphans left by a longer previous
version must still trigger the prune.

`n_chunks` is stored per record so completeness is checkable without
re-deriving it. A record with no `n_chunks` (everything pre-migration) is
treated as **incomplete**, never as satisfied — the alternative means old
episodes are never checked at all.

## Invariant 4: `episode_guid` is stamped and forwarded

The feed's guid is stored on every record and threaded through the planner to
the exclusion check.

An arm no live path reaches is not protection, it is decoration — and this one
was decoration for a while: the writer never read the feed's guid, so
`episode.get("guid")` was always `None` and the guid arm was dead in
production while looking implemented. `tests/test_showplan.py` now pins the
forward, verified by mutation.

## Invariant 5: exclusions are a human list, not a heuristic

`corpus/exclusions.py` holds four cross-posted episodes by hand.

Cross-posts cannot be detected reliably: enclosure URLs differ across feeds
(Captivate re-hosts, zero shared URLs), so the only available signal is a
fuzzy title-and-date match. **Code that silently discards an episode because a
regex thought two titles matched is a failure mode with no alarm on it.**
Reconciliation therefore *reports* suspected cross-posts; exclusion is a
decision a human records.

Two arms, either sufficient: the triple, and the guid scoped by show. Both are
needed — the triple survives an episode ageing off its feed (six episodes have,
and can never be assigned a guid), and the guid survives an episode-number
backfill within a show.

**Guid sharing is a property of the individual cross-post, not of the
platform.** Measured 2026-09-01: the 2026 pair share their originals' guids
(which is what made an unscoped arm exclude the originals), and the 2025 pair
do not. There is no way to tell which kind you have without checking the feed,
which is why the scoping is unconditional.

## Invariant 6: the auditor is independent of the repairer

`corpus/reconcile.py` compares three sources — the transcript volume, the
store, and the live feeds — and reports faults it does not fix.

Fault fields (any non-empty means unclean): `missing`, `extra`,
`non_contiguous`, `shared_prefixes`, `excluded_with_records`, `incomplete`.

Informational, never faults: `feed_unreachable` (episodes that aged off their
feed are a fact, not a problem) and `suspected_cross_posts` — which is
**never populated today**, because no episode titles reach `reconcile()`. An
empty list there means "this check does not run", not "none were found".

## What is still open

- **Same-day unnumbered collisions.** Two episodes of one show, same date,
  both `"Unknown"`, generate identical ids. The scheme cannot represent them.
- **Index-shifted repair.** `is_complete` now *detects* an episode whose
  indices start above zero, and the fix is a re-embed, which the planner will
  now schedule. But nothing repairs a partially-shifted episode in place.
