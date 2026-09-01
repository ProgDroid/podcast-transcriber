# Chroma Cloud: measured behaviour

Everything on this page was **measured**, not read from documentation. Each
fact was established against a throwaway collection on Chroma Cloud
(`geo-podcasts-us`, chromadb client **1.5.9**), created and dropped by the
probe, with the production collections untouched. Non-raising results were
verified by count delta and re-get, never by the absence of an exception.

These behaviours are why several things in this repo look more defensive than
they need to be. If you are tempted to simplify one of them, the reason it
exists is here.

`tests/conftest.py`'s `FakeCollection` encodes exactly these facts and
nothing else. **Do not teach it a behaviour that has not been measured** — its
whole value is that a test passing against it means something.

## The 300-record request cap

| Operation | Limit | Behaviour past the limit |
|---|---|---|
| `get(limit=N)` | 300 | `N=301` **raises** |
| `get()` with no limit | 300 | **Silently returns 300.** No error, no warning. |
| `upsert(...)` | 300 | >300 **raises** |
| `delete(ids=[...])` | 300 | 301 **raises, atomically** — count unchanged, not a partial delete |
| `delete(where={...})` | **uncapped** | 400 records removed cleanly in one call |

The cap counts **records named in the request**, not records affected. That is
why the `ids` form is capped and the `where` form is not — they are different
request shapes with different limits.

**The silent one is the dangerous one.** An unlimited `get()` against a
431-chunk episode returns 300 and raises nothing, so a naive count reports the
episode short, judges a healthy episode torn, and re-embeds it on every run
forever. This is why `corpus/store.py` has `paged_get` and why calling the
unpaged form is a bug rather than a style choice.

Batching at 250 (`corpus/writing.py`'s `BATCH`) is measured-safe: a 250-record
delete and a 300-record delete both succeed cleanly; 301 raises.

## Filter semantics

**`$eq` does not match a record that lacks the key.** Probed directly: an
`$eq` on `episode_guid` returned only the record carrying that key, not the
records with no `episode_guid` at all.

**`$ne` DOES include records lacking the key.** `$eq` and `$ne` are *not*
complements over a corpus with heterogeneous metadata — and this corpus is
heterogeneous, because pre-migration records carry no `episode_guid` and book
records carry no episode fields. `FakeCollection` raises on `$ne` so nothing
in this repo depends on it today; anyone implementing it must not assume
symmetry.

The `$eq` fact is load-bearing: it is what makes book records (which have no
`episode_guid` key) safe from the prune's guid arm. That safety is measured,
not argued.

## Paged reads under a filter are stable

`corpus/store.py`'s `paged_get_ids` walks a filtered collection with
`limit`/`offset` and stops on an empty page. Chroma documents no total
ordering, so this was an open question with a real failure attached: unstable
ordering means duplicate or missed ids, a wrong count, and a healthy episode
re-embedded nightly forever.

Measured: 600 records seeded with the exact filter shape `episode_where`
emits (`$and` of three `$eq`), swept twice with `limit=250`:

- sweep A: 600 ids, 600 unique — sweep B: 600 ids, 600 unique
- 0 missing, 0 duplicated, in both
- **A and B in identical order**

Filtered paging is complete, duplicate-free and stable across independent
sweeps.

## Upsert merges metadata

An upsert onto an existing id **replaces the document but merges the
metadata**. A key present on the old record and absent from the new one
survives.

This is relied upon rather than worked around — it is how records written
before a field existed acquire it without a backfill, and it is why
`bulk_embed` can pass `episode_guid=None` without erasing a guid the
migration stamped. The consequence to remember: **a metadata key cannot be
removed by rewriting the episode**, only by deleting the record.

## Collections and regions

- `chromadb.CloudClient` selects its region purely by the presence or absence
  of a `cloud_host` argument. With none, it defaults to `aws-us-east-1`.
- Chroma Cloud **cannot change a database's region in place**. A region move
  is a copy, a validation, and a cutover — see `migration/`.
- `get_or_create_collection` will happily **create** a typo'd name and serve
  zero results forever. The reader uses `get_collection` so a wrong name
  raises instead. See [operations.md](operations.md).
