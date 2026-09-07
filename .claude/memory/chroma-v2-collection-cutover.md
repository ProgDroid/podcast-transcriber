---
name: chroma-v2-collection-cutover
description: "Live collection is podcast_transcripts_v2 (re-IDed corpus), selected by CHROMA_COLLECTION in the Modal secret; v1 podcast_transcripts was DELETED 2026-09-02 - there is no rollback"
metadata: 
  node_type: memory
  type: project
  originSessionId: b447fd06-f75b-4b96-910c-dcd2173ba117
  modified: 2026-09-01T16:35:06.951Z
---

On 2026-09-01 the corpus was migrated within `geo-podcasts-us` from
`podcast_transcripts` to **`podcast_transcripts_v2`**, which is now the live
collection. The move existed to re-ID every record onto a date-bearing scheme —
the old IDs were keyed on show + episode_number only, so two episodes sharing an
episode number (or both falling back to the literal `"Unknown"`) collided and
silently overwrote each other.

**The collection name is NOT in the repo.** Both apps read
`CHROMA_COLLECTION` from the Modal secret `podcast-secrets`, defaulting to
`podcast_transcripts` if unset. So the live collection cannot be determined by
reading the code — check the secret, or use the tell below.

- **There is NO in-band version tell.** `n_chunks` was documented as one in
  the README, the runbook and here; it is put into the searcher's result dict
  (`mcp_server.py:126`) and never rendered into the string a caller receives,
  so no consumer could ever see it. Verified 2026-09-02 against the deployed
  endpoint and corrected in all three places. What answers "which collection is
  serving" is the `Reading from collection: <name>` line `load()` prints,
  authoritative once the app has been cycled.
- **v1 `podcast_transcripts` was DELETED on 2026-09-02** (28,541 records),
  after a clean reconciliation and with pre-registered counts asserted in the
  deletion script. **There is now no rollback path at all** for either the
  region move or the collection cutover -- `podcast_transcripts_v2` in
  `geo-podcasts-us` is the only copy of the corpus that exists, exactly as in
  [[chroma-us-region-migration]].
- The reader uses `get_collection`, not `get_or_create_collection`, so a wrong
  name raises instead of silently serving an empty collection.
- A warm container holds a `Collection` handle with no name to re-resolve, so
  after changing the secret both apps must be **cycled**, not waited out.

Record count after the repair and the approved deletions: **29,160**.

**Why:** episode identity, not storage, was the defect — see the corpus
integrity work merged to `main` as `4aa1138`. Two 2025 Jacob Shapiro cross-posts
(175 and 50 records) were permanently deleted with the user's explicit approval;
`corpus/exclusions.py` is what stops the 09:00 UTC cron restoring them.

**How to apply:** treat `podcast_transcripts_v2` as authoritative. Before
concluding anything about which collection is serving, check `CHROMA_COLLECTION`
rather than reading the source. There is nothing left to
delete and nothing to fall back to, so treat any destructive operation on
`podcast_transcripts_v2` as unrecoverable. See [[chroma-migration-scripts]] for
the copy/validate tooling.

**A transient feed failure can make reconciliation report `CLEAN: True` when a
show's verdict is actually UNKNOWN.** Seen 2026-09-02: The Observing Japan
Podcast's Substack feed failed to load, so all 7 of its episodes fell into
`FEED_UNREACHABLE` and `feed=` dropped 433 -> 426, while every fault field still
read 0. The two signatures together -- every episode of one show unreachable,
and a drop in `feed=` -- mean re-run before acting. A second run was clean.
