---
name: chroma-us-region-migration
description: Podcast transcripts live SOLELY on US Chroma (geo-podcasts-us); EU geo-podcasts deleted 2026-07-09 (no rollback); region controlled by cloud_host presence
metadata: 
  node_type: memory
  type: project
---

On 2026-07-09 the `podcast_transcripts` collection (27,455 records) was migrated from the
EU Chroma Cloud DB `geo-podcasts` (`europe-west1.gcp.trychroma.com`) to the US DB
`geo-podcasts-us` (default region `aws-us-east-1`). **Same tenant for both** (the tenant UUID is
in the `CHROMA_TENANT` env var, deliberately not recorded here); only the database name + api key
+ region differed.

**Cutover is complete and the EU DB `geo-podcasts` was DELETED on 2026-07-09** after the user
confirmed all data was on US only. There is **no rollback path** anymore — US `geo-podcasts-us` is
the sole source of truth. If EU-region errors recur, this would be a fresh migration, not a revert.

Current live config:
- Code (`transcribe.py`, `mcp_server.py`, `upload_book.py`) no longer passes `cloud_host`/`cloud_port`
  to `chromadb.CloudClient`, so it **defaults to `aws-us-east-1`**. Region is selected purely by the
  presence/absence of `cloud_host` — to target another region, add `cloud_host=<region host>` back.
- Modal secret `podcast-secrets`: `CHROMA_API_KEY`=US key, `CHROMA_DATABASE`=`geo-podcasts-us`,
  `CHROMA_TENANT` unchanged, `HF_TOKEN` unchanged.
- Both apps redeployed against US; `podcast-transcriber`'s 09:00 UTC cron (`scheduled_job`) is active.

**Why:** EU Chroma had an availability issue; Chroma Cloud can't move a DB's region in place, so it was a
copy to a new US DB, validated, cut over, then the EU DB was removed.

**How to apply:** Treat `geo-podcasts-us` (aws-us-east-1) as authoritative. Do NOT reintroduce a
`cloud_host` EU override. See [[chroma-migration-scripts]] for the reusable copy/validate tooling if
another region move is ever needed.
