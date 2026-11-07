# Project Memory Index

- [Chroma US-region migration](chroma-us-region-migration.md) — transcripts live SOLELY on US `geo-podcasts-us`; EU `geo-podcasts` deleted 2026-07-09 (no rollback); region set by `cloud_host` presence
- [Chroma migration scripts](chroma-migration-scripts.md) — reusable copy/validate tooling in `migration/`, plus chromadb 1.5.4 + Modal gotchas
- [Chroma v2 collection cutover](chroma-v2-collection-cutover.md) — live collection is `podcast_transcripts_v2`, set by `CHROMA_COLLECTION` in the Modal secret, not in the repo; v1 deleted 2026-09-02, no rollback exists
- [downloaded/ goes stale](downloaded-cache-goes-stale.md) — gitignored cache of the Modal volume; refresh before measuring anything, needs `MSYS_NO_PATHCONV=1`; 439 episodes at 2026-09-02, Observing Japan dormant not broken
- [Manual check over cloud credentials](manual-check-over-cloud-credentials.md) — he'd rather run a step by hand than provision a token to a cloud env; the friction rule has a credential boundary
- [Probe cheap before scoping](probe-cheap-before-scoping.md) — he takes the cheap disambiguating measurement first, then the thorough option; offer the probe as its own choice
- [Speaker ID: Tier 1 failed](speaker-id-tier1-failed.md) — text-only tier measured 2026-09-02 and dead; labels are now a dev set; voice embedding is the only path; resume from spec Part 8
- [Ship 2 rejected](ship-2-speaker-grouping-rejected.md) — archive-wide voice grouping measured and REJECTED 2026-09-04 (4 groups vs a required 15); do not re-propose without a new histogram
- [Ingest assigns speaker_id](new-episodes-lack-speaker-id.md) — gap CLOSED 2026-09-04 (1f9bb83, deployed); backfill before that date, ingest after; a $eq filter over a partly-absent key narrows silently
- [Speaker identity: next session](speaker-identity-next-session.md) — landing-order steps 1 AND 2 done 2026-09-08 (backlog 428 not 432 — exclusions; stamp applied, 28644/28644 verified); NEXT is step 3, expect `0` on all three shows, then the cap bump; phase 2 designed not built
- [What is actually deployed](what-is-actually-deployed.md) — `modal app list` shows CREATION not last deploy (use `app history`); MCP clients cache the tool schema until reconnect; a 401 never proves `MCP_ALLOWED_HOST`
- [Chroma Cloud retirement](chroma-cloud-retirement.md) — CUT OVER to self-hosted Chroma on Fly 2026-11-07 (3bfe827 deployed, verified live); follow-ups: first nightly + first weekly backup + restore drill, Fly uptime check on/after 2026-11-14; CLI `flyctl`, org alias `personal`
- [Hosting boundaries](hosting-boundaries.md) — never expose his DIY server to the internet; confirm a paid host's price against his current bill first
