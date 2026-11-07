---
name: chroma-cloud-retirement
description: "Chroma Cloud shuts down (Replit acquisition): read-only from 2026-11-22, reads stop 2026-12-22; the live corpus must move off it"
metadata:
  node_type: memory
  type: project
  originSessionId: 24cdeabe-8957-4646-96f1-bb20fa34d62f
  modified: 2026-11-07T18:06:47.703Z
---

Chroma Cloud is being retired after Chroma agreed to join Replit (announced 2026-10-23, per secondary sources; Chroma's own notice not read). It goes **read-only on 2026-11-22**, which breaks the nightly ingest's writes. It **stops serving reads on 2026-12-22**, which kills `search_podcasts`. Self-hosted OSS Chroma (Apache 2.0) is unaffected.

**Why:** the US db `geo-podcasts-us` is the only copy of the corpus ([[chroma-us-region-migration]], [[chroma-v2-collection-cutover]]).

**State 2026-11-07:** he chose self-hosted Chroma and ruled out his DIY server for anything internet-facing. Fly.io is the leading host. His personal Fly org already runs one small app and bills under $5 a month. Fly does not collect bills under $5, per his emails; the docs don't say this. So Chroma must **auto-stop** to keep the total under $5. Ask him for current figures; they are deliberately not recorded here, because this repo is public and its memory folder is committed. Local measurements are in `docs/self-hosted-chroma-behaviour.md`: 1.5.9 has no built-in auth (a proxy is required), and steady state is ~180 MiB. fly-mcp cannot create apps, so he ran `flyctl apps create`. He also wants a second copy kept outside Fly. He chose option 2: a Modal Volume export. He can regenerate the corpus from the podcasts and audio, so backups save time and spend, not irreplaceable data. **Done and committed 2026-11-07:** `backup.py` and `corpus/backup.py` were written, and the first Cloud export (30,483 records) was verified and its restore drilled locally. See `docs/backups.md`. Chroma 1.5.9 is still the latest stable release, so no version bump is needed.

**Deployed 2026-11-07:** he ran the flyctl commands; `deploy/chroma/` is live on Fly (1 GB, auto-stop). Measured: heartbeat 200; no, wrong or empty token gives 401; the right token gives 200; http redirects 301 to https; DNS resolves to one shared IPv4 (no dedicated IP). The token is in his Windows user env as `DST_CHROMA_TOKEN`. On his machine the CLI is **`flyctl`**, not `fly`, and pass the org as **`personal`** (flyctl's alias for the personal org; the API reports a different slug, but he said `personal` is what he had to use): write commands that way. The fly-mcp grant is per app: it refused every call on the new app until he widened it on 2026-11-07, so after creating an app, ask him to add it to the grant. `/api/v2/version` reports `"1.0.0"` on the pinned 1.5.9 image locally too: it is the API version, not the image tag.

**Copied 2026-11-07 (55399d8):** smoke, copy and validate all ran on Modal via `migration/modal_run.py` (secrets `podcast-secrets` + `chroma-selfhosted`). The Fly copy holds 30,483/30,483 records and validate PASSED: cosine on both sides, 0 problems. He removed the stale local `CHROMA_*` env vars on 2026-11-07. The only Cloud key left is in `podcast-secrets`. For local self-hosted access, `DST_CHROMA_TOKEN` is in his user env. **The Fly copy goes stale with every nightly ingest until cutover.** At cutover, run `--step copy --resume` and then `--step validate` again.

**CUT OVER 2026-11-07 ~20:50 UTC (3bfe827, deployed):** `transcribe.py`, `mcp_server.py` and `backup.py` are deployed and read and write Fly through `corpus/chroma_client.py` (secret `chroma-selfhosted`, no Cloud fallback). The machine is at 512 MB, and fly.toml says so. Validated twice beforehand: 30,483 records, 0 problems. Proof it is live: a connector `search_podcasts` call (speaker=Marko Papic) returned results, and Fly logged a proxy start at that same second. Chroma Cloud is now unused and can simply be left to expire.

**The Fly app name is deliberately absent from the repo** (decided 2026-11-07; the repo is public and the name is the hostname). fly.toml has no `app =` line, so pass `-a <app>`; the examples use `<your-app>.fly.dev` and `chroma.example.test`. The three unpushed commits that named it were rewritten (now 7869e73, 55399d8 and 3bfe827), and a check of every patch, message and tree on main found 0 hits. **Never write the name into the repo, docs or this memory folder**, which is committed. Get it from fly-mcp `list_apps` or the `chroma-selfhosted` secret. If the uptime check shows bots anyway, the next step is a randomly named new app.

**Follow-up checks (the agent owns these, not him):**
- After the 2026-11-08 09:00 UTC nightly, the Fly count should rise above 30,483 if new episodes landed.
- After Sun 2026-11-08 10:00 UTC, the first scheduled backup should appear (`backend: chroma-selfhosted`); then do a restore drill from it.
- **On or after 2026-11-14:** check the machine's uptime through fly-mcp `list_machine_events`. Refused requests also wake it. Its name is kept out of the repo, but a name can leak in other ways. Measured: it stops about 6.5 min after the last request, so only a hit every ~6 min around the clock would keep it always on (~$3.69/mo, which could push the org past the $5 no-collect line). If it is awake far more than his own use explains, propose an unguessable app name (Fly apps cannot be renamed: new app plus copy).

**How to apply:** treat any Chroma Cloud work as migration work. Options were weighed on 2026-11-07: self-hosted Chroma (reuses `migration/` copy and validate tooling), Qdrant, or pgvector. `docs/chroma-cloud-behaviour.md` facts are Cloud-only and must be re-measured on whatever replaces it. Other consumers: news-brief and the claude.ai connector reach it only through the MCP URL. The Aegypt OpenWebUI container is OSS Chroma, so it is unaffected.
