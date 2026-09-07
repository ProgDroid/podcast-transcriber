---
name: what-is-actually-deployed
description: "Three ways this project's tooling reports something adjacent to what is live, and the probe that actually answers it"
metadata: 
  node_type: memory
  type: project
  originSessionId: 51c8f7e3-2a7b-4afb-b12e-b856aad7cc42
  modified: 2026-09-07T19:03:10.850Z
---

**Nothing in the obvious place tells you what code is currently serving.**
Three measured instances, all 2026-09-07, all the same shape: a command
succeeded, printed something plausible, and answered a question next to the
one asked.

1. **`modal app list`'s "Created at" is the app's CREATION date, never its
   last deploy.** It showed `podcast-transcriber` as 2026-09-01 while
   `modal app history podcast-transcriber` showed the live version was **v6,
   deployed 2026-09-04** — a three-day error, made while deciding whether an
   ungated phase-1 build was already live. **`modal app history <app>` is the
   only answer**: it prints Version / Time deployed / Deployed by.

2. **An MCP client caches the tool schema per connection.** After deploying a
   tool-signature change to `mcp_server.py`, the running session still
   advertises the OLD schema — the new `speaker` argument was absent until
   `/mcp` reconnected, even though the server already served it. A deployed
   parameter is not a callable one.

3. **A 401 on the endpoint proves less than it looks.** An unauthenticated GET
   returning 401 proves the endpoint registered and Modal's edge gate works —
   a 404 would have meant it never registered. It says NOTHING about
   `MCP_ALLOWED_HOST`, because the 421 host check runs **inside** the
   container, past the edge, and the probe never reached one. Only a real
   authenticated call exercises it.

**The one probe that does answer it:** call the deployed thing and read a
field where old and new code DISAGREE. Verified v18 by first checking that
v17's `rendering.py:41` rendered `result.get("speaker")` — the raw cluster
label — so a resolved name like `Jacob Shapiro` could only come from v18's
`display_speaker`. Without that step, seeing names proves only that something
served.

**Why:** deploys here are gated on a landing order where acting on the wrong
belief about what is live is the expensive failure — deploying the identity
signal before the stamp queues all 433 episodes for audio re-download.

**How to apply:** before any gated deploy, `modal app history` for the live
version, then map it to a commit by timestamp. After deploying, verify with a
call whose output the previous version could not have produced. Redeploying
does NOT move the URL as long as the app name and the `@modal.asgi_app`
function name are unchanged — that is what keeps `MCP_ALLOWED_HOST` valid.

Related: [[speaker-identity-next-session]]
