---
name: manual-check-over-cloud-credentials
description: He dropped a scheduled check rather than provision Modal tokens to a cloud environment — the friction rule has a credential boundary
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2afe646f-8773-4394-b138-12c570649dbe
  modified: 2026-09-08T15:27:51.894Z
---

**When automating a step would require putting a credential somewhere new, he will
do the step by hand instead.** Offered three ways to schedule the step-3 backlog
check on 2026-09-08, he picked none: *"let's just drop it then, I will start a
session at the right time tomorrow."* The blocking constraint was that a
`RemoteTrigger` routine runs as a cloud session with no access to `~/.modal.toml`,
so the only fully-automatic option required adding `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` to the cloud environment.

**Why:** it bounds the friction rule, which otherwise reads as absolute — "never
design a solution that adds a step he has to remember, put the remembering on the
agent instead". That rule loses to credential placement. Consistent with the
semgrep-token episode in the global CLAUDE.md: he treats a credential landing
somewhere new as a cost in its own right, not as setup.

**How to apply:** when scoping an automation, work out its credential requirement
BEFORE presenting it, and say so in the offer rather than discovering it midway.
Lead with the option that needs no new credential placement even when it is the
less automatic one — the reverse of the usual "offer the more thorough option
first" default. Do not push back when he declines; hand him the resume command
and the pre-registered expectation instead, which is what makes the manual path
cheap. **One observation, not yet a law** — if he declines a second credential-
provisioning offer, promote this to the global CLAUDE.md next to the friction
bullet.

Related: [[probe-cheap-before-scoping]], [[speaker-identity-next-session]]
