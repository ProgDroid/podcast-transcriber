---
name: hosting-boundaries
description: "Never propose exposing his DIY home server to the internet; confirm a paid host's price against his current bill before choosing it"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 24cdeabe-8957-4646-96f1-bb20fa34d62f
  modified: 2026-11-07T18:06:48.367Z
---

**Never propose exposing his DIY/home server to the internet.** It is for LAN or temporary local use only. Anything internet-facing goes on a chosen host. Stated 2026-11-07: "we're either doing it as a temporary measure to test the bits you want to test, or we choose a host and migrate there straight away".

**Before choosing a paid host, confirm its pricing against what he pays today.** On 2026-11-07 he said "let's confirm fly.io pricing first, I'm currently not paying anything" before agreeing to Fly. His personal Fly org bills under the $5 that Fly does not collect.

**In a public repo, keep the hostname of a scale-to-zero service out of committed files.** Refused requests still wake the machine and bill it. On 2026-11-07 he chose to scrub the name from unpushed history and leave the app as it was, rather than create a randomly named app. The cost is bounded and visible, so he matched the effort to it. A scrape-proof name was enough; it did not need to be unguessable.

**Why:** the home server is a security boundary he will not cross. Cost matters as a threshold: staying under a free line beats a slightly better host.

**How to apply:** when a service needs a public endpoint, offer hosts, not his server. For local tests, use Docker on this machine. Before recommending a host, find out what he pays now and size the option to stay under his threshold (e.g. Fly auto-stop). Related: [[manual-check-over-cloud-credentials]], [[chroma-cloud-retirement]].
