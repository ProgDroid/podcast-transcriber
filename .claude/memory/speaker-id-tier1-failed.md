---
name: speaker-id-tier1-failed
description: Speaker identification — the cheap text-only tier was measured on 2026-09-02 and failed; voice embedding is the only remaining path
metadata: 
  node_type: memory
  type: project
  originSessionId: b1d639af-de60-496e-986e-aae48020777f
  modified: 2026-09-02T20:44:51.619Z
---

Speaker identification was designed in two tiers: Tier 1 assigns the show's host to whichever diarisation cluster holds the most speech (no GPU, claimed to cover 76% of the archive), and Tier 2 does voice embedding for the residual. **Tier 1 was measured against 379 hand labels on 2026-09-02 and fails.** The gate required zero misattributions; there are 24, and the dominant cluster is the host in under a third of episodes. Full write-up in `docs/superpowers/specs/2026-09-01-speaker-identification-eval-design.md` **Part 8**, with the resume instructions in that file's `NEXT SESSION STARTS HERE` block.

**Why:** the Jacob Shapiro Podcast has recurring co-hosts who out-talk the host — Rob Larity (40 labels) and Marko Papic (19). Larity was in no roster in the design. Adding him routes 149 of 356 episodes away, dropping Tier 1's coverage to 46%, and 7 errors survive a complete roster anyway because they occur in episodes that never name the speaker. The verdict does not depend on the 112 unlabelled clips: the ceiling that assumes every one of them is the host is 87.5%, still ten points under the gate.

**How to apply:** do not re-run the Tier 1 gate on `speakers/labels.json` — those 214 labels are a **dev set** now, valid as a voice-enrolment seed and invalid for any new precision claim. Before designing Tier 2, settle the two cheap Part 7 unknowns (whether the Hugging Face acceptance covers an embedding model; what the deployed whisperx image exposes) — both fail late and expensively and neither needs a GPU. And note the reframing that makes Tier 2 tractable: the user recognises voices but cannot recall names, so the deliverable is **stable grouping**, not naming. `unknown_1` is a usable identity. See [[downloaded-cache-goes-stale]] for the corpus state this was measured on.
