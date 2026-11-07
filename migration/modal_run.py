"""
Run migrate.py / validate.py on Modal, where the live Chroma Cloud key lives.

The Cloud key exists only in the `podcast-secrets` Modal secret (the local
CHROMA_API_KEY is an old one scoped to the deleted EU database). The
self-hosted server's URL and token come from a second secret, so the copy
needs no edit to `podcast-secrets`:

  modal secret create chroma-selfhosted CHROMA_URL=https://... CHROMA_TOKEN=...

Usage (from the repo root):
  modal run migration/modal_run.py --step smoke
      read-only: credentials work, source non-empty, destination empty
  modal run migration/modal_run.py --step copy
      copy every collection (idempotent upserts)
  modal run migration/modal_run.py --step validate
      the cutover gate; fails on any mismatch
  modal run migration/modal_run.py --step copy --resume
      re-run into a non-empty destination

Writes go only to the destination; the source is read-only throughout.
"""

import os
import sys

import modal

app = modal.App("podcast-chroma-migrate")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb==1.5.9")
    .add_local_python_source("migration")
)

STEPS = ("smoke", "copy", "validate")


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("podcast-secrets"),
        modal.Secret.from_name("chroma-selfhosted"),
    ],
    timeout=3600,
    memory=2048,
)
def run(step: str, resume: bool = False) -> None:
    os.environ.update(
        SRC_KEY=os.environ["CHROMA_API_KEY"],
        SRC_TENANT=os.environ["CHROMA_TENANT"],
        SRC_DB=os.environ["CHROMA_DATABASE"],
        DST_URL=os.environ["CHROMA_URL"],
        DST_TOKEN=os.environ["CHROMA_TOKEN"],
    )
    if os.environ.get("CHROMA_HOST"):
        os.environ["SRC_HOST"] = os.environ["CHROMA_HOST"]
    if resume:
        os.environ["ALLOW_NONEMPTY_DST"] = "1"
    # migrate.py and validate.py import their siblings by bare name.
    sys.path.insert(0, "/root/migration")

    if step == "validate":
        import validate

        validate.main()
        return
    if step == "smoke":
        os.environ["SMOKE_ONLY"] = "1"
    import migrate

    migrate.main()


@app.local_entrypoint()
def main(step: str, resume: bool = False) -> None:
    if step not in STEPS:
        raise SystemExit(f"--step must be one of {STEPS}")
    run.remote(step, resume)
