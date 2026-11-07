"""
Copy every collection from a source Chroma DB to a destination Chroma DB.

Credentials come ONLY from environment variables (never hard-coded/printed).
Each side is either Chroma Cloud or a self-hosted server behind a bearer token:

  Cloud:        <P>_KEY, <P>_TENANT, <P>_DB[, <P>_HOST]  (omit _HOST for aws-us-east-1)
  Self-hosted:  <P>_URL, <P>_TOKEN   (e.g. DST_URL=https://<your-app>.fly.dev)

where <P> is SRC or DST. Setting <P>_URL selects self-hosted for that side.
Used 2026-07-09 for EU Cloud -> US Cloud, and 2026-11 for Cloud -> Fly.

Optional:
  ALLOW_NONEMPTY_DST=1   proceed even if the destination already has records
                         (needed only for an intentional resume/re-run).

Usage:
  python migrate.py
"""

import os
import sys
from urllib.parse import urlparse

import chromadb

from chroma_migrate import (
    copy_collection,
    create_dest_collection,
    get_collection_readonly,
    list_summary,
)


def is_self_hosted(prefix):
    return bool(os.environ.get(f"{prefix}_URL"))


def make_client(prefix):
    if is_self_hosted(prefix):
        return make_http(prefix)
    return make_cloud(prefix)


def make_http(prefix):
    url = urlparse(os.environ[f"{prefix}_URL"])
    token = os.environ[f"{prefix}_TOKEN"]
    if url.scheme not in ("http", "https") or not url.hostname:
        sys.exit(f"ABORT: {prefix}_URL must be an http(s) URL with a host.")
    ssl = url.scheme == "https"
    return chromadb.HttpClient(
        host=url.hostname,
        port=url.port or (443 if ssl else 80),
        ssl=ssl,
        headers={"Authorization": f"Bearer {token}"},
    )


def make_cloud(prefix):
    key = os.environ[f"{prefix}_KEY"]
    tenant = os.environ[f"{prefix}_TENANT"]
    database = os.environ[f"{prefix}_DB"]
    host = os.environ.get(f"{prefix}_HOST")
    kwargs = dict(api_key=key, tenant=tenant, database=database)
    if host:
        kwargs.update(cloud_host=host, cloud_port=443)
    return chromadb.CloudClient(**kwargs)


def main():
    src = make_client("SRC")
    dst = make_client("DST")

    print("== Smoke test ==")
    src_summary = list_summary(src)
    dst_summary = list_summary(dst)
    print(f"  SRC collections: {src_summary}")
    print(f"  DST collections: {dst_summary}")

    dst_total = sum(n for _, n in dst_summary)
    if dst_total and os.environ.get("ALLOW_NONEMPTY_DST") != "1":
        sys.exit(
            f"ABORT: destination is not empty ({dst_total} records). "
            f"A non-empty dest usually means wrong credentials. "
            f"Set ALLOW_NONEMPTY_DST=1 only for an intentional resume."
        )

    if not src_summary:
        sys.exit("ABORT: source has no collections - check SRC_* credentials/host.")

    if os.environ.get("SMOKE_ONLY") == "1":
        print("\nSMOKE OK: credentials valid, source non-empty, destination empty.")
        return

    print("\n== Copy ==")
    for name, _count in src_summary:
        print(f"Collection: {name}")
        src_col = get_collection_readonly(src, name)
        dst_col = create_dest_collection(dst, src_col)
        copy_collection(src_col, dst_col)

    print("\nCopy complete. Now run: python validate.py")


if __name__ == "__main__":
    main()
