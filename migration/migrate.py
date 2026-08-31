"""
Copy every collection from the EU source Chroma Cloud DB to the US dest DB.

Credentials come ONLY from environment variables (never hard-coded/printed):

  SRC_KEY, SRC_TENANT, SRC_DB, SRC_HOST   (EU source; SRC_HOST is the region host)
  DST_KEY, DST_TENANT, DST_DB[, DST_HOST] (US dest; omit DST_HOST for aws-us-east-1)

Optional:
  ALLOW_NONEMPTY_DST=1   proceed even if the destination already has records
                         (needed only for an intentional resume/re-run).

Usage:
  python migrate.py
"""

import os
import sys

import chromadb

from chroma_migrate import (
    copy_collection,
    create_dest_collection,
    get_collection_readonly,
    list_summary,
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
    src = make_cloud("SRC")
    dst = make_cloud("DST")

    print("== Smoke test ==")
    src_summary = list_summary(src)
    dst_summary = list_summary(dst)
    print(f"  SRC (EU) collections: {src_summary}")
    print(f"  DST (US) collections: {dst_summary}")

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
