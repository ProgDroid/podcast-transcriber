"""
Validate the US destination against the EU source. Cutover gate.

Reads the same SRC_* / DST_* environment variables as migrate.py.
Exits non-zero on ANY mismatch. Do NOT cut over unless this exits clean.

Usage:
  python validate.py
"""

import sys

from migrate import make_cloud
from chroma_migrate import get_collection_readonly, validate_collection


def main():
    src = make_cloud("SRC")
    dst = make_cloud("DST")

    src_names = {c.name for c in src.list_collections()}
    dst_names = {c.name for c in dst.list_collections()}
    print(f"SRC collections: {sorted(src_names)}")
    print(f"DST collections: {sorted(dst_names)}")

    all_problems = []
    if src_names != dst_names:
        all_problems.append(
            f"collection set mismatch: only-in-src={sorted(src_names - dst_names)} "
            f"only-in-dst={sorted(dst_names - src_names)}"
        )

    for name in sorted(src_names & dst_names):
        print(f"\n== Validating '{name}' ==")
        src_col = get_collection_readonly(src, name)
        dst_col = get_collection_readonly(dst, name)
        problems = validate_collection(src_col, dst_col)
        for p in problems:
            print(f"  PROBLEM: {p}")
        all_problems.extend(problems)

    print("\n== Result ==")
    if all_problems:
        print(f"FAIL: {len(all_problems)} problem(s). Do NOT cut over.")
        sys.exit(1)
    print("PASS: destination matches source. Safe to cut over.")


if __name__ == "__main__":
    main()
