"""
Shared EU->US Chroma copy + validation logic.

These functions operate on already-constructed chromadb clients / collections,
so the exact same code paths run in the local self-test (PersistentClient) and
against Chroma Cloud (CloudClient). No credentials are read or printed here.

Guardrails honoured:
  - Embeddings copied verbatim (never recomputed).
  - Collections opened with embedding_function=None (we supply vectors).
  - get() paged at <= 250 (Cloud caps get at 300).
  - upsert => idempotent / resumable.
  - all-None columns collapsed to None (Chroma rejects [None, ...]).
  - source count re-checked at the end (proves the freeze held).
"""

import numpy as np

PAGE = 250  # Chroma Cloud caps get() at 300; 250 is the guide's safe value.
INCLUDE = ["embeddings", "documents", "metadatas", "uris"]


def drop_all_none(column):
    """Chroma rejects an all-None column; pass None for the whole argument."""
    if column is None:
        return None
    if all(v is None for v in column):
        return None
    return column


def _embeddings_to_list(embs):
    """Chroma returns embeddings as numpy; upsert wants plain lists."""
    if embs is None:
        return None
    if hasattr(embs, "tolist"):
        return embs.tolist()
    return embs


def get_collection_readonly(client, name):
    """Open a collection without an embedding function (we supply vectors)."""
    return client.get_collection(name=name, embedding_function=None)


def create_dest_collection(dst_client, src_col, dest_name=None):
    """
    Reproduce the source collection on the destination.

    `dest_name` defaults to the source's name, which is correct for a
    DB-to-DB copy. For a SAME-DATABASE re-ID it must be given: without it the
    name already exists on the "destination" (it is the source), the resume
    branch below hands back the SOURCE COLLECTION ITSELF, and the copy
    silently upserts v1 into v1 and reports success.

    There is deliberately no programmatic guard against that. The obvious one
    -- comparing `dst_client` against `src_col._client` -- is inert on
    chromadb 1.5.9, where `Client.get_collection` constructs
    `Collection(client=self._server, ...)`, so `_client` is the ServerAPI and
    never the CloudClient the caller holds. A guard that is always False is
    worse than none, because it reads as protection. The caller is responsible
    for passing a `dest_name` that differs from the source.

    Resume-safe: if it already exists on the destination (a re-run), reuse it.
    Otherwise copy the schema wholesale so distance space / index enablement /
    key-specific + sparse indexes carry over. Falls back to metadata-based
    creation if this build/collection predates the Schema API.
    """
    name = dest_name or src_col.name
    existing = {c.name for c in dst_client.list_collections()}
    if name in existing:
        print(f"  [dest] collection '{name}' already exists - reusing (resume).")
        return dst_client.get_collection(name=name, embedding_function=None)

    try:
        from chromadb import Schema

        js = src_col.schema.serialize_to_json()
        # Schema carries the distance space / index config, so passing metadata
        # alongside it errors ("Cannot set both collection config and schema").
        # Create from schema only.
        col = dst_client.create_collection(
            name=name,
            schema=Schema.deserialize_from_json(js),
        )
        print(f"  [dest] created '{name}' from copied schema.")
        return col
    except Exception as e:  # noqa: BLE001 - fall back deliberately
        meta = src_col.metadata or {"hnsw:space": "cosine"}
        col = dst_client.create_collection(name=name, metadata=meta)
        print(
            f"  [dest] schema round-trip unavailable ({type(e).__name__}: {e}); "
            f"created '{name}' from metadata={meta}."
        )
        return col


def copy_collection(src_col, dst_col):
    """Page the source and upsert into the destination. Returns count copied."""
    total = src_col.count()
    print(f"  source count: {total}")
    offset = 0
    copied = 0
    while offset < total:
        b = src_col.get(limit=PAGE, offset=offset, include=INCLUDE)
        ids = b["ids"]
        if not ids:
            break
        dst_col.upsert(
            ids=ids,
            embeddings=_embeddings_to_list(b.get("embeddings")),
            documents=drop_all_none(b.get("documents")),
            metadatas=drop_all_none(b.get("metadatas")),
            uris=drop_all_none(b.get("uris")),
        )
        copied += len(ids)
        offset += len(ids)
        print(f"  upserted {copied}/{total}")

    final = src_col.count()
    if final != total:
        raise SystemExit(
            f"SOURCE CHANGED during copy ({total} -> {final}). "
            f"A writer was not frozen. Aborting - do NOT cut over."
        )
    return copied


def _cell(batch, key, idx):
    col = batch.get(key)
    if col is None:
        return None
    if hasattr(col, "__len__") and len(col) <= idx:
        return None
    return col[idx]


def _emb_cell(batch, idx):
    embs = batch.get("embeddings")
    if embs is None:
        return None
    if hasattr(embs, "__len__") and len(embs) <= idx:
        return None
    return np.asarray(embs[idx], dtype=float)


def validate_collection(
    src_col, dst_col, atol=1e-4, id_map=None, allowed_new_keys=frozenset()
):
    """
    Full validation gate. Returns a list of problem strings (empty == clean).
    Compares counts, schema (best-effort), then every source id's document,
    metadata, uri and embedding against the destination, and finishes with an
    index self-query.

    `id_map` maps source id -> destination id, for a migration that re-IDs.
    Without it, destination lookups use source ids and every record of a
    re-IDed copy reports missing.

    `allowed_new_keys` are metadata keys the destination may carry that the
    source does not. Every other key must match exactly, and a key present in
    both must have the same value.
    """
    problems = []

    s_total = src_col.count()
    d_total = dst_col.count()
    print(f"  counts: src={s_total} dst={d_total}")
    if s_total != d_total:
        problems.append(f"count mismatch: src={s_total} dst={d_total}")

    try:
        s_schema = src_col.schema.serialize_to_json()
        d_schema = dst_col.schema.serialize_to_json()
        if s_schema != d_schema:
            problems.append("schema mismatch (serialized schemas differ)")
        else:
            print("  schema: identical")
    except Exception as e:  # noqa: BLE001
        print(f"  schema: comparison skipped ({type(e).__name__}: {e})")

    if (src_col.metadata or None) != (dst_col.metadata or None):
        print(
            f"  [warn] collection.metadata differs: "
            f"src={src_col.metadata} dst={dst_col.metadata}"
        )

    offset = 0
    checked = 0
    last_id = None
    last_emb = None
    while offset < s_total:
        b = src_col.get(limit=PAGE, offset=offset, include=INCLUDE)
        ids = b["ids"]
        if not ids:
            break
        d_ids = [id_map.get(i, i) for i in ids] if id_map else ids
        d = dst_col.get(ids=d_ids, include=INCLUDE)
        d_index = {i: k for k, i in enumerate(d["ids"])}
        for k, _id in enumerate(ids):
            mapped = id_map.get(_id, _id) if id_map else _id
            if mapped not in d_index:
                problems.append(f"missing id in dst: {_id} -> {mapped}")
                continue
            j = d_index[mapped]
            if _cell(b, "documents", k) != _cell(d, "documents", j):
                problems.append(f"document mismatch: {_id}")
            s_meta = _cell(b, "metadatas", k) or {}
            d_meta = _cell(d, "metadatas", j) or {}
            extra = set(d_meta) - set(s_meta) - set(allowed_new_keys)
            missing = set(s_meta) - set(d_meta)
            changed = {
                key for key in set(s_meta) & set(d_meta) if s_meta[key] != d_meta[key]
            }
            if extra or missing or changed:
                problems.append(
                    f"metadata mismatch for {_id}: "
                    f"extra={sorted(extra)} missing={sorted(missing)} "
                    f"changed={sorted(changed)}"
                )
            if _cell(b, "uris", k) != _cell(d, "uris", j):
                problems.append(f"uri mismatch: {_id}")
            se = _emb_cell(b, k)
            de = _emb_cell(d, j)
            if se is None or de is None:
                if se is not None or de is not None:
                    problems.append(f"embedding presence mismatch: {_id}")
            elif se.shape != de.shape or not np.allclose(se, de, atol=atol):
                problems.append(f"embedding mismatch: {_id}")
            last_id, last_emb = _id, se
        checked += len(ids)
        offset += len(ids)
        print(f"  validated {checked}/{s_total}")

    # Index self-query: query the destination with a real record's own vector
    # and expect that id back (or an identical-vector tie). Reported as a
    # problem, but note the guide's caveat about degenerate/colinear vectors.
    if last_id is not None and last_emb is not None:
        try:
            res = dst_col.query(
                query_embeddings=[last_emb.tolist()],
                n_results=5,
                include=["embeddings", "distances"],
            )
            hit_ids = res["ids"][0]
            ok = last_id in hit_ids
            if not ok:
                res_embs = res.get("embeddings")
                if res_embs is not None and len(res_embs) and len(res_embs[0]):
                    for cand in res_embs[0]:
                        if np.allclose(
                            np.asarray(cand, dtype=float), last_emb, atol=atol
                        ):
                            ok = True
                            break
            if ok:
                print(f"  self-query: OK (recovered {last_id} or identical-vector tie)")
            else:
                problems.append(
                    f"self-query failed: {last_id} not in top results {hit_ids}"
                )
        except Exception as e:  # noqa: BLE001
            problems.append(f"self-query errored: {type(e).__name__}: {e}")

    return problems


def list_summary(client):
    """(name, count) for every collection - safe to print, no credentials."""
    out = []
    for c in client.list_collections():
        col = client.get_collection(name=c.name, embedding_function=None)
        out.append((c.name, col.count()))
    return out
