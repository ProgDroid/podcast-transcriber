"""A Chroma stand-in that enforces the caps measured against Chroma Cloud.

Every limit here was verified empirically against a throwaway Cloud
collection on chromadb 1.5.9, not read from documentation:

  get(limit=301)          -> raises
  get() with no limit     -> SILENTLY returns 300
  upsert of >300 records  -> raises
  upsert on existing id   -> replaces the document, MERGES the metadata
  delete(ids=) of >300 records -> raises, atomically (count unchanged)
  delete(where=) matching nothing -> no-op
  delete(where=) matching >300 records -> succeeds, removes all of them --
    the cap counts records named IN THE REQUEST, not records affected, so
    the where-form is not capped the way the ids-form is
  $eq does not match a record missing the key; $ne DOES include it (not
    modelled here -- nothing in this codebase depends on $ne)

The silent one is why paging is not optional.
"""

from __future__ import annotations

import pytest

MAX_REQUEST = 300


class ChromaQuotaError(Exception):
    """Stands in for chromadb.errors.ChromaError on a quota breach."""


class FakeCollection:
    def __init__(self, name: str = "podcast_transcripts") -> None:
        self._docs: dict[str, str] = {}
        self._meta: dict[str, dict] = {}
        # migration/chroma_migrate.py's validate_collection reads .metadata at
        # line 158, OUTSIDE the try that guards .schema at 148-155. Without
        # these three attributes every Task 6 test errors with AttributeError
        # rather than failing on the assertion it is actually testing.
        self.name = name
        self.metadata = {"hnsw:space": "cosine"}
        self.schema = None
        # An observation, not an assertion -- a test reads this to check how
        # a caller batched its requests. Nothing here enforces anything.
        self.calls: list[tuple[str, int]] = []

    # -- helpers ---------------------------------------------------------
    def _matches(self, meta: dict, where: dict | None) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(self._matches(meta, c) for c in where["$and"])
        for key, cond in where.items():
            if isinstance(cond, dict):
                if "$eq" not in cond:
                    # Only $and/$eq and bare equality are modelled. Defaulting
                    # an unknown operator to True would silently misreport a
                    # filter this fake does not actually understand.
                    unknown = next(iter(cond))
                    raise ValueError(
                        f"FakeCollection does not implement operator {unknown!r}"
                    )
                if meta.get(key) != cond["$eq"]:
                    return False
            elif meta.get(key) != cond:
                return False
        return True

    def _select(self, where: dict | None) -> list[str]:
        return [i for i in sorted(self._docs) if self._matches(self._meta[i], where)]

    # -- the Chroma surface ---------------------------------------------
    def count(self) -> int:
        return len(self._docs)

    def upsert(self, ids, embeddings=None, documents=None, metadatas=None):
        if len(ids) > MAX_REQUEST:
            raise ChromaQuotaError(
                f"Quota exceeded: 'Number of records' exceeded quota limit for "
                f"action 'Upsert': current usage of {len(ids)} exceeds limit of "
                f"{MAX_REQUEST}."
            )
        for k, _id in enumerate(ids):
            if documents is not None:
                self._docs[_id] = documents[k]
            else:
                self._docs.setdefault(_id, "")
            if metadatas is not None:
                # MERGE, not replace -- verified on Cloud.
                merged = dict(self._meta.get(_id, {}))
                merged.update(metadatas[k])
                self._meta[_id] = merged
            else:
                self._meta.setdefault(_id, {})

    def get(self, ids=None, where=None, limit=None, offset=0, include=None):
        if limit is not None and limit > MAX_REQUEST:
            raise ChromaQuotaError(
                f"Quota exceeded: 'Limit value' exceeded quota limit for action "
                f"'Get': current usage of {limit} exceeds limit of {MAX_REQUEST}."
            )
        chosen = ids if ids is not None else self._select(where)
        selected = [i for i in chosen if i in self._docs]
        if ids is not None and where is not None:
            selected = [i for i in selected if self._matches(self._meta[i], where)]
        # An absent limit SILENTLY truncates at 300 on Cloud.
        effective = MAX_REQUEST if limit is None else limit
        window = selected[offset : offset + effective]
        out: dict = {"ids": window}
        if include and "documents" in include:
            out["documents"] = [self._docs[i] for i in window]
        if include and "metadatas" in include:
            out["metadatas"] = [dict(self._meta[i]) for i in window]
        return out

    def delete(self, ids=None, where=None):
        if ids is not None:
            # Measured on Cloud: an ids= delete over the cap raises, and the
            # count is UNCHANGED after -- an atomic reject, not a partial
            # delete. The where= form has no such cap (measured at 400).
            if len(ids) > MAX_REQUEST:
                raise ChromaQuotaError(
                    f"Quota exceeded: 'Number of records' exceeded quota limit "
                    f"for action 'Delete': current usage of {len(ids)} exceeds "
                    f"limit of {MAX_REQUEST}."
                )
            targets = list(ids)
        else:
            targets = self._select(where)
        self.calls.append(("delete", len(targets)))
        for _id in targets:
            self._docs.pop(_id, None)
            self._meta.pop(_id, None)


@pytest.fixture
def collection() -> FakeCollection:
    return FakeCollection()
