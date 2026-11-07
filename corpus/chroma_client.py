"""The one place that knows how to reach the corpus's Chroma server.

Since 2026-11 the corpus lives on a self-hosted Chroma 1.5.9 behind a
bearer-token proxy on Fly (deploy/chroma/), not on Chroma Cloud, which went
read-only on 2026-11-22. Every reader and writer builds its client here, from
two variables in the `chroma-selfhosted` Modal secret:

  CHROMA_URL    e.g. https://<your-app>.fly.dev
  CHROMA_TOKEN  the proxy's bearer token

There is deliberately no fallback to Chroma Cloud: a writer that quietly
reached a read-only Cloud would fail far from the cause. The collection name
still comes from CHROMA_COLLECTION in `podcast-secrets`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlparse

SECRET_NAME = "chroma-selfhosted"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ChromaConfigError(RuntimeError):
    """The Chroma connection settings are missing or unsafe."""


def connection_settings(env: Mapping[str, str] | None = None) -> dict:
    """HttpClient keyword arguments from CHROMA_URL and CHROMA_TOKEN.

    Plain http is refused except for a local server: the token would cross
    the network in the clear.
    """
    env = os.environ if env is None else env
    url = env.get("CHROMA_URL") or ""
    token = env.get("CHROMA_TOKEN") or ""
    if not url or not token:
        raise ChromaConfigError(
            "CHROMA_URL and CHROMA_TOKEN must both be set; on Modal, attach "
            f"the {SECRET_NAME!r} secret"
        )
    parsed = urlparse(url)
    host = parsed.hostname
    if parsed.scheme not in ("http", "https") or not host:
        raise ChromaConfigError("CHROMA_URL must be an http(s) URL with a host")
    ssl = parsed.scheme == "https"
    if not ssl and host not in _LOCAL_HOSTS:
        raise ChromaConfigError(
            f"CHROMA_URL must use https for a non-local host (got {host})"
        )
    return {
        "host": host,
        "port": parsed.port or (443 if ssl else 80),
        "ssl": ssl,
        "headers": {"Authorization": f"Bearer {token}"},
    }


def make_client(env: Mapping[str, str] | None = None):
    """A chromadb HttpClient for the corpus's server."""
    import chromadb

    return chromadb.HttpClient(**connection_settings(env))
