import modal
import os


def download_bge_model():
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "sentence-transformers",
        "chromadb",
        "fastapi",
        "mcp",
        "starlette",
    )
    .run_function(download_bge_model)
)

app = modal.App("podcast-mcp-server", image=image)


@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("podcast-secrets")],
)
class PodcastSearch:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer
        import chromadb

        print("Loading BGE model...")
        self.embedding_model = SentenceTransformer(
            "BAAI/bge-large-en-v1.5",
            device="cuda",
        )

        print("Connecting to ChromaDB...")
        self.chroma_client = chromadb.CloudClient(
            tenant=os.environ["CHROMA_TENANT"],
            database=os.environ["CHROMA_DATABASE"],
            api_key=os.environ["CHROMA_API_KEY"],
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="podcast_transcripts",
            metadata={"hnsw:space": "cosine"},
        )
        print("Ready.")

    @modal.method()
    def search(
        self,
        query: str,
        n_results: int = 10,
        show: str = None,
        after_date: str = None,
        before_date: str = None,
    ) -> list[dict]:
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        filters = []
        if show:
            filters.append({"show": {"$eq": show}})
        if after_date:
            filters.append({"date_ts": {"$gte": int(after_date.replace("-", ""))}})
        if before_date:
            filters.append({"date_ts": {"$lte": int(before_date.replace("-", ""))}})

        where = None
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        kwargs = dict(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "text": doc,
                    "show": meta.get("show"),
                    "episode_number": meta.get("episode_number"),
                    "episode_title": meta.get("episode_title"),
                    "date": meta.get("date"),
                    "speaker": meta.get("speaker"),
                    "start_time": meta.get("start_time"),
                    "relevance_score": round(1 - dist, 3),
                }
            )

        output.sort(key=lambda x: x.get("date") or "", reverse=True)
        return output


@app.function(
    secrets=[modal.Secret.from_name("podcast-secrets")],
)
@modal.concurrent(max_inputs=10)
# requires_proxy_auth gates the endpoint at Modal's edge, before a container is
# even started, so unauthenticated traffic never reaches this code and never
# costs a GPU cold start. Callers must send Modal-Key / Modal-Secret headers
# from a proxy auth token.
# Minting the token depends on client version: `modal workspace proxy-tokens
# create` on >=1.5, or the dashboard on 1.4.x where `modal workspace` does not
# exist. The flag itself works on both. See README.
@modal.asgi_app(requires_proxy_auth=True)
def mcp_server():
    from mcp.server.fastmcp import FastMCP

    # A `host=` kwarg used to be passed here, hardcoded to this deployment's
    # own *.modal.run name. It was doing nothing, and it is dropped rather than
    # parameterised. FastMCP stores it on Settings.host, which is only read by
    # run() when it binds uvicorn itself. Modal owns the serving, we hand it the
    # ASGI app from streamable_http_app(), and that method never looks at
    # settings.host.
    #
    # It is specifically NOT the allowed-Host value for DNS-rebinding
    # protection, which is easy to assume. That lives in a separate
    # `transport_security` argument, and TransportSecurityMiddleware disables
    # protection outright when none is passed, whatever the field default on
    # TransportSecuritySettings says.
    #
    # Access control here is Modal proxy auth, above. If host allowlisting is
    # ever wanted as well, pass transport_security=TransportSecuritySettings(
    # enable_dns_rebinding_protection=True, allowed_hosts=[...]) explicitly.
    mcp = FastMCP(
        "Podcast Transcript Search",
        stateless_http=True,
        json_response=True,
    )
    searcher = PodcastSearch()

    @mcp.tool()
    def search_podcasts(
        query: str,
        n_results: int = 10,
        show: str = None,
        after_date: str = None,
        before_date: str = None,
    ) -> str:
        """
        Search podcast transcripts semantically. Returns relevant chunks with
        speaker attribution, episode metadata, and publication date.
        Use this whenever the user asks about topics, opinions, or statements
        from the podcasts. Results are sorted most-recent first so recency is
        always visible — factor in publication date when reasoning about whether
        views may have changed.

        Args:
            query: What to search for, e.g. "China Taiwan" or "Fed interest rates"
            n_results: Number of results to return (default 10)
            show: Filter to a specific show name (optional)
            after_date: Only include episodes after this date, YYYY-MM-DD (optional)
            before_date: Only include episodes before this date, YYYY-MM-DD (optional)
        """
        try:
            results = searcher.search.remote(
                query=query,
                n_results=n_results,
                show=show,
                after_date=after_date,
                before_date=before_date,
            )
        except Exception as e:
            raise Exception(f"Search failed: {e}")

        if not results:
            return "No results found."

        lines = []
        for r in results:
            lines.append(
                f"[{r['date']}] {r['show']} - Episode {r['episode_number']} "
                f"({r['episode_title']}) @ {r['start_time']:.1f}s "
                f"[relevance: {r['relevance_score']}]\n{r['text']}\n"
            )
        return "\n---\n".join(lines)

    @mcp.tool()
    def latest_on_topic(topic: str, n_results: int = 5) -> str:
        """
        Find the most recent things said about a topic across all podcasts.
        Use this when the user wants the current or latest view on something,
        or when asking whether a position has changed recently.

        Args:
            topic: The topic to search for
            n_results: Number of results to return (default 5)
        """
        try:
            results = searcher.search.remote(query=topic, n_results=n_results)
        except Exception as e:
            raise Exception(f"Search failed: {e}")

        if not results:
            return "No results found."

        lines = []
        for r in results:
            lines.append(
                f"[{r['date']}] {r['show']} - Episode {r['episode_number']} "
                f"({r['episode_title']}) @ {r['start_time']:.1f}s\n{r['text']}\n"
            )
        return "\n---\n".join(lines)

    return mcp.streamable_http_app()
