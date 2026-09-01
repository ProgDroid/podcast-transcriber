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
        # get_collection, NOT get_or_create_collection. All three call sites
        # used get-or-create, so a typo at cutover would silently create an
        # empty third collection and make search_podcasts return
        # "No results found." instead of erroring. The reader must fail loudly;
        # the writer may still create.
        self.collection = self.chroma_client.get_collection(
            name=os.environ.get("CHROMA_COLLECTION", "podcast_transcripts"),
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
                    # Cutover tell: populated proves v2, None proves v1 or a
                    # stale warm container. Kept permanently.
                    "n_chunks": meta.get("n_chunks"),
                    "episode_guid": meta.get("episode_guid"),
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

    # `host` IS load-bearing here. In the mcp version this image installs, it
    # becomes the allowed-Host value for the transport's DNS-rebinding
    # protection, so it must match the Host header clients actually send or
    # every request comes back 421 "invalid host header". It is not merely a
    # uvicorn bind address.
    #
    # Verified the hard way: removing it on the belief it was inert took the
    # endpoint down with 421s. The belief came from introspecting the mcp
    # package installed on the DEV machine, where TransportSecurityMiddleware
    # defaults protection off. That is a different version from the one in this
    # image, so the local reading said nothing about the deployed behaviour.
    #
    # WARNING: `mcp` is pip-installed unpinned in the image above, so a future
    # image rebuild can resolve a version where this kwarg no longer feeds
    # validation, silently changing behaviour. Pin `mcp` before relying on this.
    #
    # Required, with no default on purpose. A wrong-but-present default fails as
    # a 421 on every request, which reads like a client problem; a missing key
    # fails loudly at startup instead. Set it to this deployment's own
    # *.modal.run hostname. See README.
    mcp = FastMCP(
        "Podcast Transcript Search",
        stateless_http=True,
        json_response=True,
        host=os.environ["MCP_ALLOWED_HOST"],
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
