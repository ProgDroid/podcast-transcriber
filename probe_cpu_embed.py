"""Probe: can the search query be embedded on CPU instead of a T4?

Question (2026-11-16): a search_podcasts call timed out because Modal took
6 min 13 s to allocate a T4 for PodcastSearch. If one query embeds fast enough
on CPU, the GPU (and its scheduling wait) can be dropped from the search path.

Measures, in the deployed mcp_server image:
  - load time and per-query encode latency on CPU at 1, 2 and 4 cores
    (several repeats each, since one timing is an anecdote);
  - cosine similarity between the CPU and T4 embedding of the same query.
    Retrieval is unchanged if that is ~1.0 for every query.

Run:  modal run probe_cpu_embed.py
"""

import time

import modal

from mcp_server import image

app = modal.App("probe-cpu-embed", image=image)

QUERIES = [
    "China Taiwan",
    "Fed interest rates",
    "What does Marko think about European defence spending?",
    "Japan LDP leadership election and the yen",
    "oil",
]
REPEATS = 5
MODEL = "BAAI/bge-large-en-v1.5"


def _measure(device: str) -> dict:
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL, device=device)
    load_s = time.perf_counter() - t0

    embeddings = {}
    timings = []
    for q in QUERIES:
        for _ in range(REPEATS):
            t = time.perf_counter()
            emb = model.encode(q, normalize_embeddings=True).tolist()
            timings.append(time.perf_counter() - t)
        embeddings[q] = emb
    return {"device": device, "load_s": load_s, "timings": timings, "embeddings": embeddings}


@app.function(gpu="T4")
def on_gpu() -> dict:
    return _measure("cuda")


@app.function(cpu=1.0, memory=4096)
def on_cpu1() -> dict:
    return _measure("cpu")


@app.function(cpu=2.0, memory=4096)
def on_cpu2() -> dict:
    return _measure("cpu")


@app.function(cpu=4.0, memory=4096)
def on_cpu4() -> dict:
    return _measure("cpu")


@app.local_entrypoint()
def main():
    import statistics

    calls = {
        "gpu": on_gpu.spawn(),
        "cpu1": on_cpu1.spawn(),
        "cpu2": on_cpu2.spawn(),
        "cpu4": on_cpu4.spawn(),
    }
    results = {k: c.get() for k, c in calls.items()}

    for k, r in results.items():
        ts = sorted(r["timings"])
        # The first encode warms kernels; report it separately from steady state.
        print(
            f"{k:5} load {r['load_s']:.1f}s  first {r['timings'][0]*1000:.0f}ms  "
            f"median {statistics.median(ts)*1000:.0f}ms  max {ts[-1]*1000:.0f}ms  "
            f"n={len(ts)}"
        )

    gpu = results["gpu"]["embeddings"]
    for k in ("cpu1", "cpu2", "cpu4"):
        cpu = results[k]["embeddings"]
        cos = [sum(a * b for a, b in zip(gpu[q], cpu[q])) for q in QUERIES]
        print(f"cosine(gpu, {k}) min {min(cos):.6f}  all {[round(c, 6) for c in cos]}")
