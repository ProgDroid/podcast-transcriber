"""Run the reconciliation against production. Read-only."""

from __future__ import annotations

import os
import re

import modal

from corpus.identity import parse_transcript_filename
from corpus.reconcile import reconcile
from corpus.store import PAGE

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb==1.5.9", "feedparser==6.0.12")
    .add_local_python_source("corpus")
)

app = modal.App("podcast-reconcile", image=image)
volume = modal.Volume.from_name("podcast-transcripts")
VOLUME_PATH = "/transcripts"

FEEDS = {
    "Geopolitical Cousins": "https://feeds.captivate.fm/geopolitical-cousins/",
    "The Jacob Shapiro Podcast": "https://feeds.captivate.fm/jacob-shapiro/",
    "The Observing Japan Podcast": (
        "https://api.substack.com/feed/podcast/868206/s/386602.rss"
    ),
}


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=1800,
)
def run() -> None:
    import email.utils

    import chromadb
    import feedparser

    volume_keys = set()
    for name in os.listdir(VOLUME_PATH):
        key = parse_transcript_filename(name)
        if key:
            volume_keys.add(key)

    feed_keys = set()
    for show, url in FEEDS.items():
        for entry in feedparser.parse(url).entries:
            number = entry.get("itunes_episode")
            if number is None:
                m = re.search(r"\b[Ee]p\.?\s*(\d+)", entry.get("title", ""))
                number = m.group(1) if m else "Unknown"
            try:
                date_str = email.utils.parsedate_to_datetime(
                    entry.get("published")
                ).strftime("%Y-%m-%d")
            except Exception:
                continue
            feed_keys.add((show, str(number), date_str))

    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    col = client.get_collection(
        os.environ.get("CHROMA_COLLECTION", "podcast_transcripts")
    )

    # Never call get() unlimited -- it silently truncates at 300 on Cloud.
    # reconcile() itself filters out non-episode (book) records via
    # corpus.remap.is_non_episode, so paging in everything is safe here.
    records = []
    offset = 0
    while True:
        page = col.get(limit=PAGE, offset=offset, include=["metadatas"])
        if not page["ids"]:
            break
        records.extend(zip(page["ids"], page["metadatas"], strict=True))
        offset += len(page["ids"])

    print(f"volume={len(volume_keys)} feed={len(feed_keys)} records={len(records)}")
    report = reconcile(volume_keys, records, feed_keys)
    for name in (
        "missing",
        "extra",
        "non_contiguous",
        "shared_prefixes",
        "excluded_with_records",
        "feed_unreachable",
    ):
        values = getattr(report, name)
        print(f"\n{name.upper()} ({len(values)}):")
        for v in values:
            print("   ", v)
    # An empty list here would read as "no cross-posts found" -- a confident
    # negative. It is not computed at all: reconcile() never receives episode
    # titles, and cross-post detection needs a fuzzy title match. Render that
    # explicitly so nobody mistakes silence for a clean result.
    print("\nSUSPECTED_CROSS_POSTS: not computed (no title data reaches reconcile())")
    print(f"\nCLEAN: {report.is_clean()}")


@app.local_entrypoint()
def main() -> None:
    run.remote()
