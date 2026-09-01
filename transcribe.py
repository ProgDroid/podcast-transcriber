import modal
import os
import re

from corpus.chunking import build_chunks, build_chunks_from_text
from corpus.feed import entry_guid

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .pip_install(
        "torch", "torchaudio", extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    .apt_install("ffmpeg")
    .pip_install(
        "whisperx",
        "feedparser",
        "requests",
        "sentence-transformers",
        "chromadb",
    )
    .add_local_python_source("corpus")
)

app = modal.App("podcast-transcriber", image=image)

volume = modal.Volume.from_name("podcast-transcripts", create_if_missing=True)
model_cache = modal.Volume.from_name("whisperx-model-cache", create_if_missing=True)

VOLUME_PATH = "/transcripts"
MODEL_PATH = "/models"


def parse_all_episodes(feed_url: str):
    import feedparser
    import email.utils

    feed = feedparser.parse(feed_url)
    episodes = []

    for entry in feed.entries:
        title = entry.get("title", "Unknown Title")

        episode_number = entry.get("itunes_episode", None)
        if episode_number is None:
            match = re.search(r"\b[Ee]p\.?\s*(\d+)", title)
            episode_number = match.group(1) if match else "Unknown"

        published = entry.get("published", None)
        if published:
            try:
                dt = email.utils.parsedate_to_datetime(published)
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = "Unknown Date"
        else:
            date_str = "Unknown Date"

        audio_url = None
        for link in entry.get("links", []):
            if link.get("rel") == "enclosure":
                audio_url = link["href"]
                break
        if not audio_url and entry.get("enclosures"):
            audio_url = entry["enclosures"][0]["href"]

        if audio_url:
            episodes.append(
                {
                    "title": title,
                    "episode_number": str(episode_number),
                    "audio_url": audio_url,
                    "date": date_str,
                    "guid": entry_guid(entry),
                }
            )

    return episodes


def embed_and_store(
    chunks: list,
    embedding_model,
    chroma_collection,
    show: str,
    episode_number: str,
    date_str: str,
    episode_guid: str | None = None,
):
    """Embed chunks and write them as a full replacement of the episode."""
    from corpus.writing import upsert_then_prune

    if not chunks:
        # All 438 transcripts were verified to parse to at least one chunk,
        # so zero chunks means something is wrong with the transcript, not
        # with the episode. Leaving the old records in place is the safe
        # direction (a superset, never a hole) -- do NOT prune against an
        # empty set -- but this must reach the caller's failures list rather
        # than pass silently, or "produced nothing" and "not re-embedded"
        # become indistinguishable in the corpus.
        raise RuntimeError(
            "0 chunks parsed; not stored, existing records left in place"
        )

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    result = upsert_then_prune(
        chroma_collection,
        chunks,
        embeddings,
        show=show,
        episode_number=episode_number,
        date_str=date_str,
        episode_guid=episode_guid,
    )
    print(f"  Stored {result['written']} chunks, pruned {result['pruned']}.")


def get_chroma_collection(chroma_api_key, chroma_tenant, chroma_database):
    import chromadb

    from corpus.store import resolve_collection_name

    # Resolved and validated BEFORE the client is constructed: a bad secret
    # should not cost a client construction and an auth round-trip on its way
    # to failing. resolve_collection_name refuses an unreviewed name rather
    # than letting get_or_create_collection silently provision a phantom
    # collection -- both transcribe and bulk_embed mount the same secret, so
    # one unreviewed key in the Modal dashboard would otherwise repoint every
    # nightly write with no code diff and no error at either end.
    name = resolve_collection_name(
        os.environ.get("CHROMA_COLLECTION", "podcast_transcripts")
    )
    print(f"Writing to collection: {name}")

    client = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database,
    )
    # mcp_server.py (the reader) also calls get_or_create_collection today,
    # against a hardcoded name with no allowlist of its own -- it is not yet
    # the "must not create" reader this comment used to claim. A later task
    # switches it to get_collection (fail if absent) as part of the reviewed
    # cutover; until then, scheduled_job spawning fresh containers nightly
    # just means this side has no warm-container exposure to worry about.
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


@app.function(
    gpu="T4",
    timeout=7200,
    volumes={
        VOLUME_PATH: volume,
        MODEL_PATH: model_cache,
    },
    secrets=[modal.Secret.from_name("podcast-secrets")],
)
def transcribe(feed_url: str, show_name: str):
    import whisperx
    import requests
    import gc
    from whisperx.diarize import DiarizationPipeline
    from sentence_transformers import SentenceTransformer

    hf_token = os.environ["HF_TOKEN"]
    device = "cuda"
    compute_type = "float16"

    collection = get_chroma_collection(
        os.environ["CHROMA_API_KEY"],
        os.environ["CHROMA_TENANT"],
        os.environ["CHROMA_DATABASE"],
    )

    print("Loading BGE embedding model...")
    embedding_model = SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        device=device,
        cache_folder=MODEL_PATH,
    )

    episodes = parse_all_episodes(feed_url)
    print(f"Found {len(episodes)} episodes in feed.")

    pending = []
    for episode in episodes:
        out_path = f"{VOLUME_PATH}/{show_name} - Episode {episode['episode_number']} - {episode['date']}.txt"
        if os.path.exists(out_path):
            print(
                f"Skipping Episode {episode['episode_number']} — already transcribed."
            )
        else:
            pending.append(episode)

    if not pending:
        print("All episodes already transcribed.")
        return

    print(f"{len(pending)} episodes to transcribe.")
    print("Loading whisperx model...")
    model = whisperx.load_model(
        "large-v2",
        device,
        compute_type=compute_type,
        download_root=MODEL_PATH,
    )

    failures: list[str] = []

    for episode in pending:
        episode_number = episode["episode_number"]
        audio_url = episode["audio_url"]
        episode_title = episode["title"]
        date_str = episode["date"]
        out_path = (
            f"{VOLUME_PATH}/{show_name} - Episode {episode_number} - {date_str}.txt"
        )

        print(
            f"\nProcessing: {show_name} - Episode {episode_number} - {episode_title} ({date_str})"
        )

        try:
            print("Downloading audio...")
            audio_path = f"/tmp/episode_{episode_number}.mp3"
            response = requests.get(audio_url, stream=True)
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(audio_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"  Download progress: {downloaded / total * 100:.1f}%")
            print("Download complete.")

            print("Transcribing...")
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=16)
            print(f"Transcription complete. Detected language: {result['language']}")

            print("Aligning...")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=device,
                model_dir=MODEL_PATH,
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            print("Alignment complete.")
            del model_a
            gc.collect()

            print("Diarising...")
            diarize_model = DiarizationPipeline(token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print("Diarisation complete.")

            lines = [
                f"# {show_name} - Episode {episode_number}",
                f"# {episode_title}",
                f"# Published: {date_str}",
                "",
            ]
            for segment in result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment["text"].strip()
                start = segment["start"]
                lines.append(f"[{speaker}] {start:.1f}s - {text}")

            with open(out_path, "w") as f:
                f.write("\n".join(lines))
            volume.commit()
            print(f"Transcript saved: {out_path}")

            print("Chunking and embedding...")
            chunks = build_chunks(
                result["segments"],
                show_name,
                episode_number,
                episode_title,
                date_str,
                episode_guid=episode.get("guid"),
            )
            embed_and_store(
                chunks,
                embedding_model,
                collection,
                show_name,
                episode_number,
                date_str,
                episode.get("guid"),
            )

            os.remove(audio_path)

        except Exception as e:
            print(f"Failed Episode {episode_number}: {type(e).__name__}: {e}")
            failures.append(f"{show_name} ep{episode_number} ({date_str}): {e}")
            continue

    del model
    gc.collect()
    print("\nAll done.")

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(pending)} episodes failed:\n"
            + "\n".join(failures)
        )


@app.function(
    gpu="T4",
    timeout=7200,
    volumes={
        VOLUME_PATH: volume,
        MODEL_PATH: model_cache,
    },
    secrets=[modal.Secret.from_name("podcast-secrets")],
)
def bulk_embed(show_name: str):
    """
    Embed and upload all already-transcribed episodes for a given show
    that are not yet in ChromaDB. Safe to re-run — upserts are idempotent.
    """
    from sentence_transformers import SentenceTransformer

    device = "cuda"

    collection = get_chroma_collection(
        os.environ["CHROMA_API_KEY"],
        os.environ["CHROMA_TENANT"],
        os.environ["CHROMA_DATABASE"],
    )

    print("Loading BGE embedding model...")
    embedding_model = SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        device=device,
        cache_folder=MODEL_PATH,
    )

    # Find all transcript files for this show in the volume
    all_files = [
        e.path
        for e in volume.listdir("/")
        if e.path.startswith(show_name) and e.path.endswith(".txt")
    ]

    if not all_files:
        print(f"No transcripts found for '{show_name}'.")
        return

    print(f"Found {len(all_files)} transcripts for '{show_name}'.")

    failures: list[str] = []

    for filename in sorted(all_files):
        # Parse show name, episode number and date from filename
        # Format: {Show Name} - Episode {N} - {YYYY-MM-DD}.txt
        m = re.match(r"^(.+) - Episode (\w+) - (\d{4}-\d{2}-\d{2})\.txt$", filename)
        if not m:
            print(f"Skipping unrecognised filename format: {filename}")
            continue

        parsed_show = m.group(1)
        episode_number = m.group(2)
        date_str = m.group(3)

        # Check if already in ChromaDB by looking for any chunk with this episode ID prefix
        existing = collection.get(
            where={
                "$and": [
                    {"show": {"$eq": parsed_show}},
                    {"episode_number": {"$eq": episode_number}},
                ]
            },
            limit=1,
        )
        if existing["ids"]:
            print(f"Skipping Episode {episode_number} — already in ChromaDB.")
            continue

        print(f"\nEmbedding: {filename}")

        try:
            file_path = f"{VOLUME_PATH}/{filename}"
            with open(file_path, "r") as f:
                text = f.read()

            # Extract episode title from header comment
            episode_title = "Unknown Title"
            for line in text.splitlines():
                if (
                    line.startswith("# ")
                    and "Episode" not in line
                    and "Published" not in line
                ):
                    episode_title = line.lstrip("# ").strip()
                    break

            # bulk_embed has no feed entry in hand -- pass episode_guid=None
            # explicitly rather than appearing to have one. Safe because
            # upsert MERGES metadata: a guid already stamped by the
            # migration survives a re-embed that omits it here.
            chunks = build_chunks_from_text(
                text,
                parsed_show,
                episode_number,
                episode_title,
                date_str,
                episode_guid=None,
            )

            embed_and_store(
                chunks,
                embedding_model,
                collection,
                parsed_show,
                episode_number,
                date_str,
                episode_guid=None,
            )

        except Exception as e:
            print(f"Failed {filename}: {type(e).__name__}: {e}")
            failures.append(f"{filename}: {e}")
            continue

    print("\nBulk embed complete.")

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(all_files)} transcripts failed:\n"
            + "\n".join(failures)
        )


@app.function(
    schedule=modal.Cron("0 9 * * *"),
    timeout=7200,
)
def scheduled_job():
    # `transcribe` now raises when any of its episodes failed, so these three
    # calls must be isolated from each other. Left as bare sequential calls,
    # one show's bad episode would abort the two shows after it -- exactly the
    # "abort the batch over one bad item" behaviour the per-episode failure
    # list exists to avoid, reintroduced one level up. Run all three, then
    # fail if any did, so the run is still loudly unsuccessful.
    shows = [
        (
            "https://feeds.captivate.fm/geopolitical-cousins/",
            "Geopolitical Cousins",
        ),
        (
            "https://feeds.captivate.fm/jacob-shapiro/",
            "The Jacob Shapiro Podcast",
        ),
        (
            "https://api.substack.com/feed/podcast/868206/s/386602.rss",
            "The Observing Japan Podcast",
        ),
    ]
    failures: list[str] = []
    for feed_url, show_name in shows:
        try:
            transcribe.remote(feed_url=feed_url, show_name=show_name)
        except Exception as e:
            print(f"Show failed: {show_name}: {type(e).__name__}: {e}")
            failures.append(f"{show_name}: {e}")

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(shows)} shows failed:\n" + "\n".join(failures)
        )


@app.local_entrypoint()
def main(
    feed_url: str = "https://feeds.captivate.fm/geopolitical-cousins/",
    show_name: str = "Geopolitical Cousins",
):
    call = transcribe.spawn(feed_url=feed_url, show_name=show_name)
    print(f"Job started. Track it at https://modal.com/apps")
    print(f"Call ID: {call.object_id}")


@app.local_entrypoint()
def bulk_upload(show_name: str = "Geopolitical Cousins"):
    """
    Embed and upload all already-transcribed episodes for a show.
    Run once per show to backfill ChromaDB.

    Usage:
        modal run transcribe.py::bulk_upload --show-name "Geopolitical Cousins"
        modal run transcribe.py::bulk_upload --show-name "The Jacob Shapiro Podcast"
    """
    call = bulk_embed.spawn(show_name=show_name)
    print(f"Bulk upload started for '{show_name}'.")
    print(f"Track it at https://modal.com/apps")
    print(f"Call ID: {call.object_id}")
