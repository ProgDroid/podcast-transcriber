import modal
import os
import re

from corpus.chunking import build_chunks, build_chunks_from_text
from corpus.feed import entry_guid
from corpus.store import episode_where, is_complete, paged_get_ids

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .apt_install("ffmpeg")
    .pip_install(
        "whisperx==3.8.5",
        "feedparser==6.0.12",
        "requests==2.33.1",
        "sentence-transformers==5.4.1",
        "chromadb==1.5.9",
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
    """Embed chunks and write them as a full replacement of the episode.

    Replacement is of the chunk SET, not of each record: upsert merges
    metadata, so a key on the old record that the new one omits survives.
    See corpus/writing.py's module docstring.
    """
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
    # mcp_server.py (the reader) now calls get_collection (fail if absent),
    # not get_or_create_collection -- it IS the "must not create" reader this
    # comment used to say a later task would produce. The load-bearing fact:
    # it reads CHROMA_COLLECTION from the same podcast-secrets secret as
    # resolve_collection_name above, so writer and reader can no longer
    # diverge -- a cutover moves both sides atomically via one secret edit,
    # not a code change on one side. Anyone relying on "the reader only
    # changes by code review" is mistaken. The reader has no allowlist of its
    # own (its brief scoped that out; see mcp_server.py::load), which is fine
    # here specifically because that shared trigger makes divergence, not an
    # unreviewed name, the failure this needs to guard against.
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _transcript_path(show_name: str, episode: dict) -> str:
    from corpus.identity import transcript_filename

    return (
        f"{VOLUME_PATH}/"
        f"{transcript_filename(show_name, episode['episode_number'], episode['date'])}"
    )


@app.function(
    timeout=7200,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("podcast-secrets")],
)
def transcribe(feed_url: str, show_name: str):
    """Plan a show on CPU, then hand only the real work to a GPU.

    THIS FUNCTION HAS NO GPU, deliberately. Planning costs two Chroma round
    trips per episode and re-chunks every transcript it finds -- roughly 870
    round trips and 433 re-chunks a night across the three shows -- and none
    of it needs an accelerator. Held inside the T4 container it originally
    ran in, that is minutes of GPU time burned nightly before any GPU work
    begins, and on most nights to conclude there is nothing to do at all.
    When the plan is empty no GPU container is started, which is the common
    case.

    The cost of the split is a plan-then-act gap: the corpus could in
    principle change between planning here and acting there. Accepted --
    nothing else writes overnight, and upsert_then_prune is idempotent, so a
    stale plan costs a redundant re-embed rather than a wrong corpus.
    """
    from corpus.showplan import plan_show

    # A warm CPU container carries the volume view it started with. Without
    # this, a container reused from an earlier run plans against a stale
    # filesystem, sees no transcript for an episode that was transcribed
    # since, and sends it back to the GPU to be transcribed AGAIN. The old
    # code could not hit this because the same container that read the volume
    # had also written it.
    volume.reload()

    collection = get_chroma_collection(
        os.environ["CHROMA_API_KEY"],
        os.environ["CHROMA_TENANT"],
        os.environ["CHROMA_DATABASE"],
    )

    episodes = parse_all_episodes(feed_url)
    print(f"Found {len(episodes)} episodes in feed.")

    def read_transcript(episode: dict) -> str | None:
        path = _transcript_path(show_name, episode)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    plan = plan_show(
        collection,
        show=show_name,
        episodes=episodes,
        read_transcript=read_transcript,
    )
    print(f"{len(plan.to_transcribe)} to transcribe, {len(plan.to_embed)} to re-embed.")

    failures = list(plan.failures)

    if plan.has_work:
        failures.extend(
            process_episodes.remote(
                show_name,
                plan.to_embed,
                plan.to_transcribe,
            )
        )
    else:
        print("Nothing to do -- no GPU container started.")

    print("\nAll done.")

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(plan.to_transcribe) + len(plan.to_embed)} "
            f"episodes failed:\n" + "\n".join(failures)
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
def process_episodes(
    show_name: str,
    to_embed: list[dict],
    to_transcribe: list[dict],
) -> list[str]:
    """Do the accelerator work a plan asked for. Returns the failure list.

    Called only when there is work, so reaching here always justifies the
    GPU. Returns failures rather than raising them: the caller owns the
    show's verdict and has planning failures of its own to combine with
    these, and an exception crossing .remote() would lose that list.
    """
    import gc

    import requests
    import whisperx
    from sentence_transformers import SentenceTransformer
    from whisperx.diarize import DiarizationPipeline

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

    failures: list[str] = []

    for episode in to_embed:
        episode_number = episode["episode_number"]
        date_str = episode["date"]
        try:
            with open(
                _transcript_path(show_name, episode), encoding="utf-8", errors="replace"
            ) as fh:
                text = fh.read()
            chunks = build_chunks_from_text(
                text,
                show_name,
                episode_number,
                episode["title"],
                date_str,
                episode_guid=episode.get("guid"),
            )
            print(f"Re-embedding Episode {episode_number} from transcript...")
            embed_and_store(
                chunks,
                embedding_model,
                collection,
                show_name,
                episode_number,
                date_str,
                episode.get("guid"),
            )
        except Exception as e:
            # One failing re-embed (including embed_and_store's own
            # RuntimeError on 0 chunks) must not skip the remaining
            # EMBED_ONLY episodes or the transcription phase below -- the
            # "abort the batch over one bad item" bug, one level down.
            print(
                f"Failed to re-embed Episode {episode_number} "
                f"({date_str}): {type(e).__name__}: {e}"
            )
            failures.append(
                f"{show_name} ep{episode_number} ({date_str}) [embed_only]: {e}"
            )
            continue

    if not to_transcribe:
        # Guarded because loading whisperx costs a large-v2 read and a VAD
        # startup, and an embed-only run used to pay both for nothing.
        print("No episodes to transcribe.")
        return failures

    print(f"{len(to_transcribe)} episodes to transcribe.")
    print("Loading whisperx model...")
    model = whisperx.load_model(
        "large-v2",
        device,
        compute_type=compute_type,
        download_root=MODEL_PATH,
    )

    for episode in to_transcribe:
        episode_number = episode["episode_number"]
        audio_url = episode["audio_url"]
        episode_title = episode["title"]
        date_str = episode["date"]
        out_path = _transcript_path(show_name, episode)

        print(
            f"\nProcessing: {show_name} - Episode {episode_number} - "
            f"{episode_title} ({date_str})"
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

    return failures


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

            # Root cause 1's third mechanism, verbatim: a bare presence
            # check "matches the first member of a colliding group and skips
            # the rest." A torn episode with one surviving chunk also read
            # as healthy. is_complete compares the full stored id count
            # against this episode's own expected chunk count instead.
            stored_ids = paged_get_ids(
                collection, episode_where(parsed_show, episode_number, date_str)
            )
            if is_complete(stored_ids, len(chunks)):
                print(f"Skipping Episode {episode_number} — already complete.")
                continue

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
