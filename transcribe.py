import modal
import os
import re

from corpus.chunking import build_chunks, build_chunks_from_text
from corpus.feed import entry_guid, episode_number_of
from corpus.store import episode_where, is_complete, paged_get, paged_get_ids

# Enrolment inputs: committed to the repo, mounted into the container. The
# five centroids are DERIVED from these at run time and never stored --
# `corpus/enrolment.py` refuses a stored copy because two hand-maintained
# copies drift asymmetrically and the stale one works right up until it
# matters.
ENROLMENT_LABELS_LOCAL = "speakers/labels.json"
ENROLMENT_VECTORS_LOCAL = (
    "speakers/clip_vectors_speaker-diarization-community-1.json"
)
ENROLMENT_LABELS = "/enrolment/labels.json"
ENROLMENT_VECTORS = "/enrolment/clip_vectors.json"

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
    # The enrolment inputs, so identity is decided at ingest instead of by a
    # later backfill. `copy=False` (the default) MOUNTS these at container
    # startup rather than baking an image layer, so adding them does not
    # change the image's content address and does not trigger the rebuild the
    # verbatim spec exists to avoid. Together ~2.2 MB; the 49 MB of `clips/`
    # audio is NOT needed, because the clip vectors are already computed.
    .add_local_file(ENROLMENT_LABELS_LOCAL, ENROLMENT_LABELS, copy=False)
    .add_local_file(ENROLMENT_VECTORS_LOCAL, ENROLMENT_VECTORS, copy=False)
    # add_local_python_source stays LAST.
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

        # Derived in corpus/feed.py so speaker_tool.py can resolve a
        # transcript filename back to its feed entry using the SAME rule.
        # Read its docstring before touching the fallback -- widening it is
        # actively wrong here, for a reason that is not obvious.
        episode_number = episode_number_of(entry)

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


def _load_enrolment(hf_token: str, show_name: str):
    """Build the enrolment centroids and the voice embedder, once.

    Extracted from `process_episodes` so `repair_identity` uses the SAME
    loader rather than a lookalike -- two copies of this drift, and the stale
    one keeps working right up until it matters.

    Returns `(enrolment, embedder, failures)`. Failures are returned rather
    than raised because identity is an enhancement: a model fault must never
    stop the corpus growing.
    """
    import json as _json

    from corpus.enrolment import build_centroids
    from corpus.voice import COMMUNITY, load_embedder

    failures: list[str] = []
    try:
        with open(ENROLMENT_LABELS) as handle:
            labels = _json.load(handle)
        with open(ENROLMENT_VECTORS) as handle:
            clip_vectors = _json.load(handle)
        enrolment = build_centroids(labels, clip_vectors)
        embedder = load_embedder(COMMUNITY, hf_token, model_home=MODEL_PATH)
        print(f"Enrolled voices: {sorted(enrolment)}")
    except Exception as e:
        # Named ONCE and loudly, rather than repeating silently for every
        # episode in the batch.
        print(f"ENROLMENT UNAVAILABLE: {type(e).__name__}: {e}")
        failures.append(
            f"{show_name}: speaker enrolment unavailable, this batch is "
            f"stored WITHOUT speaker_id: {type(e).__name__}: {e}"
        )
        return {}, None, failures
    else:
        # `else`, NOT a bare `if`. Both arms describe the same outcome -- no
        # centroids -- but they are different faults, and appending from both
        # reported one failure twice. `transcribe()` raises on any non-empty
        # failure list, so every extra entry dilutes it as a triage signal.
        #
        # Reaching here means the files loaded and parsed and STILL produced
        # nothing: no voice-verified label overlapped the clip vectors. That
        # is a keying fault, not a missing mount.
        if not enrolment:
            failures.append(
                f"{show_name}: enrolment inputs loaded but yielded zero "
                f"centroids -- no voice-verified label overlapped the clip "
                f"vectors. Episodes are stored WITHOUT speaker_id and will "
                f"not appear in speaker filters."
            )
    return enrolment, embedder, failures


def _download_audio(audio_url: str, dest: str) -> None:
    """Stream an episode's mp3 to `dest`. Extracted so there is one copy."""
    import requests

    print("Downloading audio...")
    response = requests.get(audio_url, stream=True)
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as handle:
        for block in response.iter_content(chunk_size=65536):
            handle.write(block)
            downloaded += len(block)
            if total:
                print(f"  Download progress: {downloaded / total * 100:.1f}%")
    print("Download complete.")


def _carry_identity_forward(
    collection, chunks: list, show: str, episode_number: str, date_str: str
) -> list[str]:
    """Re-apply this episode's stored cluster->person map to new chunks.

    Called before any re-chunking write. WITHOUT IT a `RULES_VERSION` bump
    would empty the corpus of names and queue 433 episodes of audio to
    recompute names that are already known -- 56 GB to re-derive a mapping
    the records in front of us already hold.

    The join is on the diarisation label, never on position, because the
    positional layout is precisely what a re-chunk moves.

    Returns notes for the caller's failure list. A refused or unmapped carry
    writes NO version, so the episode is classified IDENTIFY_ONLY on the next
    plan and re-earns its identity from audio rather than being certified with
    a hole in it.
    """
    from corpus.identify import carry_identity
    from corpus.identity import IDENTITY_VERSION

    prior = paged_get(
        collection, episode_where(show, episode_number, date_str), include=["metadatas"]
    )["metadatas"]
    stamped, refused, unmapped = carry_identity(
        prior, chunks, identity_version=IDENTITY_VERSION
    )

    notes: list[str] = []
    if refused:
        # NOT called corruption. A same-day unnumbered triple collision makes
        # two distinct episodes share episode_id_prefix and interleave, at
        # which point one label legitimately carries two people. Refusing is
        # right either way; naming the cause is not ours to do here.
        print(
            f"IDENTITY CONFLICT ep{episode_number}: {sorted(refused)} carry "
            f"conflicting names in stored records."
        )
        notes.append(
            f"{show} ep{episode_number} ({date_str}): cluster(s) "
            f"{sorted(refused)} carry conflicting names in stored records, so "
            f"no identity was carried forward. Episode left queued for repair."
        )
    if unmapped:
        print(
            f"IDENTITY NOT CARRIED ep{episode_number}: {sorted(unmapped)} have "
            f"no stored record to map from."
        )
        notes.append(
            f"{show} ep{episode_number} ({date_str}): cluster(s) "
            f"{sorted(unmapped)} are absent from the stored map, so no "
            f"identity was carried forward. Episode left queued for repair."
        )
    if not notes:
        print(f"  Carried identity onto {stamped}/{len(chunks)} chunks.")
    return notes


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
    from corpus.completeness import plan_episode
    from corpus.planning import IDENTITY_REPAIR_CAP, Action, cap_work
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
    # The cap bounds the RATE, per show and per run. There is no
    # cross-show accumulator: scheduled_job calls transcribe.remote once
    # per show behind its own try, so a budget threaded through a call
    # that raises on failure would simply be lost.
    to_identify, identity_left = cap_work(plan.to_identify, IDENTITY_REPAIR_CAP)

    print(
        f"{len(plan.to_transcribe)} to transcribe, "
        f"{len(plan.to_embed)} to re-embed."
    )
    # THE ABSOLUTE BACKLOG, printed every night. A count reported only as
    # a delta is indistinguishable, on night 6 of a drain, from a queue
    # stuck since night 2 -- unless somebody diffs yesterday's log. This
    # is the number a human compares across nights.
    print(
        f"identity backlog {len(plan.to_identify)} "
        f"(repairing {len(to_identify)} this run, {identity_left} left)"
    )

    failures = list(plan.failures)

    # Gated on the CAPPED lists, not on plan.has_work. With the cap at 0
    # the plan has work and nothing may act on it, and has_work would
    # start a GPU container to do nothing -- which is the whole cost the
    # plan/act split exists to avoid.
    if plan.to_transcribe or plan.to_embed:
        failures.extend(
            process_episodes.remote(
                show_name,
                plan.to_embed,
                plan.to_transcribe,
            )
        )
    elif not to_identify:
        print("Nothing to do -- no GPU container started.")

    if to_identify:
        # Chunk work settles before identity work: identity is derived
        # from chunks, and repairing over a chunk set about to be
        # rewritten would stamp positions rather than earn them.
        failures.extend(repair_identity.remote(show_name, to_identify))

        # RE-MEASURED, never subtracted. Arithmetic (337 - 25) would be
        # blind to precisely the failure a metadata-only update() invites:
        # a repair that reports success and writes nothing. Re-planning
        # only the episodes just repaired catches it the same night, for
        # a bounded number of round trips rather than a second full pass.
        volume.reload()
        verified = 0
        for episode in to_identify:
            try:
                action = plan_episode(
                    collection,
                    show=show_name,
                    episode_number=episode["episode_number"],
                    date_str=episode["date"],
                    transcript_text=read_transcript(episode),
                    episode_guid=episode.get("guid"),
                )
            except Exception as e:
                print(f"Could not verify ep{episode['episode_number']}: {e}")
                continue
            if action is not Action.IDENTIFY_ONLY:
                verified += 1
        print(f"repaired {len(to_identify)}, verified current {verified}")
        if verified < len(to_identify):
            failures.append(
                f"{show_name}: {len(to_identify) - verified} identity "
                f"repair(s) reported done but are still classified "
                f"IDENTIFY_ONLY -- the update wrote nothing."
            )

    print("\nAll done.")

    if failures:
        raise RuntimeError(
            f"{len(failures)} of "
            f"{len(plan.to_transcribe) + len(plan.to_embed) + len(to_identify)} "
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

    from corpus.enrolment import MARGIN, THRESHOLD
    from corpus.identify import (
        has_stored_identity,
        identify_clusters,
        stamp_chunks,
    )
    from corpus.identity import IDENTITY_VERSION
    from corpus.store import episode_where, paged_get

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
            # Same transcript, same diarisation labels -- only the
            # positional layout moves. Carry the names the records
            # already hold rather than losing them to a re-chunk.
            failures.extend(
                _carry_identity_forward(
                    collection, chunks, show_name, episode_number, date_str
                )
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

    # Identity is settled while the audio is on disk, which is what removes
    # the need for a backfill: the job already holds the diarised segments
    # and the file, so `speaker_id` is written WITH the chunk rather than
    # repaired months later. Built once per container, not per episode.
    enrolment: dict[str, list[float]] = {}
    embedder = None
    if to_transcribe:
        enrolment, embedder, load_failures = _load_enrolment(hf_token, show_name)
        failures.extend(load_failures)

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
            audio_path = f"/tmp/episode_{episode_number}.mp3"
            _download_audio(audio_url, audio_path)

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

            # Stamped BEFORE the write, so one upsert carries both the text
            # and the identity. A no-match writes nothing: an absent
            # `speaker_id` is the unidentified state, and a placeholder would
            # be a value meaning "no value" that the MCP filter would match.
            #
            # Its own try/except on purpose. The enclosing handler skips the
            # episode, and identity is an enhancement -- a model fault must
            # not stop the corpus growing. It is reported loudly because the
            # quiet version of this failure is invisible: Chroma's `$eq` does
            # not match a record missing the key, so an episode without
            # `speaker_id` silently drops out of every speaker-filtered
            # query rather than erroring.
            if enrolment and embedder is not None:
                try:
                    identified = identify_clusters(
                        result["segments"],
                        audio_path,
                        embedder,
                        enrolment,
                        threshold=THRESHOLD,
                        margin=MARGIN,
                        episode_name=f"{show_name} - Episode {episode_number}",
                    )
                    # PHASE-1 GUARD. This path has no delete before its
                    # write until phase 2 lands, and Chroma merges
                    # metadata onto POSITIONAL ids -- so an episode that
                    # already carries names keeps them on chunk i while
                    # this run's diarisation has renumbered the clusters.
                    # Writing identity_version over that would CERTIFY
                    # the stale names and retire the episode from
                    # IDENTIFY_ONLY permanently, and phase 2 ships no
                    # rescan. Withhold, report, and let the repair path
                    # refuse it again rather than half-fix it.
                    prior = paged_get(
                        collection,
                        episode_where(show_name, episode_number, date_str),
                        include=["metadatas"],
                    )["metadatas"]
                    guarded = has_stored_identity(prior)
                    stamped = stamp_chunks(
                        chunks,
                        identified,
                        identity_version=None if guarded else IDENTITY_VERSION,
                    )
                    if guarded:
                        print(
                            f"IDENTITY NOT CERTIFIED ep{episode_number}: "
                            f"{len(prior)} stored record(s) already carry a "
                            f"speaker_id, and this write cannot clear them. "
                            f"Names written; identity_version withheld so "
                            f"the episode stays queued for repair."
                        )
                        failures.append(
                            f"{show_name} ep{episode_number} ({date_str}): "
                            f"re-transcribed over existing speaker_id(s); "
                            f"identity_version withheld, episode left "
                            f"queued for IDENTIFY_ONLY."
                        )
                    print(
                        f"Identity: {len(identified)} cluster(s) named "
                        f"{sorted(set(identified.values()))}; "
                        f"{stamped}/{len(chunks)} chunks stamped."
                    )
                    missing = set(identified) - {
                        c["metadata"].get("speaker") for c in chunks
                    }
                    if missing:
                        # PARTIAL drift -- some labels joined, some did not.
                        # A WARNING and deliberately NOT a failure, because
                        # it has a real false positive: `build_chunks` emits
                        # a chunk only when a segment carries words, while
                        # `select_crop_turns` gates on a turn duration
                        # derived from segment GAPS. A music bed or a long
                        # silence can therefore be long enough to crop and
                        # identify while producing no words and so no chunk.
                        print(
                            f"WARNING ep{episode_number}: identified clusters "
                            f"with no matching chunk: {sorted(missing)}"
                        )

                    if identified and not stamped:
                        # Clusters were named and NOT ONE chunk took a
                        # value. Both come from the same segments, so a
                        # named cluster always owns chunks -- this can only
                        # mean the `speaker` metadata key drifted between
                        # `chunking.py` and `stamp_chunks`. Left unreported
                        # it prints a cheerful "3 cluster(s) named" every
                        # night, exits zero, and stores the whole corpus
                        # with no speaker_id.
                        #
                        # NOT a raise: identity failure must never lose an
                        # episode, and a join defect hits every episode
                        # equally, so raising would stop the corpus growing
                        # rather than merely stop it being filterable.
                        #
                        # The predicate is deliberately `stamped == 0` and
                        # not `stamped < expected`. Any "expected" here
                        # would be recomputed from `identified` and the same
                        # chunk key -- the same arithmetic stamp_chunks just
                        # did, which is a check that cannot fail.
                        print(
                            f"STAMP JOIN BROKEN ep{episode_number}: "
                            f"{len(identified)} clusters identified, 0 chunks "
                            f"stamped."
                        )
                        failures.append(
                            f"{show_name} ep{episode_number} ({date_str}): "
                            f"{len(identified)} clusters were identified but "
                            f"NO chunk received speaker_id -- the metadata key "
                            f"joining chunks to clusters has drifted. Episode "
                            f"stored without speaker_id."
                        )
                except Exception as e:
                    print(
                        f"IDENTITY FAILED ep{episode_number}: "
                        f"{type(e).__name__}: {e}"
                    )
                    failures.append(
                        f"{show_name} ep{episode_number} ({date_str}): speaker "
                        f"identity failed, episode stored WITHOUT speaker_id "
                        f"and will not appear in speaker filters: "
                        f"{type(e).__name__}: {e}"
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
def repair_identity(show_name: str, to_identify: list[dict]) -> list[str]:
    """Re-earn identity for episodes stored without it. Metadata only.

    Started only when `to_identify` is non-empty. It deliberately does NOT
    reuse `process_episodes`, which loads the BGE SentenceTransformer at
    container start before its `to_embed` loop: a pure identity repair needs
    neither that nor whisper.

    NEVER RE-DIARISES. The saved transcript's `SPEAKER_XX` labels are what the
    stored `speaker` metadata contains. Diarisation is not deterministic, so
    re-diarising would produce different labels and the join would be to a
    different world -- every record would look unmapped and the repair would
    either write nothing or write nonsense.

    Every write is a metadata-only `update()` scoped to one episode, so a
    mid-run kill is harmless: an episode that did not get its version stamped
    is simply still queued.

    Known limitation, named rather than guarded: `parse_all_episodes` only
    emits entries carrying an enclosure URL, so an episode aged off its feed
    never enters a plan and can never be queued for repair. That mirrors
    reconciliation's `feed_unreachable` class -- a fact about the feed, not a
    fault in the corpus -- and it bounds the poison-pill risk, since a
    permanently failing repair needs an episode still IN the feed whose
    download keeps failing.
    """
    from corpus.chunking import parse_transcript_segments
    from corpus.enrolment import MARGIN, THRESHOLD
    from corpus.identify import identify_clusters, plan_repair_updates
    from corpus.identity import IDENTITY_VERSION
    from corpus.store import batched

    failures: list[str] = []

    # A warm container carries the volume view it started with, and this
    # function reads transcripts written by an earlier one.
    volume.reload()

    collection = get_chroma_collection(
        os.environ["CHROMA_API_KEY"],
        os.environ["CHROMA_TENANT"],
        os.environ["CHROMA_DATABASE"],
    )

    enrolment, embedder, load_failures = _load_enrolment(
        os.environ["HF_TOKEN"], show_name
    )
    failures.extend(load_failures)
    if not enrolment or embedder is None:
        # Already reported once, loudly. Every episode stays queued.
        return failures

    print(f"{len(to_identify)} episode(s) to repair identity for.")

    for episode in to_identify:
        episode_number = episode["episode_number"]
        date_str = episode["date"]
        audio_path = f"/tmp/repair_{episode_number}.mp3"
        try:
            path = _transcript_path(show_name, episode)
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            segments = parse_transcript_segments(text)
            if not segments:
                raise RuntimeError(
                    "saved transcript parsed to zero segments; nothing to identify from"
                )

            _download_audio(episode["audio_url"], audio_path)

            identified = identify_clusters(
                segments,
                audio_path,
                embedder,
                enrolment,
                threshold=THRESHOLD,
                margin=MARGIN,
                episode_name=f"{show_name} - Episode {episode_number}",
            )

            stored = paged_get(
                collection,
                episode_where(show_name, episode_number, date_str),
                include=["metadatas"],
            )
            plan = plan_repair_updates(
                stored["ids"],
                stored["metadatas"],
                identified,
                identity_version=IDENTITY_VERSION,
            )

            if plan.refusal is not None:
                print(f"IDENTITY REPAIR REFUSED ep{episode_number}: {plan.refusal}")
                failures.append(
                    f"{show_name} ep{episode_number} ({date_str}) [identity]: "
                    f"{plan.refusal}"
                )
                continue

            for batch in batched(list(range(len(plan.ids)))):
                collection.update(
                    ids=[plan.ids[i] for i in batch],
                    metadatas=[plan.metadatas[i] for i in batch],
                )
            print(
                f"Repaired Episode {episode_number}: {len(identified)} "
                f"cluster(s) named {sorted(set(identified.values()))}; "
                f"{len(plan.ids)} record(s) updated."
            )
        except Exception as e:
            # Reported under its own heading with the episode triple, not
            # folded into undifferentiated failure text -- a repeatedly
            # failing episode has to be recognisable at a glance rather than
            # by reading identical prose every night.
            print(
                f"IDENTITY REPAIR FAILED ep{episode_number} ({date_str}): "
                f"{type(e).__name__}: {e}"
            )
            failures.append(
                f"{show_name} ep{episode_number} ({date_str}) [identity]: "
                f"{type(e).__name__}: {e}"
            )
            continue
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

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

            # Brought under the same rule as EMBED_ONLY: this re-chunks
            # the same transcript outside the planner, so without the
            # carry it would strip every name and leave records the
            # repair path's contradiction check then has to catch.
            failures.extend(
                _carry_identity_forward(
                    collection, chunks, parsed_show, episode_number, date_str
                )
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
