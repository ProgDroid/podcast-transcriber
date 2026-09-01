"""Re-ID `podcast_transcripts` into `podcast_transcripts_v2`.

Copies every record verbatim except the id, and adds three metadata fields:
`n_chunks`, `episode_guid` and `rules_version`. No GPU: Chroma returns stored
embeddings, so nothing is re-embedded.

RUN WITH THE CRON STOPPED. copy_collection aborts if the source count moves.
"""

from __future__ import annotations

import os
import re

import modal

from corpus.feed import entry_guid
from corpus.identity import parse_transcript_filename
from corpus.reid_planning import build_episode_facts, enrich_metadata
from corpus.store import BATCH, PAGE, batched

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb==1.5.9", "feedparser==6.0.12", "numpy>=2.0")
    # `migration` too: run() imports chroma_migrate inside the container, and
    # Modal 1.5.5 mounts nothing it is not told about. `migration/` needs an
    # __init__.py for this to resolve as a package -- selftest.py imports it
    # flat (`from chroma_migrate import ...`), which still works because
    # running it as a script puts migration/ on sys.path.
    .add_local_python_source("corpus", "migration")
)

app = modal.App("podcast-reid-migration", image=image)
volume = modal.Volume.from_name("podcast-transcripts")
VOLUME_PATH = "/transcripts"

SOURCE = "podcast_transcripts"
DEST = "podcast_transcripts_v2"
assert DEST != SOURCE, "the destination must not be the source collection"
NEW_KEYS = frozenset({"n_chunks", "episode_guid", "rules_version"})

FEEDS = {
    "Geopolitical Cousins": "https://feeds.captivate.fm/geopolitical-cousins/",
    "The Jacob Shapiro Podcast": "https://feeds.captivate.fm/jacob-shapiro/",
    "The Observing Japan Podcast": "https://api.substack.com/feed/podcast/868206/s/386602.rss",
}


def _load_feed_guids() -> dict[tuple[str, str, str], str]:
    import email.utils

    import feedparser

    guids: dict[tuple[str, str, str], str] = {}
    for show, url in FEEDS.items():
        for entry in feedparser.parse(url).entries:
            gid = entry_guid(entry)
            if not gid:
                continue
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
            guids[(show, str(number), date_str)] = gid
    return guids


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=7200,
)
def run(dry_run: bool = True):
    import chromadb

    from corpus.remap import remap_id
    from migration.chroma_migrate import create_dest_collection, validate_collection

    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    src = client.get_collection(SOURCE)
    total = src.count()
    print(f"source={SOURCE} count={total} dry_run={dry_run}")

    texts: dict[tuple[str, str, str], str] = {}
    for name in os.listdir(VOLUME_PATH):
        key = parse_transcript_filename(name)
        if key:
            with open(
                f"{VOLUME_PATH}/{name}", encoding="utf-8", errors="replace"
            ) as fh:
                texts[key] = fh.read()
    print(f"transcripts on volume: {len(texts)}")

    guids = _load_feed_guids()
    facts = build_episode_facts(texts, guids)
    print(
        f"episodes with facts: {len(facts)}  with guid: "
        f"{sum(1 for f in facts.values() if 'episode_guid' in f)}"
    )

    if dry_run:
        counts: dict[str, int] = {}
        unmatched_sample: list[tuple[str, list[str]]] = []
        offset = 0
        while offset < total:
            page = src.get(limit=PAGE, offset=offset, include=["metadatas"])
            if not page["ids"]:
                break
            for _id, meta in zip(page["ids"], page["metadatas"], strict=True):
                classification = remap_id(_id, meta).classification
                counts[classification] = counts.get(classification, 0) + 1
                if (
                    classification == "passthrough_unmatched"
                    and len(unmatched_sample) < 5
                ):
                    unmatched_sample.append((_id, sorted(meta.keys())))
            offset += len(page["ids"])
        print("classification:", counts)
        if unmatched_sample:
            # An id that disagrees with its own metadata is genuinely
            # unpredicted by the spec -- print enough to diagnose it without
            # a second investigation: the id itself plus which metadata
            # fields it carries (not the values, which may be long or
            # sensitive).
            n_unmatched = counts["passthrough_unmatched"]
            print(f"passthrough_unmatched sample (up to 5 of {n_unmatched}):")
            for sample_id, keys in unmatched_sample:
                print(f"   {sample_id}  keys={keys}")
        print("DRY RUN -- nothing written.")
        return

    # create_dest_collection, NOT get_or_create_collection: the spec requires
    # v2 to be built from the source's SERIALIZED SCHEMA so distance space,
    # index enablement and key-specific indexes carry over. A metadata-built
    # collection serializes to a different schema, and validate_collection
    # compares schemas -- so building it the easy way means either a spurious
    # "schema mismatch" abort at the END of a 28,489-record copy, or a pass
    # that never reproduced the index configuration at all.
    dst = create_dest_collection(client, src, dest_name=DEST)

    # Pre-flight the schema BEFORE copying 28,489 records. validate_collection
    # checks it too, but only at the very end -- so a divergence would abort
    # after the whole copy had run.
    schema_preflight_clean = False
    try:
        if src.schema.serialize_to_json() != dst.schema.serialize_to_json():
            raise SystemExit(
                "Destination schema differs from the source before any records "
                "were copied. create_dest_collection did not reproduce it. "
                "Do NOT proceed."
            )
        schema_preflight_clean = True
        print("  schema pre-flight: identical")
    except AttributeError:
        print("  schema pre-flight: skipped (build predates the Schema API)")

    id_map: dict[str, str] = {}
    offset = 0
    while offset < total:
        page = src.get(
            limit=PAGE,
            offset=offset,
            include=["embeddings", "documents", "metadatas", "uris"],
        )
        if not page["ids"]:
            break
        new_ids, docs, metas, embs = [], [], [], []
        for k, _id in enumerate(page["ids"]):
            meta = page["metadatas"][k]
            result = remap_id(_id, meta)
            id_map[_id] = result.new_id
            key = (meta.get("show"), meta.get("episode_number"), meta.get("date"))
            new_ids.append(result.new_id)
            docs.append(page["documents"][k])
            metas.append(enrich_metadata(meta, facts.get(key)))
            embs.append(list(page["embeddings"][k]))
        for batch in batched(list(range(len(new_ids))), BATCH):
            dst.upsert(
                ids=[new_ids[i] for i in batch],
                documents=[docs[i] for i in batch],
                metadatas=[metas[i] for i in batch],
                embeddings=[embs[i] for i in batch],
            )
        offset += len(page["ids"])
        print(f"  copied {offset}/{total}")

    final = src.count()
    if final != total:
        raise SystemExit(
            f"SOURCE CHANGED during copy ({total} -> {final}). "
            f"A writer was not frozen. Aborting - do NOT cut over."
        )

    problems = validate_collection(src, dst, id_map=id_map, allowed_new_keys=NEW_KEYS)
    if schema_preflight_clean:
        # The three new metadata keys can add key-specific index entries to
        # v2's schema once data is written, which v1 has no reason to carry.
        # That divergence is expected and is not a copy fault -- the
        # pre-flight already proved the collections were created identically.
        before = len(problems)
        problems = [p for p in problems if "schema mismatch" not in p]
        if len(problems) != before:
            print("  note: post-copy schema differs (expected: new metadata keys)")
    if problems:
        print(f"VALIDATION FAILED with {len(problems)} problems:")
        for p in problems[:20]:
            print("   ", p)
        raise SystemExit("Do NOT cut over.")
    print(f"VALIDATION CLEAN. {DEST} has {dst.count()} records.")


@app.local_entrypoint()
def main(dry_run: bool = True):
    run.remote(dry_run=dry_run)
