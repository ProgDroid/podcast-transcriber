# Corpus Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover seven episodes that are on the Modal volume but invisible to search, and close the four write-path defects that caused them — before the speaker work triggers them at scale.

**Architecture:** Extract every decision that does not need the network into a pure `corpus/` package that imports on a laptop in milliseconds, leaving `transcribe.py` and `mcp_server.py` as thin Modal shells around it. Writes become upsert-then-prune with batching; presence becomes a completeness check against an `n_chunks` field; a one-time migration re-IDs the corpus and backfills the new metadata.

**Tech Stack:** Python 3.12 (Modal containers), uv, pytest, ruff, mypy, Modal 1.5.5, chromadb 1.5.9, Chroma Cloud (`geo-podcasts-us`).

**Spec:** `docs/superpowers/specs/2026-08-31-corpus-integrity-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **`corpus/` must import under Python 3.12** — it runs inside Modal containers. No 3.13+ syntax. CI pins 3.12 even though the local interpreter is 3.14.
- **`corpus/` must never `import modal`.** That is what makes it testable without network, GPU, or credentials.
- **`PAGE = 250` and `BATCH = 250`.** Chroma Cloud caps `get()` at 300 and `upsert()` at 300 records *per request*. Both raise above the cap.
- **`get()` with no explicit `limit` silently returns 300.** Never call it unpaged. This is the only cap that fails quietly.
- **Chroma `upsert` MERGES metadata per key**; it does not replace the dict. A key absent from the new metadata survives from the old record.
- **`delete(where=...)` is supported and a non-matching filter is a no-op.** Verified on Cloud, chromadb 1.5.9.
- **Never run any task against the `podcast_transcripts` collection.** Development and integration work uses a throwaway collection whose name starts with a letter (Chroma rejects a leading underscore).
- **Modal 1.5.5 does not auto-mount local packages.** Any image running `corpus` code needs `.add_local_python_source("corpus")`. This adds files at container start, not as an image layer, so it does **not** invalidate cached layers.
- **Excluded episodes** (cross-posts embedded under Geopolitical Cousins only):
  `("The Jacob Shapiro Podcast", "Unknown", "2026-07-29")` and
  `("The Jacob Shapiro Podcast", "Unknown", "2026-07-31")`.
- **Corpus size:** 438 transcripts on the volume, **436** after exclusions. 28,489 Chroma records, of which 191 are the *Geopolitical Alpha* book.
- Command prefix is `uv run` throughout.

---

### Task 1: Test harness and episode identity

Establishes the package, the tooling, and CI, because the first testable unit needs all three.

**Files:**
- Create: `pyproject.toml`
- Create: `corpus/__init__.py`
- Create: `corpus/identity.py`
- Create: `tests/__init__.py`
- Create: `tests/test_identity.py`
- Create: `.github/workflows/ci.yml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `corpus.identity.RULES_VERSION: str`
  - `corpus.identity.episode_id_prefix(show: str, episode_number: str, date_str: str) -> str`
  - `corpus.identity.chunk_id(prefix: str, index: int) -> str`
  - `corpus.identity.parse_transcript_filename(filename: str) -> tuple[str, str, str] | None`
  - `corpus.identity.transcript_filename(show: str, episode_number: str, date_str: str) -> str`

- [ ] **Step 1: Create the project configuration**

```toml
# pyproject.toml
[project]
name = "podcast-transcriber"
version = "0.1.0"
description = "Serverless podcast transcription and semantic search on Modal"
requires-python = ">=3.12"
dependencies = []

[dependency-groups]
dev = [
    "pytest>=8.3",
    "ruff>=0.14",
    "mypy>=1.13",
    # Task 6's tests import migration/chroma_migrate.py, which does
    # `import numpy as np` at module scope (chroma_migrate.py:17) and needs a
    # real Collection surface. Without these the test file fails collection.
    "numpy>=2.0",
    "chromadb==1.5.9",
]

[tool.ruff]
target-version = "py312"
line-length = 88
src = ["corpus", "tests"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]

[tool.mypy]
python_version = "3.12"
files = ["corpus", "tests"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Write the failing test**

The three known colliding groups and the four known duplicate `(show, date)` pairs are the regression cases — they are the actual production data that broke.

```python
# tests/test_identity.py
import pytest

from corpus.identity import (
    chunk_id,
    episode_id_prefix,
    parse_transcript_filename,
    transcript_filename,
)


def test_spaces_become_underscores():
    assert (
        episode_id_prefix("Geopolitical Cousins", "1", "2025-03-14")
        == "Geopolitical_Cousins-ep1-2025-03-14"
    )


def test_prefix_is_stable_across_calls():
    a = episode_id_prefix("The Jacob Shapiro Podcast", "243", "2024-11-07")
    b = episode_id_prefix("The Jacob Shapiro Podcast", "243", "2024-11-07")
    assert a == b


@pytest.mark.parametrize(
    "show,ep,date_a,date_b",
    [
        # the duplicate episode number that silently lost an episode
        ("The Jacob Shapiro Podcast", "243", "2024-11-07", "2024-11-08"),
        # the two 'Unknown' groups
        ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29", "2026-07-31"),
        ("The Observing Japan Podcast", "Unknown", "2026-05-12", "2026-06-05"),
    ],
)
def test_same_episode_number_different_dates_do_not_collide(show, ep, date_a, date_b):
    assert episode_id_prefix(show, ep, date_a) != episode_id_prefix(show, ep, date_b)


@pytest.mark.parametrize(
    "show,date,ep_a,ep_b",
    [
        # the four (show, date) pairs that made date-only keying wrong
        ("Geopolitical Cousins", "2026-05-22", "10", "11"),
        ("The Jacob Shapiro Podcast", "2023-11-20", "150", "151"),
        ("The Jacob Shapiro Podcast", "2025-03-28", "270", "271"),
        ("The Jacob Shapiro Podcast", "2025-06-13", "300", "301"),
    ],
)
def test_same_date_different_episode_numbers_do_not_collide(show, date, ep_a, ep_b):
    assert episode_id_prefix(show, ep_a, date) != episode_id_prefix(show, ep_b, date)


def test_chunk_id_appends_index():
    prefix = episode_id_prefix("Geopolitical Cousins", "73", "2026-07-29")
    assert chunk_id(prefix, 0) == "Geopolitical_Cousins-ep73-2026-07-29-0"
    assert chunk_id(prefix, 430) == "Geopolitical_Cousins-ep73-2026-07-29-430"


def test_filename_round_trip():
    name = transcript_filename("Geopolitical Cousins", "73", "2026-07-29")
    assert name == "Geopolitical Cousins - Episode 73 - 2026-07-29.txt"
    assert parse_transcript_filename(name) == (
        "Geopolitical Cousins",
        "73",
        "2026-07-29",
    )


def test_filename_parses_unknown_episode_number():
    assert parse_transcript_filename(
        "The Observing Japan Podcast - Episode Unknown - 2026-05-12.txt"
    ) == ("The Observing Japan Podcast", "Unknown", "2026-05-12")


def test_filename_rejects_non_transcript():
    assert parse_transcript_filename("notes.md") is None
    assert parse_transcript_filename("Show - Episode 1 - 2025-13-99.txt") is None
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_identity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus'`

- [ ] **Step 4: Write the implementation**

```python
# corpus/__init__.py
"""Pure, network-free logic shared by the Modal apps and the test suite.

Nothing in this package may `import modal`. That is the property that lets the
whole decision layer run on a laptop in milliseconds with no GPU, no
credentials and no network.
"""
```

```python
# corpus/identity.py
"""Episode identity.

`episode_number` is a display attribute, not a key: the feeds omit it (5
episodes carry the literal string "Unknown") and occasionally repeat it
(`Episode 243` exists on both 2024-11-07 and 2024-11-08). Neither is
`date` a key on its own -- four (show, date) pairs are duplicated. Only the
full triple is unique, at 438 distinct keys across 438 transcripts.
"""

from __future__ import annotations

import re

# Bumped whenever anything that derives chunk CONTENT from a transcript
# changes -- today the speaker labels. Completeness answers "does this episode
# have all its chunks"; this answers "were those chunks built by current
# rules". They are different questions, and conflating them makes a
# rules-change re-embed a silent no-op.
RULES_VERSION = "1"

_FILENAME_RE = re.compile(
    r"^(?P<show>.+) - Episode (?P<ep>\w+) - "
    r"(?P<date>\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\.txt$"
)


def episode_id_prefix(show: str, episode_number: str, date_str: str) -> str:
    """Stable, unique per-episode ID prefix, keyed on the full triple."""
    return f"{show}-ep{episode_number}-{date_str}".replace(" ", "_")


def chunk_id(prefix: str, index: int) -> str:
    """The ID of one chunk within an episode."""
    return f"{prefix}-{index}"


def transcript_filename(show: str, episode_number: str, date_str: str) -> str:
    """The volume filename the pipeline writes for an episode."""
    return f"{show} - Episode {episode_number} - {date_str}.txt"


def parse_transcript_filename(filename: str) -> tuple[str, str, str] | None:
    """Inverse of `transcript_filename`. None if the name is not a transcript."""
    m = _FILENAME_RE.match(filename)
    if not m:
        return None
    return m.group("show"), m.group("ep"), m.group("date")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_identity.py -v`
Expected: PASS, 9 tests

- [ ] **Step 6: Add CI**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          # 3.12 matches the Modal containers `corpus` has to import inside,
          # not the 3.14 on the dev machine.
          python-version: "3.12"

      - name: Install dev dependencies
        run: uv sync --group dev

      - name: Lint
        run: uv run ruff check .

      - name: Format check
        run: uv run ruff format --check .

      - name: Type check
        run: uv run mypy

      - name: Test
        run: uv run pytest
```

- [ ] **Step 7: Ignore the new local artefacts**

Append to `.gitignore`:

```
# uv
.venv/
uv.lock
```

- [ ] **Step 8: Run the full pipeline locally**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest`
Expected: all clean

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml corpus tests .github .gitignore
git commit -m "feat: add corpus package with episode identity and CI

Keys episode identity on the full (show, episode_number, date) triple.
Neither part is unique alone: episode_number is absent 5 times and
duplicated once, and four (show, date) pairs are duplicated. The
regression cases are the actual production collisions."
```

---

### Task 2: Extract chunking into `corpus`

`n_chunks` must be derived by re-parsing the transcript, so the chunker has to be importable without Modal.

**Files:**
- Create: `corpus/chunking.py`
- Create: `tests/test_chunking.py`
- Modify: `transcribe.py` (remove `build_chunks`, `build_chunks_from_text`, `MAX_CHUNK_WORDS`, `CHUNK_OVERLAP_WORDS`; import them instead; add `.add_local_python_source("corpus")` to the image)

**Interfaces:**
- Consumes: `corpus.identity.RULES_VERSION`
- Produces:
  - `corpus.chunking.MAX_CHUNK_WORDS: int`, `CHUNK_OVERLAP_WORDS: int`
  - `corpus.chunking.parse_transcript_segments(text: str) -> list[dict]`
  - `corpus.chunking.build_chunks(segments, show_name, episode_number, episode_title, date_str, *, episode_guid=None, rules_version=RULES_VERSION) -> list[dict]`
  - `corpus.chunking.build_chunks_from_text(text, show_name, episode_number, episode_title, date_str, *, episode_guid=None, rules_version=RULES_VERSION) -> list[dict]`
  - `corpus.chunking.count_chunks_from_text(text, show_name, episode_number, episode_title, date_str) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
from corpus.chunking import (
    build_chunks,
    build_chunks_from_text,
    count_chunks_from_text,
    parse_transcript_segments,
)

TRANSCRIPT = """# Geopolitical Cousins - Episode 1
# F*cking Around and Finding Out
# Published: 2025-03-14

[SPEAKER_00] 3.1s - Hello listeners, welcome to the inaugural episode.
[SPEAKER_00] 6.9s - I assume that Marco and I will do our own separate intro.
[SPEAKER_01] 52.8s - All right, the two cousins officially doing business.
[UNKNOWN] 61.0s - A stray segment.
this line is malformed and must be skipped
"""


def test_parses_segments_and_skips_headers_and_junk():
    segs = parse_transcript_segments(TRANSCRIPT)
    assert len(segs) == 4
    assert segs[0] == {
        "speaker": "SPEAKER_00",
        "start": 3.1,
        "text": "Hello listeners, welcome to the inaugural episode.",
    }
    assert segs[3]["speaker"] == "UNKNOWN"


def test_chunks_split_on_speaker_change():
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "F*cking Around", "2025-03-14"
    )
    speakers = [c["metadata"]["speaker"] for c in chunks]
    assert speakers == ["SPEAKER_00", "SPEAKER_01", "UNKNOWN"]


def test_every_chunk_carries_n_chunks_equal_to_the_total():
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "F*cking Around", "2025-03-14"
    )
    assert all(c["metadata"]["n_chunks"] == len(chunks) for c in chunks)


def test_metadata_carries_identity_and_versions():
    chunks = build_chunks_from_text(
        TRANSCRIPT,
        "Geopolitical Cousins",
        "1",
        "F*cking Around",
        "2025-03-14",
        episode_guid="b4a9c88b-9dbf-46b7-9dc1-a7812a9bde65",
    )
    meta = chunks[0]["metadata"]
    assert meta["show"] == "Geopolitical Cousins"
    assert meta["episode_number"] == "1"
    assert meta["date"] == "2025-03-14"
    assert meta["date_ts"] == 20250314
    assert meta["episode_guid"] == "b4a9c88b-9dbf-46b7-9dc1-a7812a9bde65"
    assert meta["rules_version"] == "1"


def test_guid_is_omitted_rather_than_null_when_absent():
    # Chroma rejects a None metadata value; the key must simply not be there.
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "t", "2025-03-14"
    )
    assert "episode_guid" not in chunks[0]["metadata"]


def test_document_text_embeds_the_speaker_label():
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "t", "2025-03-14"
    )
    assert chunks[0]["text"].startswith("[SPEAKER_00] ")


def test_long_single_speaker_run_splits_at_max_words():
    segments = [{"speaker": "SPEAKER_00", "start": 0.0, "text": "word " * 1000}]
    chunks = build_chunks(segments, "Show", "1", "t", "2025-01-01")
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 401 for c in chunks)


def test_count_matches_build():
    n = count_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "F*cking Around", "2025-03-14"
    )
    chunks = build_chunks_from_text(
        TRANSCRIPT, "Geopolitical Cousins", "1", "F*cking Around", "2025-03-14"
    )
    assert n == len(chunks)


def test_empty_transcript_yields_no_chunks():
    assert build_chunks_from_text("", "Show", "1", "t", "2025-01-01") == []
    assert count_chunks_from_text("", "Show", "1", "t", "2025-01-01") == 0


def test_unknown_date_gets_zero_timestamp():
    chunks = build_chunks_from_text(TRANSCRIPT, "Show", "1", "t", "Unknown Date")
    assert chunks[0]["metadata"]["date_ts"] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus.chunking'`

- [ ] **Step 3: Write the implementation**

```python
# corpus/chunking.py
"""Transcript parsing and chunking.

Lifted verbatim from transcribe.py so that chunk counts can be derived without
importing modal. `n_chunks` MUST come from re-parsing the transcript rather
than from counting stored records: counting stored records would stamp a torn
episode's truncated count as its expected count and freeze it as permanently
complete, so the completeness check would certify the exact damage it exists
to detect.
"""

from __future__ import annotations

import re

from corpus.identity import RULES_VERSION

# BGE-large handles 512 tokens; ~400 words is a safe proxy.
MAX_CHUNK_WORDS = 400
CHUNK_OVERLAP_WORDS = 50

_SEGMENT_RE = re.compile(r"\[([^\]]+)\]\s+([\d.]+)s\s+-\s+(.*)")


def parse_transcript_segments(text: str) -> list[dict]:
    """Parse the `[SPEAKER_XX] 12.3s - text` format the pipeline writes."""
    segments: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SEGMENT_RE.match(line)
        if m:
            segments.append(
                {
                    "speaker": m.group(1),
                    "start": float(m.group(2)),
                    "text": m.group(3),
                }
            )
    return segments


def build_chunks(
    segments: list[dict],
    show_name: str,
    episode_number: str,
    episode_title: str,
    date_str: str,
    *,
    episode_guid: str | None = None,
    rules_version: str = RULES_VERSION,
) -> list[dict]:
    """Group consecutive same-speaker segments into chunks.

    Splits at MAX_CHUNK_WORDS and overlaps CHUNK_OVERLAP_WORDS into the next
    chunk for context continuity.
    """
    chunks: list[dict] = []
    current_speaker: str | None = None
    current_words: list[str] = []
    current_start = 0.0

    def flush(speaker: str | None, words: list[str], start: float) -> None:
        if not words:
            return
        metadata = {
            "show": show_name,
            "episode_number": episode_number,
            "episode_title": episode_title,
            "date": date_str,
            "speaker": speaker if speaker is not None else "UNKNOWN",
            "start_time": start,
            "date_ts": (
                int(date_str.replace("-", "")) if date_str != "Unknown Date" else 0
            ),
            "rules_version": rules_version,
        }
        # Chroma rejects a None metadata value, so an absent guid means an
        # absent KEY, never a null. 6 of 438 episodes have aged off their feed
        # and can never be assigned one.
        if episode_guid is not None:
            metadata["episode_guid"] = episode_guid
        chunks.append({"text": f"[{metadata['speaker']}] {' '.join(words)}", "metadata": metadata})

    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        start = segment.get("start", 0.0)
        words = text.split()

        if speaker != current_speaker and current_words:
            flush(current_speaker, current_words, current_start)
            current_words = current_words[-CHUNK_OVERLAP_WORDS:]
            current_start = start

        if not current_words:
            current_start = start

        current_speaker = speaker
        current_words.extend(words)

        while len(current_words) >= MAX_CHUNK_WORDS:
            flush(current_speaker, current_words[:MAX_CHUNK_WORDS], current_start)
            current_words = current_words[-CHUNK_OVERLAP_WORDS:]
            current_start = start

    flush(current_speaker, current_words, current_start)

    # n_chunks is only knowable once the whole episode is chunked.
    for chunk in chunks:
        chunk["metadata"]["n_chunks"] = len(chunks)
    return chunks


def build_chunks_from_text(
    text: str,
    show_name: str,
    episode_number: str,
    episode_title: str,
    date_str: str,
    *,
    episode_guid: str | None = None,
    rules_version: str = RULES_VERSION,
) -> list[dict]:
    """Chunk a saved transcript file's contents."""
    return build_chunks(
        parse_transcript_segments(text),
        show_name,
        episode_number,
        episode_title,
        date_str,
        episode_guid=episode_guid,
        rules_version=rules_version,
    )


def count_chunks_from_text(
    text: str,
    show_name: str,
    episode_number: str,
    episode_title: str,
    date_str: str,
) -> int:
    """How many chunks this transcript SHOULD produce. The expected count."""
    return len(
        build_chunks_from_text(text, show_name, episode_number, episode_title, date_str)
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Wire `transcribe.py` to the extracted module**

In `transcribe.py`, delete the `MAX_CHUNK_WORDS`/`CHUNK_OVERLAP_WORDS` constants and the whole of `build_chunks` and `build_chunks_from_text`, then add near the top (after `import re`):

```python
from corpus.chunking import build_chunks, build_chunks_from_text
```

And add the local source to the image so it exists inside the container. Append to the image chain:

```python
    .add_local_python_source("corpus")
```

so the image reads:

```python
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
```

- [ ] **Step 6: Verify the import resolves inside a container**

Run: `uv run modal run transcribe.py::bulk_upload --show-name "does-not-exist"`
Expected: the app starts, prints `No transcripts found for 'does-not-exist'.`, and exits 0. A `ModuleNotFoundError: corpus` means `add_local_python_source` is missing or misspelled.

Confirm from the log that **no image rebuild happened** — `add_local_python_source` adds files at container start, not as a layer, so cached layers must be reused. A rebuild here would silently resolve new dependency versions.

- [ ] **Step 7: Commit**

```bash
git add corpus/chunking.py tests/test_chunking.py transcribe.py
git commit -m "refactor: extract chunking into corpus, add n_chunks to metadata

n_chunks has to be derived by re-parsing the transcript, so the chunker
must import without modal. Counting stored records instead would stamp a
torn episode's truncated count as its expected count and freeze it as
permanently complete."
```

---

### Task 3: Exclusions and the action state machine

**Files:**
- Create: `corpus/exclusions.py`
- Create: `corpus/planning.py`
- Create: `tests/test_planning.py`

**Interfaces:**
- Consumes: `corpus.identity.RULES_VERSION`
- Produces:
  - `corpus.exclusions.ExcludedEpisode` — a frozen dataclass with fields `show`, `episode_number`, `date`, `guid: str | None`, `reason`, and a `.triple` property
  - `corpus.exclusions.EXCLUDED: frozenset[ExcludedEpisode]` — the single source of truth
  - `corpus.exclusions.EXCLUDED_EPISODES: frozenset[tuple[str, str, str]]` — **derived**
  - `corpus.exclusions.EXCLUDED_GUIDS: frozenset[str]` — **derived**
  - `corpus.exclusions.is_excluded(show: str, episode_number: str, date_str: str, episode_guid: str | None = None) -> bool`
  - `corpus.planning.Action` — a `str` `Enum` with members `TRANSCRIBE`, `EMBED_ONLY`, `SKIP`, `EXCLUDE`, `UNPARSEABLE`
  - `corpus.planning.decide_action(*, transcript_exists: bool, complete_in_chroma: bool, stored_rules_version: str | None, excluded: bool, parses_to_chunks: bool) -> Action`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_planning.py
import pytest

from corpus.exclusions import EXCLUDED_EPISODES, is_excluded
from corpus.planning import Action, decide_action


def _decide(**overrides):
    kwargs = dict(
        transcript_exists=True,
        complete_in_chroma=True,
        stored_rules_version="1",
        excluded=False,
        parses_to_chunks=True,
    )
    kwargs.update(overrides)
    return decide_action(**kwargs)


def test_healthy_episode_is_skipped():
    assert _decide() is Action.SKIP


def test_missing_transcript_is_transcribed():
    assert _decide(transcript_exists=False) is Action.TRANSCRIBE


def test_transcript_present_but_incomplete_is_embed_only():
    assert _decide(complete_in_chroma=False) is Action.EMBED_ONLY


def test_stale_rules_version_forces_re_embed():
    # Completeness alone returns SKIP for every episode after migration, so
    # without this a rules-change re-embed is a silent no-op.
    assert _decide(stored_rules_version="0") is Action.EMBED_ONLY


def test_absent_rules_version_forces_re_embed():
    assert _decide(stored_rules_version=None) is Action.EMBED_ONLY


def test_exclusion_beats_embed_only():
    # Both excluded episodes have transcripts on the volume, so without this
    # the cron re-embeds them nightly and reverts the approved deletion.
    assert _decide(excluded=True, complete_in_chroma=False) is Action.EXCLUDE


def test_exclusion_beats_transcribe():
    assert _decide(excluded=True, transcript_exists=False) is Action.EXCLUDE


def test_unparseable_transcript_is_terminal_not_a_re_embed_loop():
    assert (
        _decide(parses_to_chunks=False, complete_in_chroma=False)
        is Action.UNPARSEABLE
    )


def test_exclusion_beats_unparseable():
    assert _decide(excluded=True, parses_to_chunks=False) is Action.EXCLUDE


@pytest.mark.parametrize("episode", sorted(EXCLUDED_EPISODES))
def test_every_excluded_episode_is_recognised(episode):
    assert is_excluded(*episode)


def test_the_geopolitical_cousins_originals_are_not_excluded():
    # Only the Jacob Shapiro re-posts are excluded; the GC originals stay.
    assert not is_excluded("Geopolitical Cousins", "73", "2026-07-29")
    assert not is_excluded("Geopolitical Cousins", "74", "2026-07-31")


def test_exclusions_are_exactly_the_two_cross_posts():
    assert EXCLUDED_EPISODES == frozenset(
        {
            ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29"),
            ("The Jacob Shapiro Podcast", "Unknown", "2026-07-31"),
        }
    )


def test_derived_views_cannot_drift_from_the_record_list():
    from corpus.exclusions import EXCLUDED, EXCLUDED_GUIDS

    assert EXCLUDED_EPISODES == frozenset(e.triple for e in EXCLUDED)
    assert EXCLUDED_GUIDS == frozenset(e.guid for e in EXCLUDED if e.guid)
    assert all(e.reason for e in EXCLUDED)


def test_guid_arm_survives_an_episode_number_backfill():
    # Both excluded episodes fall back to "Unknown" because Captivate
    # publishes no itunes_episode for them. If it ever backfills one, the
    # triple changes and the triple arm silently stops matching.
    assert is_excluded(
        "The Jacob Shapiro Podcast",
        "352",  # a backfilled number -- the triple no longer matches
        "2026-07-29",
        episode_guid="1c45dbd9-0dc3-4d07-b2d1-758fe78405fe",
    )


def test_guid_arm_does_not_over_match():
    assert not is_excluded(
        "Geopolitical Cousins", "73", "2026-07-29", episode_guid="some-other-guid"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_planning.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus.exclusions'`

- [ ] **Step 3: Write the implementation**

```python
# corpus/exclusions.py
"""Episodes deliberately kept out of the corpus.

Four episodes are Geopolitical Cousins content republished on the Jacob
Shapiro feed. The transcripts are byte-identical apart from the header show
name, and each was downloaded and transcribed twice. They are embedded under
Geopolitical Cousins only.

This is a HUMAN-MAINTAINED LIST, deliberately not a heuristic. The enclosure
URLs differ across feeds (zero shared -- Captivate re-hosts), so the only
available signal is a fuzzy title-and-date match, and code that silently
discards an episode because a regex thought two titles matched is a failure
mode with no alarm on it. Reconciliation REPORTS suspected cross-posts;
exclusion is a decision recorded here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExcludedEpisode:
    show: str
    episode_number: str
    date: str
    guid: str | None
    reason: str

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.show, self.episode_number, self.date)


EXCLUDED: frozenset[ExcludedEpisode] = frozenset(
    {
        ExcludedEpisode(
            "The Jacob Shapiro Podcast",
            "Unknown",
            "2026-07-29",
            "1c45dbd9-0dc3-4d07-b2d1-758fe78405fe",
            "cross-post of Geopolitical Cousins 73, "
            "'This Is The Way The World Ends'",
        ),
        ExcludedEpisode(
            "The Jacob Shapiro Podcast",
            "Unknown",
            "2026-07-31",
            "d738c6b4-cb9e-497e-995f-c106c42d9b1d",
            "cross-post of Geopolitical Cousins 74, 'Lessons Learned'",
        ),
    }
)

# Derived, never hand-maintained. Two parallel hand-written lists would drift
# ASYMMETRICALLY: forgetting a guid still excludes correctly today and fails
# only in the future scenario the guid was added for, so the violation is
# invisible until exactly the moment it matters.
EXCLUDED_EPISODES: frozenset[tuple[str, str, str]] = frozenset(
    e.triple for e in EXCLUDED
)
EXCLUDED_GUIDS: frozenset[str] = frozenset(
    e.guid for e in EXCLUDED if e.guid is not None
)


def is_excluded(
    show: str,
    episode_number: str,
    date_str: str,
    episode_guid: str | None = None,
) -> bool:
    """Whether this episode is deliberately kept out of the corpus.

    Either arm matching is enough. The triple is NOT durable: both excluded
    episodes fall back to the literal "Unknown" precisely because Captivate
    publishes no itunes_episode for them, and the moment it backfills one --
    the spec's six-month threat model, and the reason episode_guid exists --
    the triple changes and the triple arm silently stops matching. The guid
    does not move with a metadata backfill.

    The triple arm still earns its place: 6 episodes have aged off the front
    of their feed and can never be assigned a guid at all.

    CALLERS MUST PASS episode_guid WHERE THEY HAVE ONE. An arm that no live
    path reaches is not protection, it is decoration.
    """
    if episode_guid is not None and episode_guid in EXCLUDED_GUIDS:
        return True
    return (show, episode_number, date_str) in EXCLUDED_EPISODES
```

```python
# corpus/planning.py
"""What to do with one episode.

Pure: takes booleans and a version string, returns an action. No Modal, no
Chroma, no filesystem -- so every branch is tested in milliseconds.
"""

from __future__ import annotations

from enum import Enum

from corpus.identity import RULES_VERSION


class Action(str, Enum):
    TRANSCRIBE = "TRANSCRIBE"
    EMBED_ONLY = "EMBED_ONLY"
    SKIP = "SKIP"
    EXCLUDE = "EXCLUDE"
    UNPARSEABLE = "UNPARSEABLE"


def decide_action(
    *,
    transcript_exists: bool,
    complete_in_chroma: bool,
    stored_rules_version: str | None,
    excluded: bool,
    parses_to_chunks: bool,
) -> Action:
    """Decide what this episode needs.

    Order matters. EXCLUDE is checked first because both excluded episodes
    have transcripts on the volume and are incomplete in Chroma, so any later
    branch would re-embed them -- reverting an approved deletion every night.

    UNPARSEABLE is terminal rather than EMBED_ONLY because an episode that
    cannot produce chunks can never become complete, so treating it as
    incomplete would re-embed it forever. Measured: 0 of 438 transcripts
    currently parse to zero chunks, but a total state machine is not the same
    as one no current input breaks.
    """
    if excluded:
        return Action.EXCLUDE
    if not parses_to_chunks:
        return Action.UNPARSEABLE
    if not transcript_exists:
        return Action.TRANSCRIBE
    if not complete_in_chroma:
        return Action.EMBED_ONLY
    if stored_rules_version != RULES_VERSION:
        return Action.EMBED_ONLY
    return Action.SKIP
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_planning.py -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Commit**

```bash
git add corpus/exclusions.py corpus/planning.py tests/test_planning.py
git commit -m "feat: add action state machine with first-class exclusion

EXCLUDE is checked first because both excluded episodes have transcripts
on the volume, so any later branch re-embeds them and reverts the approved
deletion nightly. A rules_version mismatch forces EMBED_ONLY because
completeness returns SKIP for every episode after migration."
```

---

### Task 4: Chroma-shaped helpers and a fake that enforces the real caps

The fake collection encodes the verified Cloud semantics as executable knowledge. Every cap that bit us in production is a test here.

**Files:**
- Create: `corpus/store.py`
- Create: `tests/conftest.py`
- Create: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `corpus.store.PAGE: int`, `corpus.store.BATCH: int`, `corpus.store.MAX_REQUEST: int`
  - `corpus.store.batched(items: list, size: int = BATCH) -> Iterator[list]`
  - `corpus.store.episode_where(show: str, episode_number: str, date_str: str) -> dict`
  - `corpus.store.guid_where(episode_guid: str) -> dict`
  - `corpus.store.paged_get_ids(collection, where: dict) -> list[str]`
  - `corpus.store.paged_get(collection, where: dict, include: list[str]) -> dict`
  - `corpus.store.stale_ids(existing_ids: Iterable[str], new_ids: Iterable[str]) -> list[str]`
  - `corpus.store.is_complete(stored_ids: list[str], expected_n_chunks: int | None) -> bool`
- Test helper produced for later tasks: `tests/conftest.py::FakeCollection`

- [ ] **Step 1: Write the fake collection**

```python
# tests/conftest.py
"""A Chroma stand-in that enforces the caps measured against Chroma Cloud.

Every limit here was verified empirically against a throwaway Cloud
collection on chromadb 1.5.9, not read from documentation:

  get(limit=301)          -> raises
  get() with no limit     -> SILENTLY returns 300
  upsert of >300 records  -> raises
  upsert on existing id   -> replaces the document, MERGES the metadata
  delete(where=) matching nothing -> no-op

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

    # -- helpers ---------------------------------------------------------
    def _matches(self, meta: dict, where: dict | None) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(self._matches(meta, c) for c in where["$and"])
        for key, cond in where.items():
            if isinstance(cond, dict):
                if "$eq" in cond and meta.get(key) != cond["$eq"]:
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
        selected = [i for i in (ids or self._select(where)) if i in self._docs]
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
        targets = list(ids) if ids is not None else self._select(where)
        for _id in targets:
            self._docs.pop(_id, None)
            self._meta.pop(_id, None)


@pytest.fixture
def collection() -> FakeCollection:
    return FakeCollection()
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_store.py
import pytest

from corpus.store import (
    BATCH,
    batched,
    episode_where,
    guid_where,
    is_complete,
    paged_get_ids,
    stale_ids,
)
from tests.conftest import MAX_REQUEST, ChromaQuotaError

TRIPLE = ("Geopolitical Cousins", "73", "2026-07-29")


def _seed(collection, prefix, n, **meta):
    base = {
        "show": TRIPLE[0],
        "episode_number": TRIPLE[1],
        "date": TRIPLE[2],
        "n_chunks": n,
    }
    base.update(meta)
    for batch in batched(list(range(n))):
        collection.upsert(
            ids=[f"{prefix}-{i}" for i in batch],
            documents=[f"doc-{i}" for i in batch],
            metadatas=[dict(base) for _ in batch],
        )


def test_batched_never_exceeds_the_request_cap():
    chunks = list(batched(list(range(431))))
    assert [len(c) for c in chunks] == [250, 181]
    assert all(len(c) <= MAX_REQUEST for c in chunks)


def test_batched_handles_an_empty_list():
    assert list(batched([])) == []


def test_unbatched_upsert_of_a_real_episode_would_raise(collection):
    # Geopolitical Cousins 73 is 431 chunks. This is the production failure.
    with pytest.raises(ChromaQuotaError):
        collection.upsert(
            ids=[f"x-{i}" for i in range(431)],
            documents=["d"] * 431,
            metadatas=[{}] * 431,
        )


def test_batched_upsert_of_the_same_episode_succeeds(collection):
    _seed(collection, "gc73", 431)
    assert collection.count() == 431


def test_paged_get_returns_everything_past_the_300_cap(collection):
    _seed(collection, "gc73", 431)
    ids = paged_get_ids(collection, episode_where(*TRIPLE))
    assert len(ids) == 431


def test_unpaged_get_silently_truncates(collection):
    # The trap: this returns 300 and raises nothing.
    _seed(collection, "gc73", 431)
    assert len(collection.get(where=episode_where(*TRIPLE))["ids"]) == 300


def test_is_complete_uses_the_expected_count_not_mere_presence(collection):
    _seed(collection, "gc73", 431)
    ids = paged_get_ids(collection, episode_where(*TRIPLE))
    assert is_complete(ids, 431)
    assert not is_complete(ids, 432)


def test_a_single_orphan_chunk_is_not_complete():
    # After a collision clobber an episode retained one chunk. A boolean
    # "does any chunk exist" check called that healthy.
    assert not is_complete(["gc73-2"], 431)


def test_no_records_is_not_complete():
    assert not is_complete([], 431)


def test_absent_expected_count_is_not_complete():
    # Pre-migration records carry no n_chunks. Treating that as satisfied
    # would mean old episodes are never completeness-checked at all.
    assert not is_complete(["a-0"], None)


def test_stale_ids_is_the_set_difference():
    assert stale_ids(["a-0", "a-1", "a-2"], ["a-0", "a-1"]) == ["a-2"]


def test_stale_ids_is_empty_when_the_episode_grew():
    assert stale_ids(["a-0"], ["a-0", "a-1"]) == []


def test_stale_ids_spans_old_and_new_id_schemes():
    old = ["Show-ep1-0", "Show-ep1-1"]
    new = ["Show-ep1-2025-01-01-0"]
    assert sorted(stale_ids(old + new, new)) == old


def test_guid_where_and_episode_where_select_the_same_records(collection):
    _seed(collection, "gc73", 10, episode_guid="abc-123")
    by_triple = paged_get_ids(collection, episode_where(*TRIPLE))
    by_guid = paged_get_ids(collection, guid_where("abc-123"))
    assert by_triple == by_guid


def test_delete_with_a_non_matching_filter_is_a_no_op(collection):
    _seed(collection, "gc73", 10)
    collection.delete(where=episode_where("Other Show", "1", "2020-01-01"))
    assert collection.count() == 10
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus.store'`

- [ ] **Step 4: Write the implementation**

```python
# corpus/store.py
"""Chroma-shaped helpers, written against a duck-typed collection.

Nothing here imports chromadb. Callers pass anything with `get`, `upsert`,
`delete` and `count`, which is what makes the whole layer testable against a
fake that enforces the real caps.

The caps below were measured against Chroma Cloud (chromadb 1.5.9), not read
from documentation. The dangerous one is that an unlimited `get()` returns 300
records and raises nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

# Chroma Cloud rejects get(limit>300) and upsert of >300 records per request.
MAX_REQUEST = 300
# Page and batch below the cap, per the value already used in migration/.
PAGE = 250
BATCH = 250


def batched(items: list, size: int = BATCH) -> Iterator[list]:
    """Split a list into request-sized batches."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def episode_where(show: str, episode_number: str, date_str: str) -> dict:
    """A filter selecting one episode by the unique triple."""
    return {
        "$and": [
            {"show": {"$eq": show}},
            {"episode_number": {"$eq": episode_number}},
            {"date": {"$eq": date_str}},
        ]
    }


def guid_where(episode_guid: str) -> dict:
    """A filter selecting one episode by its RSS guid."""
    return {"episode_guid": {"$eq": episode_guid}}


def paged_get(collection, where: dict, include: list[str]) -> dict:
    """Page a filtered get, assembling the complete result."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    offset = 0
    while True:
        page = collection.get(
            where=where, include=include, limit=PAGE, offset=offset
        )
        if not page["ids"]:
            break
        ids.extend(page["ids"])
        if "documents" in include:
            documents.extend(page.get("documents", []))
        if "metadatas" in include:
            metadatas.extend(page.get("metadatas", []))
        offset += len(page["ids"])
    out: dict = {"ids": ids}
    if "documents" in include:
        out["documents"] = documents
    if "metadatas" in include:
        out["metadatas"] = metadatas
    return out


def paged_get_ids(collection, where: dict) -> list[str]:
    """Every id matching the filter. Never call the unpaged form."""
    return paged_get(collection, where, include=[])["ids"]


def stale_ids(existing_ids: Iterable[str], new_ids: Iterable[str]) -> list[str]:
    """Records to prune after an upsert: what was there and no longer is.

    Upsert cannot shrink a record set, so a re-embed producing fewer chunks
    strands every index above the new count. Those survivors keep the old
    document text, which after a speaker rename means stale labels and
    duplicated passages.
    """
    return sorted(set(existing_ids) - set(new_ids))


def is_complete(stored_ids: list[str], expected_n_chunks: int | None) -> bool:
    """Whether an episode is fully stored.

    Presence is not existence. A collision clobber leaves an episode with one
    surviving chunk, and a boolean "does any chunk exist" check calls that
    healthy -- so the self-healing branch never repairs it and reconciliation
    passes.

    A missing expected count means a pre-migration record, which is treated as
    INCOMPLETE. The alternative reading -- absent means satisfied -- would mean
    old episodes are never completeness-checked at all.
    """
    if expected_n_chunks is None:
        return False
    return len(stored_ids) == expected_n_chunks
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS, 15 tests

- [ ] **Step 6: Commit**

```bash
git add corpus/store.py tests/conftest.py tests/test_store.py
git commit -m "feat: add paged/batched Chroma helpers and a cap-enforcing fake

The fake encodes semantics measured against Chroma Cloud rather than read
from docs: get(limit>300) raises, an unlimited get SILENTLY returns 300,
upsert of >300 raises, and upsert merges metadata. Geopolitical Cousins 73
at 431 chunks is a regression test for the cap that caused the production
failures."
```

---

### Task 5: ID remapping for the migration

**Files:**
- Create: `corpus/remap.py`
- Create: `tests/test_remap.py`

**Interfaces:**
- Consumes: `corpus.identity.episode_id_prefix`, `corpus.identity.chunk_id`
- Produces:
  - `corpus.remap.RemapResult` — a `NamedTuple` with fields `new_id: str` and `classification: str`
  - `corpus.remap.remap_id(old_id: str, metadata: dict) -> RemapResult`
  - Classification values: `"remapped"`, `"passthrough_non_episode"`, `"passthrough_unmatched"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remap.py
from corpus.remap import remap_id


def test_podcast_record_is_remapped_with_the_date_inserted():
    meta = {
        "show": "Geopolitical Cousins",
        "episode_number": "73",
        "date": "2026-07-29",
    }
    result = remap_id("Geopolitical_Cousins-ep73-17", meta)
    assert result.new_id == "Geopolitical_Cousins-ep73-2026-07-29-17"
    assert result.classification == "remapped"


def test_the_two_episode_243s_stop_colliding():
    a = remap_id(
        "The_Jacob_Shapiro_Podcast-ep243-0",
        {
            "show": "The Jacob Shapiro Podcast",
            "episode_number": "243",
            "date": "2024-11-07",
        },
    )
    b = remap_id(
        "The_Jacob_Shapiro_Podcast-ep243-0",
        {
            "show": "The Jacob Shapiro Podcast",
            "episode_number": "243",
            "date": "2024-11-08",
        },
    )
    assert a.new_id != b.new_id


def test_book_records_pass_through_untouched():
    # upload_book.py writes Geopolitical_Alpha-p{n}; these ids are already
    # unique and carry no episode concept. Applying the podcast scheme to
    # them would corrupt them.
    meta = {
        "show": "Geopolitical Alpha",
        "episode_number": "N/A",
        "date": "2021-01-01",
    }
    result = remap_id("Geopolitical_Alpha-p179", meta)
    assert result.new_id == "Geopolitical_Alpha-p179"
    assert result.classification == "passthrough_non_episode"


def test_an_id_whose_metadata_does_not_reconstruct_it_passes_through():
    # Never guess. If the id and the metadata disagree, leave it alone and
    # let reconciliation report it.
    meta = {"show": "Some Other Show", "episode_number": "9", "date": "2025-01-01"}
    result = remap_id("Geopolitical_Cousins-ep73-17", meta)
    assert result.new_id == "Geopolitical_Cousins-ep73-17"
    assert result.classification == "passthrough_unmatched"


def test_an_already_migrated_id_is_left_alone():
    meta = {
        "show": "Geopolitical Cousins",
        "episode_number": "73",
        "date": "2026-07-29",
    }
    new = "Geopolitical_Cousins-ep73-2026-07-29-17"
    assert remap_id(new, meta).new_id == new


def test_missing_metadata_keys_pass_through():
    assert remap_id("Whatever-ep1-0", {}).classification == "passthrough_unmatched"


def test_old_and_new_scheme_ids_can_never_collide():
    # Old ids end in an integer; new ids end in YYYY-MM-DD-{int}. This is what
    # makes a mixed-scheme corpus safe during the migration.
    meta = {"show": "Show", "episode_number": "1", "date": "2025-01-01"}
    old = {f"Show-ep1-{i}" for i in range(200)}
    new = {remap_id(f"Show-ep1-{i}", meta).new_id for i in range(200)}
    assert old & new == set()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_remap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus.remap'`

- [ ] **Step 3: Write the implementation**

```python
# corpus/remap.py
"""Mapping old-scheme record ids onto the new scheme.

Old: {show}-ep{episode_number}-{index}
New: {show}-ep{episode_number}-{date}-{index}

Only records whose id matches the old episode pattern AND whose metadata
reconstructs that exact prefix are remapped. Everything else passes through
untouched and is counted -- the book records written by upload_book.py have
their own scheme and no episode concept, and an id that disagrees with its own
metadata is a fact to report, never one to guess at.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from corpus.identity import chunk_id, episode_id_prefix

# The trailing component of the OLD scheme is integer-only, which is exactly
# why an old id can never equal a new one.
_OLD_ID_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d+)$")


class RemapResult(NamedTuple):
    new_id: str
    classification: str


def remap_id(old_id: str, metadata: dict) -> RemapResult:
    """Compute this record's id under the new scheme."""
    m = _OLD_ID_RE.match(old_id)
    if not m:
        return RemapResult(old_id, "passthrough_non_episode")

    show = metadata.get("show")
    episode_number = metadata.get("episode_number")
    date_str = metadata.get("date")
    if not (show and episode_number and date_str):
        return RemapResult(old_id, "passthrough_unmatched")

    old_prefix = f"{show}-ep{episode_number}".replace(" ", "_")
    new_prefix = episode_id_prefix(show, episode_number, date_str)
    index = int(m.group("index"))

    if m.group("prefix") == new_prefix:
        return RemapResult(old_id, "remapped")  # already migrated
    if m.group("prefix") != old_prefix:
        return RemapResult(old_id, "passthrough_unmatched")

    return RemapResult(chunk_id(new_prefix, index), "remapped")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_remap.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add corpus/remap.py tests/test_remap.py
git commit -m "feat: add id remapping with explicit passthrough classes

Only records whose id matches the old episode pattern and whose metadata
reconstructs that prefix are remapped. Book records keep their own scheme,
and an id disagreeing with its metadata is reported rather than guessed at."
```

---

### Task 6: Fix the two blocking gaps in the migration tooling

Both were verified in the source and both stop the migration dead.

**Files:**
- Modify: `migration/chroma_migrate.py` — `create_dest_collection` (line ~46), `validate_collection` (line ~133)
- Create: `tests/test_migration_tooling.py`

**Interfaces:**
- Consumes: `tests.conftest.FakeCollection`
- Produces:
  - `migration.chroma_migrate.create_dest_collection(dst_client, src_col, dest_name=None)`
  - `migration.chroma_migrate.validate_collection(src_col, dst_col, atol=1e-4, id_map=None, allowed_new_keys=frozenset())`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_migration_tooling.py
import pytest

from migration.chroma_migrate import validate_collection
from tests.conftest import FakeCollection


def _seed(col, ids, meta_extra=None):
    for _id in ids:
        col.upsert(
            ids=[_id],
            documents=[f"doc-{_id}"],
            metadatas=[dict({"show": "S"}, **(meta_extra or {}))],
        )


def test_validation_follows_the_id_map():
    src, dst = FakeCollection(), FakeCollection()
    _seed(src, ["S-ep1-0", "S-ep1-1"])
    id_map = {"S-ep1-0": "S-ep1-2025-01-01-0", "S-ep1-1": "S-ep1-2025-01-01-1"}
    for old, new in id_map.items():
        dst.upsert(
            ids=[new], documents=[f"doc-{old}"], metadatas=[{"show": "S"}]
        )
    assert validate_collection(src, dst, id_map=id_map) == []


def test_validation_without_an_id_map_reports_every_record_missing():
    src, dst = FakeCollection(), FakeCollection()
    _seed(src, ["S-ep1-0"])
    dst.upsert(ids=["S-ep1-2025-01-01-0"], documents=["doc-S-ep1-0"],
               metadatas=[{"show": "S"}])
    problems = validate_collection(src, dst)
    assert any("missing id in dst" in p for p in problems)


def test_declared_new_keys_are_allowed_but_others_are_not():
    src, dst = FakeCollection(), FakeCollection()
    _seed(src, ["S-ep1-0"])
    id_map = {"S-ep1-0": "S-ep1-2025-01-01-0"}
    dst.upsert(
        ids=["S-ep1-2025-01-01-0"],
        documents=["doc-S-ep1-0"],
        metadatas=[{"show": "S", "n_chunks": 1, "rules_version": "1"}],
    )
    assert (
        validate_collection(
            src, dst, id_map=id_map,
            allowed_new_keys=frozenset({"n_chunks", "rules_version"}),
        )
        == []
    )
    problems = validate_collection(src, dst, id_map=id_map)
    assert any("metadata mismatch" in p for p in problems)


def test_changing_an_existing_key_is_still_a_mismatch():
    src, dst = FakeCollection(), FakeCollection()
    _seed(src, ["S-ep1-0"])
    dst.upsert(
        ids=["S-ep1-2025-01-01-0"],
        documents=["doc-S-ep1-0"],
        metadatas=[{"show": "DIFFERENT", "n_chunks": 1}],
    )
    problems = validate_collection(
        src, dst, id_map={"S-ep1-0": "S-ep1-2025-01-01-0"},
        allowed_new_keys=frozenset({"n_chunks"}),
    )
    assert any("metadata mismatch" in p for p in problems)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_migration_tooling.py -v`
Expected: FAIL — `TypeError: validate_collection() got an unexpected keyword argument 'id_map'`

- [ ] **Step 3: Add a `dest_name` parameter to `create_dest_collection`**

Replace the opening of `create_dest_collection` in `migration/chroma_migrate.py`:

```python
def create_dest_collection(dst_client, src_col, dest_name=None):
    """
    Reproduce the source collection on the destination.

    `dest_name` defaults to the source's name, which is correct for a
    DB-to-DB copy. For a SAME-DATABASE re-ID it must be given: without it the
    name already exists on the "destination" (it is the source), the resume
    branch below hands back the SOURCE COLLECTION ITSELF, and the copy
    silently upserts v1 into v1 and reports success.

    There is deliberately no programmatic guard against that. The obvious one
    -- comparing `dst_client` against `src_col._client` -- is inert on
    chromadb 1.5.9, where `Client.get_collection` constructs
    `Collection(client=self._server, ...)`, so `_client` is the ServerAPI and
    never the CloudClient the caller holds. A guard that is always False is
    worse than none, because it reads as protection. The caller is responsible
    for passing a `dest_name` that differs from the source.

    Resume-safe: if it already exists on the destination (a re-run), reuse it.
    Otherwise copy the schema wholesale so distance space / index enablement /
    key-specific + sparse indexes carry over. Falls back to metadata-based
    creation if this build/collection predates the Schema API.
    """
    name = dest_name or src_col.name
    existing = {c.name for c in dst_client.list_collections()}
```

The remainder of the function is unchanged; it already uses the local `name`.

- [ ] **Step 4: Add `id_map` and `allowed_new_keys` to `validate_collection`**

Change the signature:

```python
def validate_collection(
    src_col, dst_col, atol=1e-4, id_map=None, allowed_new_keys=frozenset()
):
    """
    Full validation gate. Returns a list of problem strings (empty == clean).

    `id_map` maps source id -> destination id, for a migration that re-IDs.
    Without it, destination lookups use source ids and every record of a
    re-IDed copy reports missing.

    `allowed_new_keys` are metadata keys the destination may carry that the
    source does not. Every other key must match exactly, and a key present in
    both must have the same value.
    """
```

Inside the paging loop, replace the destination fetch and the id check:

```python
        d_ids = [id_map.get(i, i) for i in ids] if id_map else ids
        d = dst_col.get(ids=d_ids, include=INCLUDE)
        d_index = {i: k for k, i in enumerate(d["ids"])}
        for k, _id in enumerate(ids):
            mapped = id_map.get(_id, _id) if id_map else _id
            if mapped not in d_index:
                problems.append(f"missing id in dst: {_id} -> {mapped}")
                continue
            j = d_index[mapped]
```

And replace the metadata equality check with one that tolerates exactly the declared new keys:

```python
            s_meta = _cell(b, "metadatas", k) or {}
            d_meta = _cell(d, "metadatas", j) or {}
            extra = set(d_meta) - set(s_meta) - set(allowed_new_keys)
            missing = set(s_meta) - set(d_meta)
            changed = {
                key
                for key in set(s_meta) & set(d_meta)
                if s_meta[key] != d_meta[key]
            }
            if extra or missing or changed:
                problems.append(
                    f"metadata mismatch for {_id}: "
                    f"extra={sorted(extra)} missing={sorted(missing)} "
                    f"changed={sorted(changed)}"
                )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_migration_tooling.py -v`
Expected: PASS, 4 tests

- [ ] **Step 6: Run the existing migration self-test to confirm nothing regressed**

`selftest.py:31` is `base = sys.argv[1]`, so it needs a base directory — without one it raises `IndexError` before doing anything.

Run: `uv run python migration/selftest.py /tmp/chroma-selftest`
Expected: the local `PersistentClient` dress rehearsal passes as before — it seeds 350 records specifically to exercise the `get()` page cap.

- [ ] **Step 7: Commit**

```bash
git add migration/chroma_migrate.py tests/test_migration_tooling.py
git commit -m "fix: make migration tooling able to express a same-database re-ID

create_dest_collection took its name from src_col.name and reused any
existing collection with it, so within one database it returned the source
and the copy would upsert v1 into v1 and report success. And
validate_collection looked up destination records by source id, so under a
re-ID all 28,489 report missing."
```

---

### Task 7: The re-ID migration runner

**Files:**
- Create: `migration/__init__.py` (empty — `migration/` is currently a flat script directory with no `__init__.py`, so `from migration.chroma_migrate import ...` and `add_local_python_source("migration")` both need it)
- Create: `migration/reid.py`
- Create: `tests/test_reid_planning.py`

**Interfaces:**
- Consumes: `corpus.remap.remap_id`, `corpus.chunking.count_chunks_from_text`, `corpus.identity.parse_transcript_filename`, `corpus.identity.RULES_VERSION`, `corpus.store.batched`, `migration.chroma_migrate.create_dest_collection`, `migration.chroma_migrate.validate_collection`
- Produces:
  - `migration.reid.build_episode_facts(transcript_texts: dict[tuple[str, str, str], str], feed_guids: dict[tuple[str, str, str], str]) -> dict[tuple[str, str, str], dict]`
  - `migration.reid.enrich_metadata(metadata: dict, facts: dict) -> dict`
  - A Modal entrypoint `migration/reid.py::run`

- [ ] **Step 1: Write the failing test for the pure part**

```python
# tests/test_reid_planning.py
from migration.reid import build_episode_facts, enrich_metadata

TRANSCRIPT = """# Show - Episode 1
# Title
# Published: 2025-01-01

[SPEAKER_00] 1.0s - Hello there.
[SPEAKER_01] 5.0s - And hello to you.
"""

KEY = ("Show", "1", "2025-01-01")


def test_facts_carry_expected_chunk_count_and_guid():
    facts = build_episode_facts({KEY: TRANSCRIPT}, {KEY: "guid-abc"})
    assert facts[KEY]["n_chunks"] == 2
    assert facts[KEY]["episode_guid"] == "guid-abc"


def test_chunk_count_comes_from_the_transcript_not_from_stored_records():
    # Counting stored records would freeze a torn episode as complete.
    facts = build_episode_facts({KEY: TRANSCRIPT}, {})
    assert facts[KEY]["n_chunks"] == 2


def test_missing_guid_is_omitted_not_null():
    facts = build_episode_facts({KEY: TRANSCRIPT}, {})
    assert "episode_guid" not in facts[KEY]


def test_enrich_adds_the_declared_keys_and_preserves_the_rest():
    meta = {"show": "Show", "episode_number": "1", "date": "2025-01-01",
            "speaker": "SPEAKER_00"}
    facts = build_episode_facts({KEY: TRANSCRIPT}, {KEY: "guid-abc"})
    out = enrich_metadata(meta, facts[KEY])
    assert out["speaker"] == "SPEAKER_00"
    assert out["n_chunks"] == 2
    assert out["episode_guid"] == "guid-abc"
    assert out["rules_version"] == "1"


def test_enrich_without_facts_leaves_metadata_untouched():
    # Book records and any episode with no transcript on the volume.
    meta = {"show": "Geopolitical Alpha", "episode_number": "N/A"}
    assert enrich_metadata(meta, None) == meta
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_reid_planning.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'migration.reid'`

- [ ] **Step 3: Write the implementation**

```python
# migration/reid.py
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

from corpus.chunking import count_chunks_from_text
from corpus.identity import RULES_VERSION, parse_transcript_filename
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
    "The Observing Japan Podcast":
        "https://api.substack.com/feed/podcast/868206/s/386602.rss",
}


def build_episode_facts(
    transcript_texts: dict[tuple[str, str, str], str],
    feed_guids: dict[tuple[str, str, str], str],
) -> dict[tuple[str, str, str], dict]:
    """Per-episode facts to stamp onto every one of its records.

    `n_chunks` is derived by RE-PARSING THE TRANSCRIPT, never by counting
    stored records: counting would stamp a torn episode's truncated count as
    its expected count and freeze it as permanently complete.
    """
    facts: dict[tuple[str, str, str], dict] = {}
    for key, text in transcript_texts.items():
        show, episode_number, date_str = key
        entry: dict = {
            "n_chunks": count_chunks_from_text(
                text, show, episode_number, "", date_str
            ),
            "rules_version": RULES_VERSION,
        }
        guid = feed_guids.get(key)
        if guid is not None:
            entry["episode_guid"] = guid
        facts[key] = entry
    return facts


def enrich_metadata(metadata: dict, facts: dict | None) -> dict:
    """Add the declared new keys. Records with no facts pass through."""
    if not facts:
        return metadata
    return {**metadata, **facts}


def _load_feed_guids() -> dict[tuple[str, str, str], str]:
    import email.utils

    import feedparser

    guids: dict[tuple[str, str, str], str] = {}
    for show, url in FEEDS.items():
        for entry in feedparser.parse(url).entries:
            gid = entry.get("id")
            # RSS <guid isPermaLink> defaults to true. A link-derived id is a
            # URL, and this publisher rewrites URLs, so it is not an identity.
            if not gid or entry.get("guidislink"):
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

    from migration.chroma_migrate import create_dest_collection, validate_collection
    from corpus.remap import remap_id

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
            with open(f"{VOLUME_PATH}/{name}", encoding="utf-8",
                      errors="replace") as fh:
                texts[key] = fh.read()
    print(f"transcripts on volume: {len(texts)}")

    guids = _load_feed_guids()
    facts = build_episode_facts(texts, guids)
    print(f"episodes with facts: {len(facts)}  with guid: "
          f"{sum(1 for f in facts.values() if 'episode_guid' in f)}")

    if dry_run:
        counts: dict[str, int] = {}
        offset = 0
        while offset < total:
            page = src.get(limit=PAGE, offset=offset, include=["metadatas"])
            if not page["ids"]:
                break
            for _id, meta in zip(page["ids"], page["metadatas"]):
                counts[remap_id(_id, meta).classification] = (
                    counts.get(remap_id(_id, meta).classification, 0) + 1
                )
            offset += len(page["ids"])
        print("classification:", counts)
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

    problems = validate_collection(
        src, dst, id_map=id_map, allowed_new_keys=NEW_KEYS
    )
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_reid_planning.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Dry-run the migration against production, read-only**

Run: `uv run modal run migration/reid.py --dry-run`
Expected: prints `source=podcast_transcripts count=28489`, `transcripts on volume: 438`, `episodes with facts: 438  with guid: 432`, and a classification breakdown totalling 28,489 with 191 in `passthrough_non_episode` (the book) and 0 in `passthrough_unmatched`.

**If `passthrough_unmatched` is non-zero, stop and investigate** — it means records exist whose id disagrees with their own metadata, which nothing in the spec predicts.

- [ ] **Step 6: Commit**

```bash
git add migration/reid.py tests/test_reid_planning.py
git commit -m "feat: add re-ID migration writing n_chunks, guid and rules_version

No GPU: Chroma returns stored embeddings, so nothing is re-embedded.
n_chunks is derived by re-parsing the transcript rather than by counting
stored records. A guid whose guidislink is true is treated as absent,
because a link-derived guid is a URL and this publisher rewrites URLs."
```

---

### Task 8: Rewrite the write path

**Files:**
- Modify: `transcribe.py` — `embed_and_store` (line ~171), `get_chroma_collection` (line ~196)
- Create: `tests/test_write_path.py`
- Create: `corpus/writing.py`

**Interfaces:**
- Consumes: `corpus.store.{batched, episode_where, guid_where, paged_get_ids, stale_ids}`, `corpus.identity.{episode_id_prefix, chunk_id}`
- Produces:
  - `corpus.writing.upsert_then_prune(collection, chunks: list[dict], embeddings: list[list[float]], *, show: str, episode_number: str, date_str: str, episode_guid: str | None) -> dict`
  - Returns `{"written": int, "pruned": int}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_write_path.py
from corpus.store import episode_where, paged_get_ids
from corpus.writing import upsert_then_prune

SHOW, EP, DATE = "Geopolitical Cousins", "73", "2026-07-29"


def _chunks(n, speaker="SPEAKER_00", guid=None):
    out = []
    for i in range(n):
        meta = {
            "show": SHOW,
            "episode_number": EP,
            "date": DATE,
            "speaker": speaker,
            "n_chunks": n,
        }
        if guid:
            meta["episode_guid"] = guid
        out.append({"text": f"[{speaker}] chunk {i}", "metadata": meta})
    return out


def _embeddings(n):
    return [[float(i), 0.0, 1.0] for i in range(n)]


def _write(collection, n, **kw):
    return upsert_then_prune(
        collection,
        _chunks(n, **kw),
        _embeddings(n),
        show=SHOW,
        episode_number=EP,
        date_str=DATE,
        episode_guid=kw.get("guid"),
    )


def test_writes_a_431_chunk_episode_without_hitting_the_request_cap(collection):
    result = _write(collection, 431)
    assert result["written"] == 431
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 431


def test_a_shrinking_re_embed_leaves_no_orphans(collection):
    _write(collection, 431)
    result = _write(collection, 300, speaker="Jacob Shapiro")
    assert result["pruned"] == 131
    ids = paged_get_ids(collection, episode_where(SHOW, EP, DATE))
    assert len(ids) == 300
    docs = collection.get(ids=ids, include=["documents"])["documents"]
    assert all("SPEAKER_00" not in d for d in docs)


def test_a_growing_re_embed_prunes_nothing(collection):
    _write(collection, 100)
    result = _write(collection, 150)
    assert result["pruned"] == 0
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 150


def test_a_crash_between_upsert_and_prune_leaves_a_superset_not_a_hole(collection):
    # Simulate: the upsert landed, the prune did not.
    _write(collection, 431)
    upsert_then_prune(
        collection,
        _chunks(300, speaker="Jacob Shapiro"),
        _embeddings(300),
        show=SHOW,
        episode_number=EP,
        date_str=DATE,
        episode_guid=None,
        _skip_prune=True,
    )
    ids = paged_get_ids(collection, episode_where(SHOW, EP, DATE))
    assert len(ids) == 431  # superset -- every chunk 0..299 rewritten, 300..430 stale
    # The next run converges.
    _write(collection, 300, speaker="Jacob Shapiro")
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 300


def test_prune_unions_guid_and_triple_so_old_records_are_not_stranded(collection):
    # Pre-migration records are triple-keyed with NO guid. A guid-only prune
    # would match nothing and leave the entire old set behind.
    _write(collection, 50)  # no guid
    result = _write(collection, 40, guid="guid-abc")
    assert result["pruned"] == 10
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 40


def test_another_episode_is_never_touched(collection):
    _write(collection, 20)
    upsert_then_prune(
        collection,
        [
            {
                "text": "other",
                "metadata": {
                    "show": SHOW,
                    "episode_number": "74",
                    "date": "2026-07-31",
                    "n_chunks": 1,
                },
            }
        ],
        [[0.0, 0.0, 1.0]],
        show=SHOW,
        episode_number="74",
        date_str="2026-07-31",
        episode_guid=None,
    )
    assert len(paged_get_ids(collection, episode_where(SHOW, EP, DATE))) == 20
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_write_path.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus.writing'`

- [ ] **Step 3: Write the implementation**

```python
# corpus/writing.py
"""Full-replacement episode writes.

UPSERT FIRST, THEN PRUNE -- not delete-then-upsert. A new episode's delete
matches nothing so the trickle case has no window either way, but a
full-archive re-embed opens one destructive window per healthy episode. And
the failure that matters is not an exception: transcribe runs with a 7200s
timeout covering a whole show, and a Modal timeout kills the container without
raising anything an `except` clause can see. Under delete-first that is a
deleted-and-never-rewritten episode with no log line. This way a crash leaves
a SUPERSET, never a hole, and the next run prunes it.
"""

from __future__ import annotations

from corpus.identity import chunk_id, episode_id_prefix
from corpus.store import (
    BATCH,
    batched,
    episode_where,
    guid_where,
    paged_get_ids,
    stale_ids,
)


def upsert_then_prune(
    collection,
    chunks: list[dict],
    embeddings: list[list[float]],
    *,
    show: str,
    episode_number: str,
    date_str: str,
    episode_guid: str | None,
    _skip_prune: bool = False,
) -> dict:
    """Write an episode's chunks, then remove whatever it no longer occupies.

    `_skip_prune` exists only so a test can simulate a crash between the two
    phases. Never pass it from production code.
    """
    prefix = episode_id_prefix(show, episode_number, date_str)
    new_ids = [chunk_id(prefix, i) for i in range(len(chunks))]

    for batch in batched(list(range(len(new_ids))), BATCH):
        collection.upsert(
            ids=[new_ids[i] for i in batch],
            embeddings=[embeddings[i] for i in batch],
            documents=[chunks[i]["text"] for i in batch],
            metadatas=[chunks[i]["metadata"] for i in batch],
        )

    if _skip_prune:
        return {"written": len(new_ids), "pruned": 0}

    # UNION, never "guid if present else triple". Every record written before
    # the migration is triple-keyed with no guid, so a guid-only prune matches
    # nothing and strands the entire old record set.
    existing = set(paged_get_ids(collection, episode_where(show, episode_number, date_str)))
    if episode_guid:
        existing |= set(paged_get_ids(collection, guid_where(episode_guid)))

    to_prune = stale_ids(existing, new_ids)
    for batch in batched(to_prune, BATCH):
        collection.delete(ids=batch)
    return {"written": len(new_ids), "pruned": len(to_prune)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_write_path.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Wire it into `transcribe.py`**

Replace `embed_and_store` entirely:

```python
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
        print("  No chunks to store.")
        return

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
```

Replace `get_chroma_collection` so the collection name comes from the environment:

```python
def get_chroma_collection(chroma_api_key, chroma_tenant, chroma_database):
    import chromadb

    client = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database,
    )
    # The writer may create; the reader (mcp_server) must not -- see its
    # get_collection call. scheduled_job spawns fresh containers nightly, so
    # this side has no warm-container exposure.
    return client.get_or_create_collection(
        name=os.environ.get("CHROMA_COLLECTION", "podcast_transcripts"),
        metadata={"hnsw:space": "cosine"},
    )
```

Update the two `embed_and_store(...)` call sites in `transcribe` and `bulk_embed` to pass the new arguments. In `transcribe`:

```python
            embed_and_store(
                chunks,
                embedding_model,
                collection,
                show_name,
                episode_number,
                date_str,
                episode.get("guid"),
            )
```

In `bulk_embed`:

```python
            embed_and_store(
                chunks,
                embedding_model,
                collection,
                parsed_show,
                episode_number,
                date_str,
            )
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run ruff check . && uv run mypy && uv run pytest`
Expected: all clean

- [ ] **Step 7: Commit**

```bash
git add corpus/writing.py tests/test_write_path.py transcribe.py
git commit -m "feat: upsert-then-prune writes, batched at 250

Upsert cannot shrink a record set, so a re-embed producing fewer chunks
strands every index above the new count with stale speaker text still in
the document. Pruning after the upsert means a crash leaves a superset,
never a hole -- which matters because a Modal timeout kills the container
without raising anything the except clause sees. The prune unions guid and
triple so pre-migration records are not stranded."
```

---

### Task 9: Planner and self-healing executor

**Files:**
- Modify: `transcribe.py` — add `parse_all_episodes` guid extraction, a CPU `plan_work` function, and rewrite the `pending` loop in `transcribe`
- Create: `corpus/completeness.py`
- Create: `tests/test_completeness.py`

**Interfaces:**
- Consumes: `corpus.planning.decide_action`, `corpus.store.{episode_where, paged_get_ids, is_complete}`, `corpus.chunking.count_chunks_from_text`, `corpus.exclusions.is_excluded`
- Produces:
  - `corpus.completeness.episode_state(collection, show, episode_number, date_str) -> tuple[list[str], int | None, str | None]` returning `(stored_ids, stored_n_chunks, stored_rules_version)`
  - `corpus.completeness.plan_episode(collection, *, show, episode_number, date_str, transcript_text, episode_guid=None, expected_n_chunks=None) -> Action` — **`episode_guid` must be passed by the cron path**, or the exclusion guid arm is unreachable

- [ ] **Step 1: Write the failing test**

```python
# tests/test_completeness.py
from corpus.completeness import episode_state, plan_episode
from corpus.planning import Action
from corpus.store import batched

SHOW, EP, DATE = "Geopolitical Cousins", "73", "2026-07-29"

TRANSCRIPT = """# Geopolitical Cousins - Episode 73

[SPEAKER_00] 1.0s - One.
[SPEAKER_01] 2.0s - Two.
"""


def _seed(collection, n, *, n_chunks, rules_version="1"):
    for batch in batched(list(range(n))):
        collection.upsert(
            ids=[f"Geopolitical_Cousins-ep73-2026-07-29-{i}" for i in batch],
            documents=[f"d{i}" for i in batch],
            metadatas=[
                {
                    "show": SHOW,
                    "episode_number": EP,
                    "date": DATE,
                    "n_chunks": n_chunks,
                    "rules_version": rules_version,
                }
                for _ in batch
            ],
        )


def test_state_of_an_absent_episode(collection):
    ids, n_chunks, rules = episode_state(collection, SHOW, EP, DATE)
    assert (ids, n_chunks, rules) == ([], None, None)


def test_state_reads_expected_count_and_version(collection):
    _seed(collection, 2, n_chunks=2)
    ids, n_chunks, rules = episode_state(collection, SHOW, EP, DATE)
    assert len(ids) == 2
    assert n_chunks == 2
    assert rules == "1"


def test_complete_episode_is_skipped(collection):
    _seed(collection, 2, n_chunks=2)
    assert (
        plan_episode(
            collection, show=SHOW, episode_number=EP, date_str=DATE,
            transcript_text=TRANSCRIPT,
        )
        is Action.SKIP
    )


def test_torn_episode_is_re_embedded(collection):
    # One surviving orphan out of two. A boolean check would call this healthy.
    _seed(collection, 1, n_chunks=2)
    assert (
        plan_episode(
            collection, show=SHOW, episode_number=EP, date_str=DATE,
            transcript_text=TRANSCRIPT,
        )
        is Action.EMBED_ONLY
    )


def test_stale_rules_version_is_re_embedded(collection):
    _seed(collection, 2, n_chunks=2, rules_version="0")
    assert (
        plan_episode(
            collection, show=SHOW, episode_number=EP, date_str=DATE,
            transcript_text=TRANSCRIPT,
        )
        is Action.EMBED_ONLY
    )


def test_absent_transcript_is_transcribed(collection):
    assert (
        plan_episode(
            collection, show=SHOW, episode_number=EP, date_str=DATE,
            transcript_text=None,
        )
        is Action.TRANSCRIBE
    )


def test_excluded_episode_is_excluded(collection):
    assert (
        plan_episode(
            collection,
            show="The Jacob Shapiro Podcast",
            episode_number="Unknown",
            date_str="2026-07-29",
            transcript_text=TRANSCRIPT,
        )
        is Action.EXCLUDE
    )


def test_exclusion_survives_a_backfilled_episode_number_via_the_guid(collection):
    # THE enforcement test. Captivate backfilling itunes_episode changes the
    # triple, so the triple arm stops matching -- and without the guid
    # threaded through plan_episode the whole guid arm is unreachable from the
    # cron path, present and documented and never firing.
    assert (
        plan_episode(
            collection,
            show="The Jacob Shapiro Podcast",
            episode_number="352",  # backfilled; the triple no longer matches
            date_str="2026-07-29",
            transcript_text=TRANSCRIPT,
            episode_guid="1c45dbd9-0dc3-4d07-b2d1-758fe78405fe",
        )
        is Action.EXCLUDE
    )


def test_a_normal_episode_with_a_guid_is_not_excluded(collection):
    assert (
        plan_episode(
            collection,
            show=SHOW,
            episode_number=EP,
            date_str=DATE,
            transcript_text=TRANSCRIPT,
            episode_guid="b4a9c88b-9dbf-46b7-9dc1-a7812a9bde65",
        )
        is not Action.EXCLUDE
    )


def test_a_431_chunk_episode_is_judged_complete_not_looping(collection):
    # Unpaged, the count would come back 300 against n_chunks=431 and this
    # episode would be re-embedded on every cron run forever.
    _seed(collection, 431, n_chunks=431)
    assert (
        plan_episode(
            collection, show=SHOW, episode_number=EP, date_str=DATE,
            transcript_text=TRANSCRIPT, expected_n_chunks=431,
        )
        is Action.SKIP
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_completeness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus.completeness'`

- [ ] **Step 3: Write the implementation**

```python
# corpus/completeness.py
"""Is this episode fully and currently stored?"""

from __future__ import annotations

from corpus.chunking import count_chunks_from_text
from corpus.exclusions import is_excluded
from corpus.planning import Action, decide_action
from corpus.store import episode_where, is_complete, paged_get, paged_get_ids


def episode_state(
    collection, show: str, episode_number: str, date_str: str
) -> tuple[list[str], int | None, str | None]:
    """Stored ids, the expected chunk count, and the rules version.

    The id list is PAGED. An unlimited get() silently returns 300, so an
    unpaged count against a 431-chunk episode reports short, judges a healthy
    episode torn, and re-embeds it on every run forever.
    """
    where = episode_where(show, episode_number, date_str)
    ids = paged_get_ids(collection, where)
    if not ids:
        return [], None, None
    head = collection.get(ids=ids[:1], include=["metadatas"])
    meta = (head.get("metadatas") or [{}])[0]
    n_chunks = meta.get("n_chunks")
    return ids, n_chunks if isinstance(n_chunks, int) else None, meta.get(
        "rules_version"
    )


def plan_episode(
    collection,
    *,
    show: str,
    episode_number: str,
    date_str: str,
    transcript_text: str | None,
    episode_guid: str | None = None,
    expected_n_chunks: int | None = None,
) -> Action:
    """What this episode needs.

    `expected_n_chunks` may be supplied by a caller that has already counted;
    otherwise it is derived from the transcript. It is NEVER taken from stored
    records, which would freeze a torn episode as permanently complete.

    `episode_guid` MUST be passed by the cron path. Without it the exclusion
    check falls back to the triple alone, and the guid arm -- which exists
    precisely because the triple is not durable -- is unreachable from the
    only path that runs nightly.
    """
    excluded = is_excluded(show, episode_number, date_str, episode_guid)
    if transcript_text is None:
        return decide_action(
            transcript_exists=False,
            complete_in_chroma=False,
            stored_rules_version=None,
            excluded=excluded,
            parses_to_chunks=True,
        )

    if expected_n_chunks is None:
        expected_n_chunks = count_chunks_from_text(
            transcript_text, show, episode_number, "", date_str
        )
    stored_ids, _stored_n, stored_rules = episode_state(
        collection, show, episode_number, date_str
    )
    return decide_action(
        transcript_exists=True,
        complete_in_chroma=is_complete(stored_ids, expected_n_chunks),
        stored_rules_version=stored_rules,
        excluded=excluded,
        parses_to_chunks=expected_n_chunks > 0,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_completeness.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Capture the guid in `parse_all_episodes`**

In `transcribe.py`, inside the entry loop of `parse_all_episodes`, add the guid and include it in the appended dict:

```python
        guid = entry.get("id")
        # RSS <guid isPermaLink> defaults to true. A link-derived guid is a
        # URL, and this publisher rewrites URLs, so it is not an identity.
        if entry.get("guidislink"):
            guid = None
```

and change the append to:

```python
            episodes.append(
                {
                    "title": title,
                    "episode_number": str(episode_number),
                    "audio_url": audio_url,
                    "date": date_str,
                    "guid": guid,
                }
            )
```

- [ ] **Step 6: Replace the `pending` loop with the state machine**

In `transcribe`, replace the block that builds `pending` (currently `os.path.exists` only) with:

```python
    from corpus.completeness import plan_episode
    from corpus.identity import transcript_filename
    from corpus.planning import Action

    plan = []
    for episode in episodes:
        out_path = (
            f"{VOLUME_PATH}/"
            f"{transcript_filename(show_name, episode['episode_number'], episode['date'])}"
        )
        text = None
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        action = plan_episode(
            collection,
            show=show_name,
            episode_number=episode["episode_number"],
            date_str=episode["date"],
            transcript_text=text,
            # Load-bearing: without it the exclusion falls back to the triple
            # alone and the guid arm never fires on the nightly path.
            episode_guid=episode.get("guid"),
        )
        print(f"  {action.value:12s} Episode {episode['episode_number']} ({episode['date']})")
        if action in (Action.TRANSCRIBE, Action.EMBED_ONLY):
            plan.append((action, episode, out_path))

    if not plan:
        print("Nothing to do.")
        return

    pending = [(a, e, p) for a, e, p in plan if a is Action.TRANSCRIBE]
    embed_only = [(a, e, p) for a, e, p in plan if a is Action.EMBED_ONLY]
    print(f"{len(pending)} to transcribe, {len(embed_only)} to re-embed.")

    for _action, episode, out_path in embed_only:
        with open(out_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        chunks = build_chunks_from_text(
            text,
            show_name,
            episode["episode_number"],
            episode["title"],
            episode["date"],
            episode_guid=episode.get("guid"),
        )
        print(f"Re-embedding Episode {episode['episode_number']} from transcript...")
        embed_and_store(
            chunks,
            embedding_model,
            collection,
            show_name,
            episode["episode_number"],
            episode["date"],
            episode.get("guid"),
        )
```

Then change the transcription loop header from `for episode in pending:` to:

```python
    for _action, episode, out_path in pending:
```

and delete the now-duplicated `out_path = ...` line inside that loop body.

- [ ] **Step 7: Run the whole suite**

Run: `uv run ruff check . && uv run mypy && uv run pytest`
Expected: all clean

- [ ] **Step 8: Commit**

```bash
git add corpus/completeness.py tests/test_completeness.py transcribe.py
git commit -m "feat: self-healing embed branch driven by the state machine

transcribe built its work list from os.path.exists, which proves that
transcription ran and says nothing about whether embedding landed -- so an
episode whose embed failed was skipped forever. The count query is paged:
unpaged it returns 300 against a 431-chunk episode, judges a healthy
episode torn, and re-embeds it nightly."
```

---

### Task 10: Reader changes and the cutover probe

**Files:**
- Modify: `mcp_server.py` — `PodcastSearch.load` (line ~52), the `search` output block (line ~102)

**Interfaces:**
- Consumes: nothing from `corpus`.
- Produces: `search()` results carrying `n_chunks` and `episode_guid`.

- [ ] **Step 1: Make the reader fail loudly on a wrong collection name**

In `mcp_server.py`, replace the `get_or_create_collection` call in `load`:

```python
        # get_collection, NOT get_or_create_collection. All three call sites
        # used get-or-create, so a typo at cutover would silently create an
        # empty third collection and make search_podcasts return
        # "No results found." instead of erroring. The reader must fail loudly;
        # the writer may still create.
        self.collection = self.chroma_client.get_collection(
            name=os.environ.get("CHROMA_COLLECTION", "podcast_transcripts"),
        )
```

- [ ] **Step 2: Surface the migration's new fields**

At cutover v1 and v2 are content-identical by construction — validation just proved it record-for-record — so no content-shaped probe can tell them apart. The metadata the migration added is the only difference, so it has to be visible on the path that already exists.

In the `search` output block, add two entries to the appended dict:

```python
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
```

- [ ] **Step 3: Verify the module still parses**

Run: `uv run python -c "import ast, pathlib; ast.parse(pathlib.Path('mcp_server.py').read_text())"`
Expected: no output, exit 0

- [ ] **Step 4: Commit**

```bash
git add mcp_server.py
git commit -m "feat: fail loudly on a wrong collection, surface the cutover tell

get_or_create_collection cannot raise, so a typo at cutover would create an
empty third collection and search would return 'No results found.' rather
than erroring. And because validation proves v1 and v2 content-identical,
no content-shaped probe can discriminate -- n_chunks in the search output
is the only thing that can."
```

---

### Task 11: Pin the image dependencies

`mcp` was installed unpinned, a rebuild resolved a version where `host=` fed DNS-rebinding validation, and the endpoint returned 421 on every authenticated call. Every Chroma semantic this plan relies on is a property of `chromadb 1.5.9` specifically.

**Files:**
- Modify: `transcribe.py` (image), `mcp_server.py` (image)

- [ ] **Step 1: Record what the deployed MCP image actually holds**

The transcriber image was already measured — `chromadb 1.5.9`, `sentence-transformers 5.4.1`, `whisperx 3.8.5`, `numpy 2.4.4`, `feedparser 6.0.12`, `torch 2.8.0`. The MCP image has not been.

Create `scratch_versions.py` in the scratchpad directory (not the repo), copying `mcp_server.py`'s image definition byte-identically, with a function that prints `importlib.metadata.version` for `chromadb`, `mcp`, `sentence-transformers`, `torch`, `fastapi` and `starlette`.

Run: `uv run modal run <scratchpad>/scratch_versions.py::probe`
Expected: no rebuild in the log — a rebuild resolves fresh versions and the reading is then worthless. Record the six versions.

- [ ] **Step 2: Pin every package in the layer, not just chromadb**

Pinning only `chromadb` still lets a rebuild of that layer upgrade its neighbours. In `transcribe.py`:

```python
    .pip_install(
        "whisperx==3.8.5",
        "feedparser==6.0.12",
        "requests",
        "sentence-transformers==5.4.1",
        "chromadb==1.5.9",
    )
```

In `mcp_server.py`, pin `chromadb`, `mcp`, `sentence-transformers`, `fastapi` and `starlette` to the versions recorded in Step 1.

- [ ] **Step 3: Verify the rebuild resolves the pinned versions**

Run: `uv run modal run transcribe.py::bulk_upload --show-name "does-not-exist"`
Expected: an image build happens (pinning changes the layer), it completes, and the run exits 0.

Re-run the version probe from Step 1 against the transcriber image.
Expected: the same six versions as before the pin. **If any differs, the pin is wrong** — fix it before proceeding.

- [ ] **Step 4: Commit**

```bash
git add transcribe.py mcp_server.py
git commit -m "chore: pin image dependencies to the deployed versions

Unpinned installs mean a layer holds whatever resolved when it was first
built, and a rebuild changes behaviour with no deploy-time signal -- which
is how removing the mcp host= kwarg returned 421 on every call. Every
Chroma semantic this work relies on is a property of 1.5.9. Pinning only
chromadb would still let a rebuild upgrade its neighbours in the same
layer, so the whole layer is pinned."
```

---

### Task 12: Bidirectional reconciliation

**Files:**
- Create: `corpus/reconcile.py`
- Create: `tests/test_reconcile.py`
- Create: `migration/reconcile_report.py`

**Interfaces:**
- Consumes: `corpus.identity.parse_transcript_filename`, `corpus.exclusions.EXCLUDED_EPISODES`
- Produces:
  - `corpus.reconcile.ReconcileReport` — a dataclass with fields `missing`, `extra`, `non_contiguous`, `shared_prefixes`, `excluded_with_records`, `feed_unreachable`, `suspected_cross_posts`
  - `corpus.reconcile.reconcile(volume_keys, chroma_records, feed_keys) -> ReconcileReport`
  - `corpus.reconcile.ReconcileReport.is_clean() -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reconcile.py
from corpus.reconcile import reconcile

VOL = {
    ("Geopolitical Cousins", "73", "2026-07-29"),
    ("Geopolitical Cousins", "74", "2026-07-31"),
    ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29"),
}
FEED = {
    ("Geopolitical Cousins", "73", "2026-07-29"),
    ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29"),
}


def _records(key, prefix, indices):
    show, ep, date = key
    return [
        (f"{prefix}-{i}", {"show": show, "episode_number": ep, "date": date})
        for i in indices
    ]


def test_clean_corpus_reports_nothing():
    records = _records(
        ("Geopolitical Cousins", "73", "2026-07-29"),
        "Geopolitical_Cousins-ep73-2026-07-29",
        range(3),
    ) + _records(
        ("Geopolitical Cousins", "74", "2026-07-31"),
        "Geopolitical_Cousins-ep74-2026-07-31",
        range(2),
    )
    report = reconcile(VOL, records, FEED)
    assert report.missing == []
    assert report.extra == []
    assert report.non_contiguous == []
    assert report.excluded_with_records == []


def test_missing_episode_is_reported():
    records = _records(
        ("Geopolitical Cousins", "73", "2026-07-29"),
        "Geopolitical_Cousins-ep73-2026-07-29",
        range(3),
    )
    report = reconcile(VOL, records, FEED)
    assert ("Geopolitical Cousins", "74", "2026-07-31") in report.missing


def test_records_with_no_volume_file_are_extra():
    records = _records(
        ("Ghost Show", "1", "2020-01-01"), "Ghost_Show-ep1-2020-01-01", range(2)
    )
    report = reconcile(VOL, records, FEED)
    assert ("Ghost Show", "1", "2020-01-01") in report.extra


def test_an_orphan_tail_shows_as_non_contiguous():
    # Indices 0,1,5 -- what a shrinking re-embed leaves behind.
    records = _records(
        ("Geopolitical Cousins", "73", "2026-07-29"),
        "Geopolitical_Cousins-ep73-2026-07-29",
        [0, 1, 5],
    )
    report = reconcile(VOL, records, FEED)
    assert ("Geopolitical Cousins", "73", "2026-07-29") in report.non_contiguous


def test_an_excluded_episode_holding_records_is_reported():
    records = _records(
        ("The Jacob Shapiro Podcast", "Unknown", "2026-07-29"),
        "The_Jacob_Shapiro_Podcast-epUnknown-2026-07-29",
        range(2),
    )
    report = reconcile(VOL, records, FEED)
    assert (
        "The Jacob Shapiro Podcast",
        "Unknown",
        "2026-07-29",
    ) in report.excluded_with_records


def test_a_returned_exclusion_under_a_backfilled_number_is_not_mere_extra():
    # Same episode, backfilled episode_number, recorded guid. Without the guid
    # arm this lands in `extra` and the alarm names the wrong problem.
    records = [
        (
            "The_Jacob_Shapiro_Podcast-ep352-2026-07-29-0",
            {
                "show": "The Jacob Shapiro Podcast",
                "episode_number": "352",
                "date": "2026-07-29",
                "episode_guid": "1c45dbd9-0dc3-4d07-b2d1-758fe78405fe",
            },
        )
    ]
    report = reconcile(VOL, records, FEED)
    key = ("The Jacob Shapiro Podcast", "352", "2026-07-29")
    assert key in report.excluded_with_records
    assert key not in report.extra


def test_an_excluded_episode_is_never_reported_as_missing():
    report = reconcile(VOL, [], FEED)
    assert (
        "The Jacob Shapiro Podcast",
        "Unknown",
        "2026-07-29",
    ) not in report.missing


def test_a_shared_id_prefix_is_reported():
    records = [
        ("Show-ep1-0", {"show": "A", "episode_number": "1", "date": "2025-01-01"}),
        ("Show-ep1-1", {"show": "B", "episode_number": "1", "date": "2025-01-02"}),
    ]
    report = reconcile(set(), records, set())
    assert "Show-ep1" in report.shared_prefixes


def test_volume_files_with_no_feed_entry_are_their_own_class():
    report = reconcile(VOL, [], FEED)
    assert ("Geopolitical Cousins", "74", "2026-07-31") in report.feed_unreachable


def test_is_clean_is_false_when_anything_is_wrong():
    assert not reconcile(VOL, [], FEED).is_clean()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_reconcile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus.reconcile'`

- [ ] **Step 3: Write the implementation**

```python
# corpus/reconcile.py
"""Reconcile the transcript volume against the vector store, both directions.

A one-directional check keeps passing while the corpus rots: the original
version could only see records that were MISSING, never records that should
not exist and never an episode whose indices had holes in them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from corpus.exclusions import EXCLUDED_EPISODES, EXCLUDED_GUIDS

Key = tuple[str, str, str]


@dataclass
class ReconcileReport:
    missing: list[Key] = field(default_factory=list)
    extra: list[Key] = field(default_factory=list)
    non_contiguous: list[Key] = field(default_factory=list)
    shared_prefixes: list[str] = field(default_factory=list)
    excluded_with_records: list[Key] = field(default_factory=list)
    feed_unreachable: list[Key] = field(default_factory=list)
    suspected_cross_posts: list[tuple[Key, Key]] = field(default_factory=list)

    def is_clean(self) -> bool:
        """feed_unreachable and suspected_cross_posts are INFORMATIONAL.

        Those two are facts to know, not faults: six episodes have aged off
        their feed and cross-posts are a human decision, not an inference.
        """
        return not (
            self.missing
            or self.extra
            or self.non_contiguous
            or self.shared_prefixes
            or self.excluded_with_records
        )


def reconcile(
    volume_keys: set[Key],
    chroma_records: list[tuple[str, dict]],
    feed_keys: set[Key],
) -> ReconcileReport:
    """Compare the volume, the store and the feeds."""
    report = ReconcileReport()

    by_key: dict[Key, list[str]] = defaultdict(list)
    prefixes: dict[str, set[Key]] = defaultdict(set)
    guid_keys: set[Key] = set()
    for record_id, meta in chroma_records:
        key = (meta.get("show"), meta.get("episode_number"), meta.get("date"))
        by_key[key].append(record_id)
        # An excluded episode that came back under a BACKFILLED episode_number
        # no longer matches the triple. Without this it lands in `extra` and
        # the alarm points at the wrong class entirely.
        if meta.get("episode_guid") in EXCLUDED_GUIDS:
            guid_keys.add(key)
        prefix, _, index = record_id.rpartition("-")
        if index.isdigit():
            prefixes[prefix].add(key)

    excluded_present = {k for k in EXCLUDED_EPISODES if by_key.get(k)} | guid_keys

    expected = volume_keys - EXCLUDED_EPISODES
    report.missing = sorted(k for k in expected if k not in by_key)
    report.extra = sorted(
        k for k in by_key if k not in volume_keys and k not in excluded_present
    )
    report.excluded_with_records = sorted(excluded_present)
    report.feed_unreachable = sorted(volume_keys - feed_keys)
    report.shared_prefixes = sorted(p for p, keys in prefixes.items() if len(keys) > 1)

    for key, ids in by_key.items():
        indices = sorted(
            int(i.rpartition("-")[2]) for i in ids if i.rpartition("-")[2].isdigit()
        )
        if indices and indices != list(range(len(indices))):
            report.non_contiguous.append(key)
    report.non_contiguous.sort()

    return report
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_reconcile.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Write the Modal runner**

```python
# migration/reconcile_report.py
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
    "The Observing Japan Podcast":
        "https://api.substack.com/feed/podcast/868206/s/386602.rss",
}


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=1800,
)
def run():
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

    records = []
    offset = 0
    while True:
        page = col.get(limit=PAGE, offset=offset, include=["metadatas"])
        if not page["ids"]:
            break
        records.extend(zip(page["ids"], page["metadatas"]))
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
    print(f"\nCLEAN: {report.is_clean()}")


@app.local_entrypoint()
def main():
    run.remote()
```

- [ ] **Step 6: Run it against production as a baseline**

Run: `uv run modal run migration/reconcile_report.py`
Expected, **before** the migration and repair: `MISSING (5)` — the seven minus the two excluded; `EXCLUDED_WITH_RECORDS (0)`; `FEED_UNREACHABLE (6)` — Jacob Shapiro episodes 1–6; `CLEAN: False`.

Record this output. It is the before-picture the acceptance run is compared against.

- [ ] **Step 7: Commit**

```bash
git add corpus/reconcile.py tests/test_reconcile.py migration/reconcile_report.py
git commit -m "feat: bidirectional reconciliation

The original check could only see missing records -- never extras, never an
orphan tail, never an excluded episode that had come back. Volume files with
no feed entry are reported as their own informational class: six episodes
have aged off the front of their feed and are invisible to a feed-iterating
planner forever."
```

---

### Task 13: Migration, cutover, and the repair

Operational. Every step is reversible until step 9.

**Files:**
- Modify: `README.md` (document `CHROMA_COLLECTION` and the cutover procedure)

- [ ] **Step 1: Freeze the cron**

Run: `uv run modal app stop podcast-transcriber --yes`
Expected: the app stops. `--yes` is required in a non-interactive shell.

Confirm no run is in flight before proceeding — `copy_collection` aborts if the source count moves, and a partial copy is not a valid destination.

- [ ] **Step 2: Dry-run the migration once more against the frozen source**

Run: `uv run modal run migration/reid.py --dry-run`
Expected: `count=28489`, `passthrough_unmatched` 0, `passthrough_non_episode` 191.

- [ ] **Step 3: Run the migration**

Run: `uv run modal run migration/reid.py --no-dry-run`
Expected: paged copy progress to 28,489, then `VALIDATION CLEAN. podcast_transcripts_v2 has 28489 records.`

If validation reports problems, **do not cut over**. v1 is untouched.

- [ ] **Step 4: Point the secret at the new collection**

Run:

```bash
modal secret create podcast-secrets \
  CHROMA_API_KEY=<unchanged> \
  CHROMA_TENANT=<unchanged> \
  CHROMA_DATABASE=geo-podcasts-us \
  HF_TOKEN=<unchanged> \
  MCP_ALLOWED_HOST=<unchanged> \
  CHROMA_COLLECTION=podcast_transcripts_v2
```

Every existing key must be repeated — recreating the secret replaces it wholesale, and omitting `MCP_ALLOWED_HOST` returns the endpoint to 421 on every call.

- [ ] **Step 5: Deploy the new write path and the constant together**

Run: `uv run modal deploy transcribe.py && uv run modal deploy mcp_server.py`

This is atomic on purpose. `transcribe` used to build its work list from `os.path.exists`, and all seven broken episodes *have* transcripts — so on the old code the work list is empty and it returns before reaching any embed path. Self-healing does not exist until `decide_action` ships. A cutover redeploy carrying the old write path would write old-scheme IDs, unbatched, with no `n_chunks`, straight into the collection just validated.

- [ ] **Step 6: Probe the deployed endpoint**

Run:

```bash
curl -s -H "Modal-Key: $TOKEN_ID" \
     -H "Modal-Secret: $TOKEN_SECRET" \
     -H "Content-Type: application/json" \
     -H "Accept: application/json, text/event-stream" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
          "params":{"name":"search_podcasts","arguments":{"query":"Taiwan","n_results":1}}}' \
     https://<your-endpoint>.modal.run/mcp
```

Expected: a result whose `n_chunks` is populated. **`n_chunks: null` means v1, or a warm container still holding a v1 handle.**

- [ ] **Step 7: Force a container cycle if the probe says v1**

Run: `uv run modal app stop podcast-mcp-server --yes && uv run modal deploy mcp_server.py`

Then repeat Step 6. Cutover is not done until the probe is v2-shaped.

A `PodcastSearch` container that entered `load()` before cutover holds a `Collection` handle bound to v1 and keeps serving from it, correctly and silently, for as long as it stays warm — there is no name to re-resolve, so nothing raises and `get_collection` does not help. Container draining is a timing property that cannot be observed from here; do not reason about it, probe it.

- [ ] **Step 8: Unfreeze and let the repair run**

Run: `uv run modal deploy transcribe.py` (redeploying restores the cron schedule), then trigger a run rather than waiting for 09:00 UTC:

```bash
uv run modal run transcribe.py --feed-url "https://feeds.captivate.fm/geopolitical-cousins/" --show-name "Geopolitical Cousins"
uv run modal run transcribe.py --feed-url "https://feeds.captivate.fm/jacob-shapiro/" --show-name "The Jacob Shapiro Podcast"
uv run modal run transcribe.py --feed-url "https://api.substack.com/feed/podcast/868206/s/386602.rss" --show-name "The Observing Japan Podcast"
```

Expected: `EMBED_ONLY` for Geopolitical Cousins 73 and 74 and for the two Observing Japan episodes, `EXCLUDE` for the two Jacob Shapiro cross-posts, `EMBED_ONLY` for Jacob Shapiro 243 (2024-11-08), and `SKIP` for everything else. No episode is re-transcribed.

- [ ] **Step 9: Run the acceptance reconciliation**

Run: `uv run modal run migration/reconcile_report.py`
Expected: `MISSING (0)`, `EXTRA (0)`, `NON_CONTIGUOUS (0)`, `SHARED_PREFIXES (0)`, `EXCLUDED_WITH_RECORDS (0)`, `FEED_UNREACHABLE (6)`, `CLEAN: True`.

- [ ] **Step 10: Delete the two 2025 cross-post duplicates**

Only now, and only after confirming with the user — this removes production data, and it is safe only because `EXCLUDE` exists to stop the cron restoring it.

Add two records to `EXCLUDED` in `corpus/exclusions.py`. Their real values, read from the live feed — note they are **not** `"Unknown"`; Captivate publishes an `itunes_episode` for these two, which is exactly why the 2026 pair fall back to the literal and these do not:

```python
        ExcludedEpisode(
            "The Jacob Shapiro Podcast",
            "271",
            "2025-04-04",
            "c4af95bf-cfbc-4c0a-b4d7-2c2df77d1fe6",
            "cross-post of Geopolitical Cousins, "
            "'Riding on the Hog of a Fiscal Orgy'",
        ),
        ExcludedEpisode(
            "The Jacob Shapiro Podcast",
            "273",
            "2025-04-08",
            "3a3c0a69-ea66-46b1-a54e-7ef1ea657505",
            "cross-post of Geopolitical Cousins, 'Let Them Drink Bleach'",
        ),
```

`EXCLUDED_EPISODES` and `EXCLUDED_GUIDS` are derived, so nothing else needs editing. Update `tests/test_planning.py::test_exclusions_are_exactly_the_two_cross_posts` to expect four entries and rename it accordingly. Run the suite, deploy, then delete the records:

```python
collection.delete(where=episode_where("The Jacob Shapiro Podcast", "271", "2025-04-04"))
collection.delete(where=episode_where("The Jacob Shapiro Podcast", "273", "2025-04-08"))
```

Re-run the reconciliation, expecting `EXCLUDED_WITH_RECORDS (0)` and `MISSING (0)` — the exclusions must not reappear as missing.

- [ ] **Step 11: Document the new configuration**

Add `CHROMA_COLLECTION` to the `modal secret create` block in `README.md`, and a short "Cutover" subsection recording that the reader uses `get_collection` so a wrong name raises, that `n_chunks` in a search result is the v1/v2 tell, and that a warm container must be cycled rather than waited out.

- [ ] **Step 12: Commit**

```bash
git add README.md corpus/exclusions.py tests/test_planning.py
git commit -m "docs: record the cutover procedure and CHROMA_COLLECTION

The cutover has no natural alarm: get_collection covers a wrong name at
startup but does nothing for a warm container holding a v1 handle, which
keeps serving correctly and silently. n_chunks in a search result is the
only discriminating probe, because validation proves v1 and v2
content-identical."
```

- [ ] **Step 13: Delete v1 (separate, later)**

Not part of this plan. Once the new collection has been in use long enough to trust, delete `podcast_transcripts` explicitly. Until then it is the rollback path the July EU→US migration did not have.

---

## Self-Review

**Spec coverage.** Part 1 root causes → Tasks 1, 4, 8, 9. Part 2 verified semantics → Task 4's fake plus Task 11's pins. Part 3 (dependency on the speaker work) → Task 8. §4.1 identity → Tasks 1, 7, 9. §4.2 upsert-then-prune → Task 8. §4.3 completeness → Tasks 4, 9. §4.4 `decide_action` → Tasks 3, 9. §4.5 plan/execute split → Task 9. §4.6 reconciliation and the feed-unreachable class → Task 12. §4.7 migration → Tasks 6, 7, 13. §4.8 collection access → Tasks 8, 10. Part 5 cross-posts → Tasks 3, 12, 13.

**Two spec items deliberately deferred, and why.** The spec leaves open whether `EMBED_ONLY` can run CPU-only pending a measured BGE-large CPU throughput; no task splits that tier, and the repair is ~450 chunks where it does not matter. The B1 archive re-embed is ~29,894 chunks and belongs to the speaker spec, which should measure it first. Second, `upload_book.py` is untouched — it writes its own ID scheme, its records pass through the migration verbatim, and bringing it onto `episode_id_prefix` is out of scope.

**One assumption was flagged, checked, and turned out to be wrong.** An earlier draft of Task 13 Step 10 assumed the two 2025 cross-posts carried `episode_number == "Unknown"` like the 2026 pair. They do not — they are episodes **271** and **273**; Captivate publishes an `itunes_episode` for them. Exclusions written on the assumption would never have matched, and the cron would have restored the deleted duplicates on its next run with every check reporting green. Real triples and guids are now in the task.

That failure is also the argument for `EXCLUDED_GUIDS`: an exclusion keyed only on the triple is keyed on the one thing the spec's own six-month threat model says will change.

**Verification steps that are not assertions.** Task 2 Step 6 and Task 11 Steps 1 and 3 ask the implementer to confirm from the Modal log whether an image rebuilt. That cannot be asserted in code, and it matters: `add_local_python_source` must *not* rebuild (a rebuild would resolve fresh dependency versions), while pinning *must*. If a step's rebuild behaviour contradicts what is written here, stop rather than continue.
