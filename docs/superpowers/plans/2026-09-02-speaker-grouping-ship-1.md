# Voice Enrolment (Ship 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Name the five known voices across all 433 fetchable episodes by matching diarisation clusters against enrolment centroids built from audio already on disk, and end with the measurement that decides whether Ship 2 is worth building.

**Architecture:** Pure Python modules hold every decision worth being wrong about (crop selection, centroid maths, scoring, threshold-and-margin, conflict detection) and are unit-tested without GPU, audio or network. One Modal function does the irreversible expensive thing — fetch, decode, crop, embed, discard audio — and persists **per-crop vectors** rather than centroids, so every parameter downstream stays a local decision. Identity is written to Chroma as a **new** metadata key; nothing existing is overwritten.

**Tech Stack:** Python 3.12, Modal, pyannote.audio 4.0.7 (via `whisperx==3.8.5`), ChromaDB 1.5.9, pytest, ruff, mypy.

**Spec:** [`docs/superpowers/specs/2026-09-02-speaker-grouping-design.md`](../specs/2026-09-02-speaker-grouping-design.md)

## Global Constraints

- **Ship 1 only.** Ship 2 (archive-wide grouping of unnamed voices, `corpus/grouping.py`, naming-on-demand, the 45-pair recognition gate) is **out of scope** and is built only if Task 8's histogram justifies it (spec §1.3).
- **`speaker` metadata is NEVER overwritten.** Identity is added as `speaker_id`. `chunking.py:84` bakes `[SPEAKER_01] ` into the document text and `rendering.py:29` strips it using the metadata value; overwriting `speaker` breaks the renderer (spec §2.2).
- **The backfill never writes a document or an embedding.** Metadata only.
- **Non-`SPEAKER_*` buckets are excluded by rule.** `chunking.py:71` substitutes `UNKNOWN` for segments with no speaker; 53 episodes carry such a bucket. It is audio belonging to nobody in particular (spec §0.4).
- **Seed only from `basis: "voice"`.** `speakers/labels.json` holds 208 `voice` + 6 `context` + 165 `skipped`. The 6 `context` labels were inferred from episode text; `speaker_tool.py:705` warns in its own comment that such labels score a matcher correct for the wrong reason.
- **Replicate image specs verbatim** so Modal content-addressing reuses the cache. Build lines in the output mean the spec drifted and the numbers are not the deployed ones. `.add_local_python_source("corpus")` stays **last** in any image chain.
- **Windows/Git Bash:** `modal volume get` needs `MSYS_NO_PATHCONV=1`. Commit from the **Bash** tool, never PowerShell (it prepends a UTF-8 BOM to the subject line). Use `py script.py`, not `python script.py`.
- **Constants, copied verbatim from `corpus/speakers.py`:** `GAP_CAP_S = 30.0`, `MIN_TURN_S = 1.5`, `CLIP_MIN_TURN_S = 12.0`.
- **Measured baselines** any task may assert against: 433 fetchable episodes, 1,014 real `SPEAKER_*` clusters, 970 with a croppable turn, 2,712 crops at k=3, 56.1 GB of audio, 208 voice labels all present in `clips/`.
- Run `uv run pytest`, `uv run ruff check .`, and `uv run mypy` before every commit.

---

## File Structure

| file | responsibility | new? |
|---|---|---|
| `corpus/embedding.py` | which turns to crop, where to cut, duration-weighted centroids | create |
| `corpus/enrolment.py` | centroids from labels, cosine scoring, threshold+margin, conflict detection | create |
| `speaker_embed.py` | Modal app: load an embedder; embed local clips; embed archive episodes | create |
| `backfill_speaker_id.py` | additive `speaker_id` write to Chroma, counts pre-registered | create |
| `corpus/rendering.py` | the four render states; resolve `speaker_id` → name | modify |
| `mcp_server.py` | pass `speaker_id`/names through; `speaker` filter param + docstring | modify |
| `corpus/names.py` | derived `speaker_id → display name` map; ships with the already-mounted corpus package | create |
| `tests/test_embedding.py`, `tests/test_enrolment.py` | unit tests for the pure modules | create |
| `tests/test_rendering.py` | extended for the four states | modify |

---

## Task 1: Crop selection and centroid maths

Pure. No audio, no network, no model. This is where "which three turns" and "how do we average" get pinned.

**Files:**
- Create: `corpus/embedding.py`
- Create: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `corpus.speakers.clip_window(turn, *, lead_in_s, length_s) -> tuple[float, float]`; turn records shaped `{"speaker", "start", "duration_s", "n_words", "n_segments"}`.
- Produces:
  - `CROPS_PER_CLUSTER = 3`, `CROP_LEAD_IN_S = 2.0`, `CROP_LENGTHS_S = (10.0, 30.0)`
  - `select_crop_turns(turns: list[dict], speaker: str, *, k: int, min_turn_s: float) -> list[dict]`
  - `cluster_crop_plan(episode: dict, *, min_turn_s: float, k: int = CROPS_PER_CLUSTER, lead_in_s: float = CROP_LEAD_IN_S, lengths: tuple[float, ...] = CROP_LENGTHS_S) -> list[dict]` returning records `{"episode", "cluster", "draw", "length_s", "turn_start", "turn_duration_s", "crop_start_s", "crop_length_s"}`
  - `weighted_centroid(vectors: list[list[float]], weights: list[float]) -> list[float]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embedding.py`:

```python
"""Crop selection and centroid maths.

Two quantities are deliberately different here and conflating them is the
defect this module exists to avoid: a centroid is weighted by a turn's TRUE
duration, while the audio embedded is a duration-CAPPED crop of it. Weighting
by the crop length would silently under-weight the longest, cleanest turns --
exactly the turns that carry the most evidence about a voice.
"""

from __future__ import annotations

import math

import pytest

from corpus.embedding import (
    CROP_LENGTHS_S,
    cluster_crop_plan,
    select_crop_turns,
    weighted_centroid,
)

TURNS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "duration_s": 40.0, "n_words": 100},
    {"speaker": "SPEAKER_00", "start": 100.0, "duration_s": 15.0, "n_words": 40},
    {"speaker": "SPEAKER_00", "start": 200.0, "duration_s": 60.0, "n_words": 150},
    {"speaker": "SPEAKER_00", "start": 300.0, "duration_s": 5.0, "n_words": 10},
    {"speaker": "SPEAKER_01", "start": 400.0, "duration_s": 90.0, "n_words": 200},
    {"speaker": "UNKNOWN", "start": 500.0, "duration_s": 80.0, "n_words": 180},
]


def test_selects_the_longest_turns_descending():
    picked = select_crop_turns(TURNS, "SPEAKER_00", k=3, min_turn_s=12.0)
    assert [t["duration_s"] for t in picked] == [60.0, 40.0, 15.0]


def test_excludes_turns_below_the_floor():
    picked = select_crop_turns(TURNS, "SPEAKER_00", k=10, min_turn_s=12.0)
    assert 5.0 not in [t["duration_s"] for t in picked]


def test_returns_fewer_than_k_when_the_speaker_is_short_of_turns():
    picked = select_crop_turns(TURNS, "SPEAKER_01", k=3, min_turn_s=12.0)
    assert len(picked) == 1


def test_returns_empty_when_no_turn_is_long_enough():
    # 44 real clusters archive-wide are in exactly this position. They must
    # produce no crops rather than a short one -- a two-second crop is not
    # evidence about who is speaking.
    assert select_crop_turns(TURNS, "SPEAKER_00", k=3, min_turn_s=120.0) == []


def test_equal_durations_break_ties_by_start_so_the_plan_is_deterministic():
    turns = [
        {"speaker": "A", "start": 50.0, "duration_s": 20.0},
        {"speaker": "A", "start": 10.0, "duration_s": 20.0},
    ]
    assert [t["start"] for t in select_crop_turns(turns, "A", k=2, min_turn_s=1.0)] == [
        10.0,
        50.0,
    ]


def test_crop_plan_excludes_non_speaker_buckets():
    # chunking.py:71 writes UNKNOWN for segments with no speaker. Embedding it
    # would build a centroid from audio belonging to nobody in particular.
    plan = cluster_crop_plan({"name": "ep.txt", "turns": TURNS}, min_turn_s=12.0)
    assert {r["cluster"] for r in plan} == {"SPEAKER_00", "SPEAKER_01"}


def test_crop_plan_emits_one_record_per_cluster_turn_and_length():
    plan = cluster_crop_plan({"name": "ep.txt", "turns": TURNS}, min_turn_s=12.0)
    # SPEAKER_00 contributes 3 turns, SPEAKER_01 contributes 1, times 2 lengths.
    assert len(plan) == (3 + 1) * len(CROP_LENGTHS_S)


def test_crop_never_runs_past_the_end_of_its_turn():
    plan = cluster_crop_plan({"name": "ep.txt", "turns": TURNS}, min_turn_s=12.0)
    for record in plan:
        end = record["crop_start_s"] + record["crop_length_s"]
        assert end <= record["turn_start"] + record["turn_duration_s"] + 1e-9


def test_crop_plan_records_the_true_turn_duration_not_the_crop_length():
    plan = cluster_crop_plan({"name": "ep.txt", "turns": TURNS}, min_turn_s=12.0)
    longest = max(plan, key=lambda r: r["turn_duration_s"])
    assert longest["turn_duration_s"] == 90.0
    assert longest["crop_length_s"] <= 30.0


def test_centroid_is_unit_length():
    out = weighted_centroid([[3.0, 4.0], [0.0, 1.0]], [1.0, 1.0])
    assert math.isclose(math.sqrt(sum(v * v for v in out)), 1.0, rel_tol=1e-9)


def test_centroid_weighting_pulls_toward_the_heavier_vector():
    light = weighted_centroid([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    heavy = weighted_centroid([[1.0, 0.0], [0.0, 1.0]], [9.0, 1.0])
    assert heavy[0] > light[0]


def test_centroid_rejects_empty_input():
    with pytest.raises(ValueError, match="no vectors"):
        weighted_centroid([], [])


def test_centroid_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights"):
        weighted_centroid([[1.0, 0.0]], [1.0, 2.0])


def test_centroid_rejects_ragged_vectors():
    with pytest.raises(ValueError, match="dimension"):
        weighted_centroid([[1.0, 0.0], [1.0]], [1.0, 1.0])


def test_centroid_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="weight"):
        weighted_centroid([[1.0, 0.0]], [0.0])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_embedding.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'corpus.embedding'`.

- [ ] **Step 3: Write the implementation**

Create `corpus/embedding.py`:

```python
"""Which audio to embed, and how to average the results.

No modal, no audio library, no network -- the same split `corpus/speakers.py`
and `corpus/showplan.py` argue for, so every decision that shapes a centroid
is reachable from the test suite on CPU.

TWO DURATIONS, DELIBERATELY DIFFERENT. A centroid is weighted by a turn's TRUE
duration; the audio actually embedded is a duration-capped crop of that turn.
Weighting by the crop length instead would flatten a 90-second turn and a
15-second one to the same influence, under-weighting exactly the longest and
cleanest speech in the episode.

TWO CROP LENGTHS, ON PURPOSE. Crop length is the one parameter the audio pass
cannot revisit locally: everything else (k, the weighting, thresholds) is
recomputed from persisted per-crop vectors in seconds, but changing the window
means re-downloading 56.1 GB. Embedding at two lengths in the one pass costs
minutes of GPU and converts the last irreversible parameter into a local one.
"""

from __future__ import annotations

import math

from corpus.speakers import clip_window

# Three crops per cluster. Part 5 of the eval design measured diarisation
# clusters as holding one person (0 contaminated of the 12 that carried
# enough labels to judge), which is what licenses averaging a handful of turns
# into one centroid rather than embedding all 19,782 turns.
CROPS_PER_CLUSTER = 3

# Skip into the turn before cutting. The previous speaker's tail bleeds across
# a diarisation boundary; `clip_window` documents the same reasoning.
CROP_LEAD_IN_S = 2.0

CROP_LENGTHS_S = (10.0, 30.0)


def select_crop_turns(
    turns: list[dict], speaker: str, *, k: int, min_turn_s: float
) -> list[dict]:
    """The k longest eligible turns for `speaker`, longest first.

    LONGEST, not random -- and that is the opposite of `pick_clip_turn`, which
    draws at random on purpose. The difference is what the audio is for: an
    eval clip must not be cherry-picked, because the easiest clip flatters the
    matcher. A centroid is not being scored, it is being built, so the
    cleanest available speech is simply the best evidence about the voice.

    Ties break on start time so a plan is reproducible from the transcript
    alone.
    """
    eligible = [
        t
        for t in turns
        if t["speaker"] == speaker and t["duration_s"] >= min_turn_s
    ]
    eligible.sort(key=lambda t: (-t["duration_s"], t["start"]))
    return eligible[:k]


def cluster_crop_plan(
    episode: dict,
    *,
    min_turn_s: float,
    k: int = CROPS_PER_CLUSTER,
    lead_in_s: float = CROP_LEAD_IN_S,
    lengths: tuple[float, ...] = CROP_LENGTHS_S,
) -> list[dict]:
    """Every crop this episode owes, one record per (cluster, turn, length).

    Non-`SPEAKER_*` buckets are excluded by rule. `chunking.py:71` substitutes
    `UNKNOWN` for a segment carrying no speaker, and 53 episodes hold such a
    bucket; a centroid built from it would be audio belonging to nobody in
    particular, which is the silent over-merge the design names as the error
    that matters most.
    """
    plan: list[dict] = []
    clusters = sorted(
        {t["speaker"] for t in episode["turns"] if t["speaker"].startswith("SPEAKER_")}
    )
    for cluster in clusters:
        chosen = select_crop_turns(
            episode["turns"], cluster, k=k, min_turn_s=min_turn_s
        )
        for draw, turn in enumerate(chosen):
            for length_s in lengths:
                start, actual = clip_window(
                    turn, lead_in_s=lead_in_s, length_s=length_s
                )
                plan.append(
                    {
                        "episode": episode["name"],
                        "cluster": cluster,
                        "draw": draw,
                        "length_s": length_s,
                        "turn_start": turn["start"],
                        "turn_duration_s": turn["duration_s"],
                        "crop_start_s": start,
                        "crop_length_s": actual,
                    }
                )
    return plan


def weighted_centroid(
    vectors: list[list[float]], weights: list[float]
) -> list[float]:
    """Duration-weighted mean of `vectors`, L2-normalised.

    Normalised so that downstream scoring is a plain dot product and a
    centroid built from three crops is comparable with one built from one.
    Raises rather than returning a degenerate vector: a silently-wrong
    centroid produces confident wrong identities, which is the failure mode
    with no external symptom.
    """
    if not vectors:
        raise ValueError("no vectors to average")
    if len(vectors) != len(weights):
        raise ValueError(
            f"weights must match vectors: {len(weights)} vs {len(vectors)}"
        )
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("total weight must be positive")

    dim = len(vectors[0])
    accumulator = [0.0] * dim
    for vector, weight in zip(vectors, weights, strict=True):
        if len(vector) != dim:
            raise ValueError(
                f"inconsistent dimension: {len(vector)} vs {dim}"
            )
        for i, value in enumerate(vector):
            accumulator[i] += value * weight

    mean = [value / total for value in accumulator]
    norm = math.sqrt(sum(value * value for value in mean))
    if norm == 0.0:
        raise ValueError("centroid has zero norm")
    return [value / norm for value in mean]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_embedding.py -v
uv run ruff check corpus/embedding.py tests/test_embedding.py
uv run mypy
```

Expected: 15 passed; ruff and mypy clean.

- [ ] **Step 5: Commit**

```bash
git add corpus/embedding.py tests/test_embedding.py
git commit -F- <<'EOF'
feat: pick crops and average them, with the two durations kept apart

A centroid is weighted by a turn's TRUE duration while the audio embedded is
a duration-capped crop of it. Conflating those under-weights the longest,
cleanest turns -- the ones carrying the most evidence about a voice.

Crops are cut at two window lengths in one pass. Crop length is the only
parameter the audio pass cannot revisit locally, since changing it means
re-downloading 56.1 GB; embedding at two lengths costs minutes of GPU and
makes it a local decision like every other.

Longest turns, not random ones -- the opposite of pick_clip_turn, and for a
reason: an eval clip must not be cherry-picked because the easiest clip
flatters the matcher, but a centroid is being built rather than scored.

Non-SPEAKER_ buckets are excluded by rule, not by luck. 53 episodes carry an
UNKNOWN bucket from chunking.py:71; only 3 are croppable today, and a rule
that holds at 3 keeps holding when the count changes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KhAPYARf38HUujE4PfF4G
EOF
```

---

## Task 2: Enrolment — centroids from labels, scoring, and the refusal rule

Pure. This is the decision layer: who a cluster is, or nobody.

**Files:**
- Create: `corpus/enrolment.py`
- Create: `tests/test_enrolment.py`

**Interfaces:**
- Consumes: `corpus.embedding.weighted_centroid`. Label records shaped `{clip_id: {"person": str | None, "basis": "voice" | "context" | "skipped"}}` (the on-disk shape of `speakers/labels.json`).
- Produces:
  - `build_centroids(labels: dict[str, dict], vectors: dict[str, list[float]], weights: dict[str, float] | None = None) -> dict[str, list[float]]`
  - `score(vector: list[float], centroids: dict[str, list[float]]) -> list[tuple[str, float]]` — descending by score
  - `assign(vector: list[float], centroids: dict[str, list[float]], *, threshold: float, margin: float) -> str | None`
  - `co_membership_conflicts(assignments: list[dict]) -> list[dict]` over records `{"episode", "cluster", "person"}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_enrolment.py`:

```python
"""Assigning a cluster to an enrolled voice, or to nobody.

`None` is a RESULT, not a failure. Naming a cluster that is not one of the
enrolled people is the precision violation that matters most, so every
refusal path here is tested as carefully as every match.
"""

from __future__ import annotations

import math

import pytest

from corpus.enrolment import (
    assign,
    build_centroids,
    co_membership_conflicts,
    score,
)

A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]
NEAR_A = [math.sqrt(0.99), math.sqrt(0.01), 0.0]

LABELS = {
    "clip-a1": {"person": "Jacob Shapiro", "basis": "voice"},
    "clip-a2": {"person": "Jacob Shapiro", "basis": "voice"},
    "clip-b1": {"person": "Rob Larity", "basis": "voice"},
    "clip-ctx": {"person": "Marko Papic", "basis": "context"},
    "clip-skip": {"person": None, "basis": "skipped"},
}
VECTORS = {
    "clip-a1": A,
    "clip-a2": A,
    "clip-b1": B,
    "clip-ctx": B,
    "clip-skip": A,
}


def test_builds_one_centroid_per_named_person():
    centroids = build_centroids(LABELS, VECTORS)
    assert set(centroids) == {"Jacob Shapiro", "Rob Larity"}


def test_excludes_context_labels_from_enrolment():
    # speaker_tool.py:705 warns that a label inferred from episode text scores
    # a matcher correct for the wrong reason. Part 8.4 forecloses every
    # text-only route to speaker identity; importing one here would smuggle it
    # back in as voice ground truth.
    assert "Marko Papic" not in build_centroids(LABELS, VECTORS)


def test_excludes_skipped_labels_from_enrolment():
    centroids = build_centroids(LABELS, VECTORS)
    assert None not in centroids
    assert len(centroids) == 2


def test_ignores_labels_with_no_vector():
    centroids = build_centroids(LABELS, {"clip-a1": A})
    assert set(centroids) == {"Jacob Shapiro"}


def test_centroids_are_unit_length():
    for vector in build_centroids(LABELS, VECTORS).values():
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_score_orders_candidates_best_first():
    ranked = score(NEAR_A, {"Jacob Shapiro": A, "Rob Larity": B})
    assert [name for name, _ in ranked] == ["Jacob Shapiro", "Rob Larity"]


def test_assign_returns_the_best_match_when_clear():
    centroids = {"Jacob Shapiro": A, "Rob Larity": B}
    assert assign(NEAR_A, centroids, threshold=0.5, margin=0.1) == "Jacob Shapiro"


def test_assign_refuses_below_the_threshold():
    centroids = {"Jacob Shapiro": A, "Rob Larity": B}
    assert assign(NEAR_A, centroids, threshold=0.999, margin=0.0) is None


def test_assign_refuses_when_two_candidates_are_a_hair_apart():
    # The dangerous case: a threshold alone resolves this confidently and
    # wrongly. Both score well; neither wins by enough.
    tie = [math.sqrt(0.5), math.sqrt(0.5), 0.0]
    centroids = {"Jacob Shapiro": A, "Rob Larity": B}
    assert assign(tie, centroids, threshold=0.5, margin=0.05) is None


def test_assign_returns_none_when_there_are_no_centroids():
    assert assign(A, {}, threshold=0.5, margin=0.1) is None


def test_assign_matches_a_lone_candidate_that_clears_the_threshold():
    # With one enrolled voice there is no runner-up, so margin cannot apply.
    assert assign(A, {"Jacob Shapiro": A}, threshold=0.9, margin=0.5) == "Jacob Shapiro"


def test_co_membership_flags_two_clusters_of_one_episode_sharing_a_person():
    assignments = [
        {"episode": "ep1.txt", "cluster": "SPEAKER_00", "person": "Jacob Shapiro"},
        {"episode": "ep1.txt", "cluster": "SPEAKER_01", "person": "Jacob Shapiro"},
        {"episode": "ep2.txt", "cluster": "SPEAKER_00", "person": "Jacob Shapiro"},
    ]
    conflicts = co_membership_conflicts(assignments)
    assert len(conflicts) == 1
    assert conflicts[0]["episode"] == "ep1.txt"
    assert conflicts[0]["person"] == "Jacob Shapiro"
    assert sorted(conflicts[0]["clusters"]) == ["SPEAKER_00", "SPEAKER_01"]


def test_co_membership_ignores_unassigned_clusters():
    # Two refusals in one episode are not a conflict -- they are two
    # independent "we do not know", which is the designed behaviour.
    assignments = [
        {"episode": "ep1.txt", "cluster": "SPEAKER_00", "person": None},
        {"episode": "ep1.txt", "cluster": "SPEAKER_01", "person": None},
    ]
    assert co_membership_conflicts(assignments) == []


def test_co_membership_is_empty_when_each_person_appears_once_per_episode():
    assignments = [
        {"episode": "ep1.txt", "cluster": "SPEAKER_00", "person": "Jacob Shapiro"},
        {"episode": "ep1.txt", "cluster": "SPEAKER_01", "person": "Rob Larity"},
    ]
    assert co_membership_conflicts(assignments) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_enrolment.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'corpus.enrolment'`.

- [ ] **Step 3: Write the implementation**

Create `corpus/enrolment.py`:

```python
"""Matching a cluster to an enrolled voice, or to nobody.

Centroids are DERIVED from labels and never stored. `speakers/labels.json`
holds provenance only. This follows `corpus/exclusions.py`, where the lists
are derived for the same reason: two hand-maintained copies drift
asymmetrically, and the stale one works right up until the moment it matters.

`None` IS THE ANSWER, not a missing one. Under-merging is an annoyance a
person can repair; over-merging silently attributes one person's views to
another and corrupts the "has the view moved" query this exists to serve, with
nothing in the output to show it happened.
"""

from __future__ import annotations

import math

from corpus.embedding import weighted_centroid

# Only voice-verified labels enrol. `speaker_tool.py:705` offers "heard it (v)
# or inferred from context (c)" and warns in its own comment that a context
# label scores a matcher correct for the wrong reason.
ENROLMENT_BASIS = "voice"


def build_centroids(
    labels: dict[str, dict],
    vectors: dict[str, list[float]],
    weights: dict[str, float] | None = None,
) -> dict[str, list[float]]:
    """One centroid per named person, from voice-verified labels only.

    A label with no vector is skipped rather than raising: the caller may hold
    labels for clips it has not embedded, and refusing the whole enrolment
    over one absent clip would be brittle for no gain.
    """
    grouped: dict[str, list[tuple[list[float], float]]] = {}
    for clip_id, label in labels.items():
        person = label.get("person")
        if person is None or label.get("basis") != ENROLMENT_BASIS:
            continue
        vector = vectors.get(clip_id)
        if vector is None:
            continue
        weight = 1.0 if weights is None else weights.get(clip_id, 1.0)
        grouped.setdefault(person, []).append((vector, weight))

    return {
        person: weighted_centroid([v for v, _ in items], [w for _, w in items])
        for person, items in sorted(grouped.items())
    }


def score(
    vector: list[float], centroids: dict[str, list[float]]
) -> list[tuple[str, float]]:
    """Every candidate and its similarity, best first.

    A true cosine. The CENTROID side is unit-length by construction
    (`weighted_centroid` normalises), but the query side is a raw model
    output -- `_Embedder.embed` returns pyannote's vector untouched -- so
    dividing by its norm is required rather than cosmetic. Assuming otherwise
    makes every threshold scale-dependent and silently unstable.

    Ties break on name so the order is reproducible.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        raise ValueError("cannot score a zero vector")
    ranked = [
        (
            person,
            sum(a * b for a, b in zip(vector, centroid, strict=True)) / norm,
        )
        for person, centroid in centroids.items()
    ]
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked


def assign(
    vector: list[float],
    centroids: dict[str, list[float]],
    *,
    threshold: float,
    margin: float,
) -> str | None:
    """The enrolled person this vector belongs to, or None.

    BOTH conditions must hold. A threshold alone resolves the dangerous case
    -- two high scores a hair apart -- confidently and wrongly, so the runner
    up must also be beaten by `margin`. With a single enrolled voice there is
    no runner-up and the margin cannot apply.
    """
    ranked = score(vector, centroids)
    if not ranked:
        return None
    best_person, best_score = ranked[0]
    if best_score < threshold:
        return None
    if len(ranked) > 1 and best_score - ranked[1][1] < margin:
        return None
    return best_person


def co_membership_conflicts(assignments: list[dict]) -> list[dict]:
    """Episodes where two clusters were assigned the SAME person.

    A free over-merge detector needing no labels, so unlike the seed conflict
    check it covers every cluster rather than the 38% that carry a label.

    It is a FLAG, not a gate. Diarisation can over-segment one person into two
    clusters, and reuniting those is correct -- arguably the feature working.
    Two genuinely different people collapsing into one identity is the error.
    This cannot tell them apart; it partitions the archive into the cases
    worth looking at, which is all it claims to do.
    """
    seen: dict[tuple[str, str], list[str]] = {}
    for record in assignments:
        person = record.get("person")
        if person is None:
            continue
        seen.setdefault((record["episode"], person), []).append(record["cluster"])

    return [
        {"episode": episode, "person": person, "clusters": sorted(clusters)}
        for (episode, person), clusters in sorted(seen.items())
        if len(clusters) > 1
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_enrolment.py -v
uv run ruff check corpus/enrolment.py tests/test_enrolment.py
uv run mypy
```

Expected: 14 passed; ruff and mypy clean.

- [ ] **Step 5: Commit**

```bash
git add corpus/enrolment.py tests/test_enrolment.py
git commit -F- <<'EOF'
feat: match a cluster to an enrolled voice, or refuse

None is a RESULT here, not a missing one. Under-merging is an annoyance a
person repairs by naming two groups; over-merging silently attributes one
person's views to another and corrupts the query this exists to serve, with
nothing in the output to show it happened. So assignment needs a threshold
AND a margin: the dangerous case is two high scores a hair apart, which a
threshold alone resolves confidently and wrongly.

Centroids are derived from labels and never stored, following
corpus/exclusions.py -- two hand-maintained copies drift asymmetrically and
the stale one works right up until it matters.

Only basis:voice enrols. The 6 context labels were inferred from episode text
and speaker_tool.py:705 warns in its own comment that such a label scores a
matcher correct for the wrong reason.

Also lands the same-episode co-membership flag: a free over-merge detector
that needs no labels, so it covers every cluster rather than the 38% carrying
one. A flag, not a gate -- diarisation over-segmentation and a genuine
two-person merge look identical to it, and it says so.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KhAPYARf38HUujE4PfF4G
EOF
```

---

## Task 3: The embedder, and embedding the clips already on disk

Settles spec UNKNOWN 1 (can community-1's embedder be used standalone?) on real labelled audio, at CPU cost, before a byte of the 56.1 GB is spent.

**Files:**
- Create: `speaker_embed.py`

**Interfaces:**
- Produces:
  - `image` — the transcriber image spec, verbatim
  - `load_embedder(name: str, hf_token: str)` returning an object with `.embed(path: str, start_s: float, length_s: float) -> list[float]`
  - Modal function `embed_clips(embedder_name: str) -> dict[str, list[float]]`
  - CLI: `modal run speaker_embed.py::main --mode clips --embedder <name>`
  - Writes `speakers/clip_vectors_<embedder>.json`

- [ ] **Step 1: Write the file**

Create `speaker_embed.py`:

```python
"""Voice embedding: the one expensive, irreversible step.

WHAT THIS PERSISTS AND WHY. Per-crop vectors, not centroids. Fetching and
decoding 56.1 GB of someone else's audio cannot be re-run cheaply to change a
parameter, so the pass emits the most raw artifact that is still small --
2,712 vectors of 256 floats, single-digit MB. Everything downstream (k, the
weighting, the centroid rule, thresholds, margins) then re-runs locally in
seconds. The only parameter this cannot rescue is crop WINDOW LENGTH, which is
why every crop is embedded at two lengths in the same pass.

The image spec is copied VERBATIM from transcribe.py:9-27 so Modal's
content-addressing reuses the cache. Per docs/operations.md, build lines in
the output mean the spec drifted and the versions are not the deployed ones.
`.add_local_python_source("corpus")` stays LAST.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# --- VERBATIM from transcribe.py:9-27. Do not tidy. ---------------------
_layers = (
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
)
# ------------------------------------------------------------------------

# add_local_python_source stays LAST on BOTH images. Appending add_local_dir
# to a chain that already ended with it would put a mount after the mount the
# repo requires to be final -- so the shared prefix is named once and each
# image finishes its own chain.
image = _layers.add_local_python_source("corpus")
clips_image = _layers.add_local_dir("clips", "/clips").add_local_python_source(
    "corpus"
)

app = modal.App("speaker-embed")

model_cache = modal.Volume.from_name("whisperx-model-cache", create_if_missing=True)
vectors = modal.Volume.from_name("speaker-vectors", create_if_missing=True)

MODEL_PATH = "/models"
VECTORS_PATH = "/vectors"

# The embedder that produced the archive's own diarisation clusters, and the
# verified-ungated standalone fallback. `pyannote/embedding` is GATED (403)
# and `speechbrain/spkrec-ecapa-voxceleb` is absent from this image; both were
# probed on 2026-09-02 and are not candidates.
COMMUNITY = "pyannote/speaker-diarization-community-1"
WESPEAKER = "pyannote/wespeaker-voxceleb-resnet34-LM"


class _Embedder:
    """Wraps whichever model was loaded behind one `.embed` call."""

    def __init__(self, inference, sample_rate: int = 16000) -> None:
        self._inference = inference
        self.sample_rate = sample_rate

    def embed(self, path: str, start_s: float, length_s: float) -> list[float]:
        import torchaudio

        offset = int(start_s * self.sample_rate)
        frames = int(length_s * self.sample_rate)
        waveform, rate = torchaudio.load(
            path, frame_offset=offset, num_frames=frames
        )
        if rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, rate, self.sample_rate
            )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        vector = self._inference(
            {"waveform": waveform, "sample_rate": self.sample_rate}
        )
        return [float(v) for v in vector.reshape(-1)]


def load_embedder(name: str, hf_token: str) -> _Embedder:
    """Return an embedder for `name`, or raise with the reason.

    Never returns a degraded object. A model that silently failed to load
    would produce vectors that are wrong rather than absent, and wrong vectors
    make confident wrong identities -- the failure with no external symptom.

    HF_HOME is pointed at the mounted model volume HERE rather than via
    `image.env(...)`, because changing the image would change its content
    address and force the rebuild the verbatim copy exists to avoid. Without
    it every one of the 8 containers re-downloads the model: the probe on
    2026-09-02 found /models holds the whisper and bge weights and NO pyannote
    weights, so diarisation has been paying that cost per cold start already.
    """
    os.environ.setdefault("HF_HOME", MODEL_PATH)

    from pyannote.audio import Inference, Model

    if name == COMMUNITY:
        # The embedder ships INSIDE the diarisation repo, in an `embedding`
        # subfolder -- probed 2026-09-02 from the live pipeline's config. This
        # is the standalone-use question spec UNKNOWN 1 records.
        model = Model.from_pretrained(
            name, subfolder="embedding", use_auth_token=hf_token
        )
    else:
        model = Model.from_pretrained(name, use_auth_token=hf_token)
    return _Embedder(Inference(model, window="whole"))


@app.function(
    image=clips_image,
    volumes={MODEL_PATH: model_cache, VECTORS_PATH: vectors},
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=3600,
)
def embed_clips(embedder_name: str) -> dict[str, list[float]]:
    """Embed every clip in `clips/`, keyed by clip_id.

    These are the 381 clips cut for the Tier 1 evaluation, already on disk and
    already voice-verified. Enrolment therefore costs NO new audio, and the
    embedder choice gets decided on real labelled voices before the archive
    pass runs.
    """
    embedder = load_embedder(embedder_name, os.environ["HF_TOKEN"])
    out: dict[str, list[float]] = {}
    failures: list[str] = []

    paths = sorted(Path("/clips").glob("*.mp3"))
    print(f"clips={len(paths)} embedder={embedder_name}")
    for path in paths:
        try:
            # window="whole" over the entire clip: these were cut to length
            # already, so there is nothing further to crop.
            out[path.stem] = embedder.embed(str(path), 0.0, 3600.0)
        except Exception as e:  # noqa: BLE001 - collected, not swallowed
            failures.append(f"{path.stem}: {type(e).__name__}: {e}")

    print(f"embedded={len(out)} failed={len(failures)}")
    for line in failures[:10]:
        print(f"  FAILED {line}")
    if not out:
        raise RuntimeError(
            "no clip produced a vector -- this is a broken embedder, not an "
            "empty input directory"
        )
    print("EMBED CLIPS COMPLETE")
    return out


@app.local_entrypoint()
def main(mode: str = "clips", embedder: str = COMMUNITY) -> None:
    if mode != "clips":
        raise SystemExit(f"unknown mode {mode!r}")
    result = embed_clips.remote(embedder)
    slug = embedder.split("/")[-1]
    destination = Path("speakers") / f"clip_vectors_{slug}.json"
    destination.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {len(result)} vectors to {destination}")
```

- [ ] **Step 2: Run it for the preferred embedder**

```bash
modal run speaker_embed.py --mode clips --embedder pyannote/speaker-diarization-community-1
```

Expected: no build lines (cache reuse); `clips=381`; `embedded=381 failed=0`; `EMBED CLIPS COMPLETE`; a `speakers/clip_vectors_speaker-diarization-community-1.json` written.

**If this raises on `Model.from_pretrained(..., subfolder="embedding")`,** UNKNOWN 1 has resolved against the preference. That is an expected outcome, not a blocker — record the exact error in the spec's §6.1 and proceed with the fallback in Step 3, which §0.2 verified end-to-end.

- [ ] **Step 3: Run it for the fallback**

```bash
modal run speaker_embed.py --mode clips --embedder pyannote/wespeaker-voxceleb-resnet34-LM
```

Expected: `embedded=381 failed=0`, a 256-dimension vector per clip, and a second JSON file.

- [ ] **Step 4: Commit**

```bash
git add speaker_embed.py
git commit -F- <<'EOF'
feat: embed voices, starting with the clips already on disk

Enrolment costs no new audio. clips/ still holds all 381 clips cut for the
Tier 1 evaluation, and all 208 basis:voice ones are present -- so the five
enrolment centroids, and the community-1-vs-wespeaker question the spec
carries as UNKNOWN 1, are settled on real voice-verified audio at CPU cost
before a byte of the 56.1 GB is spent.

Persists per-crop vectors rather than centroids. The audio pass is the one
irreversible step here, so it emits the most raw artifact that is still small
and leaves k, the weighting, thresholds and margins as local decisions that
re-run in seconds.

Image spec copied verbatim from transcribe.py so content-addressing reuses
the cache; build lines in the output mean it drifted.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KhAPYARf38HUujE4PfF4G
EOF
```

---

## Task 4: The bake-off — choose the embedder on labelled data

A measurement, not a feature. It produces a number and a decision, and it is explicitly **not a precision claim** — those labels are a dev set (eval design §3.4).

**Files:**
- Create: `bakeoff_embedder.py`
- Modify: `docs/superpowers/specs/2026-09-02-speaker-grouping-design.md` (record the result in §3.3 and close §6.1)

- [ ] **Step 1: Pre-register the decision rule, before looking at any number**

Append to the spec's §3.3, and commit **before** running Step 2:

> **Bake-off pre-registration, 2026-09-02.** The embedder with the higher
> leave-one-out accuracy over the 208 `basis: voice` clips wins. If the two
> are within 2 percentage points, community-1 wins on the tie-break, because
> it is the space the archive's clusters were cut in. This rule is fixed
> before either number is seen. **The resulting figure is a dev-set
> measurement and is not a precision claim.**

- [ ] **Step 2: Write the bake-off script**

Create `bakeoff_embedder.py`:

```python
"""Which embedder separates these five voices better?

Leave-one-out over the 208 voice-verified labels: hold one clip out, build
centroids from the rest, and ask whether the held-out clip is assigned to its
own labelled person. Cheap, local, and it uses only audio already on disk.

NOT A PRECISION CLAIM. Per the eval design's 3.4 these labels are a dev set --
learning from them is exactly what a dev set is for, and quoting a number off
them as precision is what 3.4 forbids. This picks a model; it does not measure
the system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corpus.enrolment import assign, build_centroids, score

LABELS_PATH = Path("speakers/labels.json")
THRESHOLD = 0.0  # scored as a pure nearest-centroid choice: no refusals here,
MARGIN = 0.0  # because the question is separability, not the operating point.


def leave_one_out(labels: dict[str, dict], vectors: dict[str, list[float]]) -> dict:
    voice = {
        clip_id: label
        for clip_id, label in labels.items()
        if label.get("basis") == "voice" and label.get("person") is not None
    }
    scored = correct = skipped = 0
    confusions: dict[tuple[str, str], int] = {}
    same: list[float] = []
    different: list[float] = []

    for held_out, label in sorted(voice.items()):
        if held_out not in vectors:
            skipped += 1
            continue
        rest = {k: v for k, v in voice.items() if k != held_out}
        centroids = build_centroids(rest, vectors)
        if label["person"] not in centroids:
            # The only clip for this person; nothing to enrol from.
            skipped += 1
            continue
        ranked = score(vectors[held_out], centroids)
        for person, value in ranked:
            (same if person == label["person"] else different).append(value)
        predicted = assign(
            vectors[held_out], centroids, threshold=THRESHOLD, margin=MARGIN
        )
        scored += 1
        if predicted == label["person"]:
            correct += 1
        else:
            confusions[(label["person"], str(predicted))] = (
                confusions.get((label["person"], str(predicted)), 0) + 1
            )

    # The SCORE DISTRIBUTION, not just the accuracy. Task 6's THRESHOLD and
    # MARGIN are the only unmeasured numbers in this plan and the spec leaves
    # them unset on purpose (6.7); they are meant to come from here. A
    # bake-off that printed accuracy alone would leave the next task guessing.
    same.sort()
    different.sort()

    def pct(values: list[float], q: float) -> float:
        return values[min(int(q * len(values)), len(values) - 1)] if values else 0.0

    return {
        "scored": scored,
        "correct": correct,
        "skipped": skipped,
        "accuracy": correct / scored if scored else 0.0,
        "confusions": {f"{a} -> {b}": n for (a, b), n in sorted(confusions.items())},
        "same_person_score_p05": pct(same, 0.05),
        "same_person_score_median": pct(same, 0.5),
        "other_person_score_p95": pct(different, 0.95),
        "other_person_score_median": pct(different, 0.5),
        "suggested_threshold": (pct(same, 0.05) + pct(different, 0.95)) / 2,
        "suggested_margin": max(0.0, (pct(same, 0.05) - pct(different, 0.95)) / 4),
    }


def main() -> int:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    results = {}
    for path in sorted(Path("speakers").glob("clip_vectors_*.json")):
        name = path.stem.replace("clip_vectors_", "")
        vectors = json.loads(path.read_text(encoding="utf-8"))
        # Prove the probe can return something before believing its verdict:
        # zero overlap between labels and vectors would score 0.0 and look
        # exactly like a model that cannot tell two people apart.
        overlap = len(set(labels) & set(vectors))
        print(f"\n== {name}: labels={len(labels)} vectors={len(vectors)} overlap={overlap}")
        if overlap == 0:
            print("  REFUSING: no clip_id in common. This is a key mismatch,")
            print("  not a model result.")
            return 2
        outcome = leave_one_out(labels, vectors)
        results[name] = outcome
        print(
            f"  accuracy={outcome['accuracy']:.3f} "
            f"({outcome['correct']}/{outcome['scored']}, skipped={outcome['skipped']})"
        )
        for pair, count in outcome["confusions"].items():
            print(f"    {pair}: {count}")
        print(
            f"  same-person p05={outcome['same_person_score_p05']:.3f} "
            f"median={outcome['same_person_score_median']:.3f}"
        )
        print(
            f"  other-person p95={outcome['other_person_score_p95']:.3f} "
            f"median={outcome['other_person_score_median']:.3f}"
        )
        print(
            f"  -> suggested threshold={outcome['suggested_threshold']:.3f} "
            f"margin={outcome['suggested_margin']:.3f}"
        )

    Path("speakers/bakeoff.json").write_text(json.dumps(results, indent=2), "utf-8")
    print("\nBAKEOFF COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run it**

```bash
py bakeoff_embedder.py
```

Expected: an `overlap` of 208 or more for each vector file, a per-embedder accuracy, a same-person and other-person score distribution, a suggested threshold and margin, and `BAKEOFF COMPLETE`. **Carry the suggested values into Task 6** — they are the only numbers in this plan the spec deliberately left unset (§6.7). An `overlap=0` refusal means the clip-id keys disagree — fix the keying, do not interpret the score.

- [ ] **Step 4: Record the result and close UNKNOWN 1**

Edit the spec: in §3.3 state which embedder won and by how much; in §6.1 replace the UNKNOWN with the measured outcome. If community-1 could not be loaded standalone, say so explicitly there — that is the answer to the question, not a failure of the task.

- [ ] **Step 5: Commit**

```bash
git add bakeoff_embedder.py speakers/bakeoff.json docs/superpowers/specs/2026-09-02-speaker-grouping-design.md
git commit -F- <<'EOF'
feat: pick the embedder by measuring it, on a rule fixed beforehand

Leave-one-out over the 208 voice-verified clips, with the decision rule
pre-registered in the spec and committed before either number was seen -- the
countermeasure that has found every real defect in this corpus's tooling.

The script refuses rather than reports when labels and vectors share no
clip_id: a keying mismatch scores 0.0 and looks identical to a model that
cannot tell two people apart.

This closes spec UNKNOWN 1 and is explicitly NOT a precision claim. Those
labels are a dev set per the eval design's 3.4; learning from them is what a
dev set is for, quoting a precision figure off them is what it forbids.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KhAPYARf38HUujE4PfF4G
EOF
```

---

## Task 5: The archive audio pass

The irreversible one. Dry run first; the real run is detached.

**Files:**
- Modify: `speaker_embed.py`

**Interfaces:**
- Consumes: `corpus.embedding.cluster_crop_plan`, `corpus.transcripts.load_episodes`, `speaker_tool.resolve_audio_urls`, `speaker_tool.FEEDS`, `corpus.speakers.GAP_CAP_S`, `corpus.speakers.CLIP_MIN_TURN_S`.
- Produces: Modal function `embed_episode(job: dict) -> list[dict]` returning crop records each carrying a `"vector"`; entrypoint mode `archive`; writes `/vectors/archive_vectors.json` on the `speaker-vectors` volume.

- [ ] **Step 1: Add the archive pass to `speaker_embed.py`**

Append to `speaker_embed.py` (keep the existing contents):

```python
# Capped deliberately, matching cut_one_episode's cap and for the same reason:
# Modal would happily fan out one container per episode, which is 433
# simultaneous downloads pointed at one podcast host -- inconsiderate, and a
# good way to be rate-limited midway through a run whose partial failures then
# look like audio rot.
@app.function(
    image=image,
    gpu="l4",
    volumes={MODEL_PATH: model_cache, VECTORS_PATH: vectors},
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=3600,
    max_containers=8,
)
def embed_episode(job: dict) -> list[dict]:
    """Fetch one episode's audio, embed every crop it owes, discard the audio.

    Returns one record per crop, each with its vector. The audio is never
    persisted: 56.1 GB of storage to avoid a re-download that this design is
    built never to need.
    """
    import tempfile

    import requests

    embedder = load_embedder(job["embedder"], os.environ["HF_TOKEN"])
    out: list[dict] = []

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
        audio_path = handle.name
        with requests.get(job["audio_url"], stream=True, timeout=300) as response:
            response.raise_for_status()
            for block in response.iter_content(chunk_size=1 << 20):
                handle.write(block)

    try:
        for crop in job["crops"]:
            try:
                vector = embedder.embed(
                    audio_path, crop["crop_start_s"], crop["crop_length_s"]
                )
            except Exception as e:  # noqa: BLE001 - recorded, not swallowed
                out.append({**crop, "vector": None, "error": f"{type(e).__name__}: {e}"})
                continue
            out.append({**crop, "vector": vector})
    finally:
        Path(audio_path).unlink(missing_ok=True)

    ok = sum(1 for r in out if r.get("vector") is not None)
    print(f"{job['episode']}: {ok}/{len(out)} crops embedded")
    return out
```

Then replace the `main` entrypoint with one that handles both modes:

```python
@app.local_entrypoint()
def main(
    mode: str = "clips",
    embedder: str = COMMUNITY,
    dry_run: bool = False,
    limit: int = 0,
) -> None:
    if mode == "clips":
        result = embed_clips.remote(embedder)
        slug = embedder.split("/")[-1]
        destination = Path("speakers") / f"clip_vectors_{slug}.json"
        destination.write_text(json.dumps(result), encoding="utf-8")
        print(f"wrote {len(result)} vectors to {destination}")
        return

    if mode != "archive":
        raise SystemExit(f"unknown mode {mode!r}")

    from corpus.embedding import cluster_crop_plan
    from corpus.identity import parse_transcript_filename
    from corpus.speakers import CLIP_MIN_TURN_S, GAP_CAP_S
    from corpus.transcripts import load_episodes
    from speaker_tool import FEEDS, resolve_audio_urls

    episodes = load_episodes(Path("downloaded"), cap=GAP_CAP_S)
    print(f"dir=downloaded files={len(episodes)} gap_cap={GAP_CAP_S}")

    url_maps = {
        show: resolve_audio_urls.local(feed, show) for show, feed in FEEDS.items()
    }

    jobs = []
    unresolved = 0
    unparsed = 0
    for episode in episodes:
        crops = cluster_crop_plan(episode, min_turn_s=CLIP_MIN_TURN_S)
        if not crops:
            continue
        # The repo's ONE episode-number rule (corpus/identity.py:48). A
        # hand-rolled split would be a second derivation that can disagree
        # with the first, which is the defect commit 2ce8264 removed.
        parsed = parse_transcript_filename(episode["name"])
        if parsed is None:
            unparsed += 1
            continue
        _show, number, date_str = parsed
        url = url_maps.get(episode["show"], {}).get(f"{number}|{date_str}")
        if not url:
            unresolved += 1
            continue
        jobs.append(
            {
                "episode": episode["name"],
                "audio_url": url,
                "embedder": embedder,
                "crops": crops,
            }
        )

    total_crops = sum(len(j["crops"]) for j in jobs)
    print(
        f"episodes={len(jobs)} unresolved={unresolved} "
        f"unparsed={unparsed} crops={total_crops}"
    )

    # PRE-REGISTERED. Measured 2026-09-02: 433 fetchable episodes, 970
    # croppable clusters, 2,712 crops at k=3, doubled by the two window
    # lengths. A world that has moved should cause a refusal, not an
    # unknown-size run against someone else's bandwidth.
    if not dry_run and not limit:
        if len(jobs) != 433 or total_crops != 2712 * len(CROP_LENGTHS_S_LOCAL):
            raise SystemExit(
                f"REFUSING: expected 433 episodes and "
                f"{2712 * len(CROP_LENGTHS_S_LOCAL)} crops, planned {len(jobs)} "
                f"and {total_crops}. Re-measure before spending the bandwidth."
            )

    if dry_run:
        print("DRY RUN — nothing downloaded.")
        print("PLAN COMPLETE")
        return

    if limit:
        jobs = jobs[:limit]
        print(f"LIMITED to {len(jobs)} episodes")

    records: list[dict] = []
    for batch in embed_episode.map(jobs):
        records.extend(batch)

    embedded = sum(1 for r in records if r.get("vector") is not None)
    print(f"crops={len(records)} embedded={embedded}")
    Path("speakers/archive_vectors.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    print("ARCHIVE PASS COMPLETE")
```

Add near the other constants, so the assertion reads the same tuple the plan uses:

```python
from corpus.embedding import CROP_LENGTHS_S as CROP_LENGTHS_S_LOCAL
```

- [ ] **Step 2: Dry-run and check the plan against the pre-registered numbers**

```bash
modal run speaker_embed.py --mode archive --dry-run
```

Expected: `episodes=433 unresolved=6 crops=5424` (2,712 crops × 2 window lengths), then `PLAN COMPLETE`. **If the episode count is not 433 or the crop count is not 5424, stop** — refresh `downloaded/` with `MSYS_NO_PATHCONV=1 modal volume get podcast-transcripts / downloaded --force` and re-measure before going further.

- [ ] **Step 3: Smoke-test on three episodes**

```bash
modal run speaker_embed.py --mode archive --limit 3
```

Expected: three per-episode lines reading `N/N crops embedded`, then `ARCHIVE PASS COMPLETE`. A vector length of 256 confirms the embedder matches the bake-off's.

- [ ] **Step 4: Run the full pass, detached**

```bash
modal run --detach speaker_embed.py --mode archive
```

`--detach` because some harnesses kill the local client at ten minutes while the remote function keeps going. **`modal app logs <id>` is not a completion probe** — it returns a window and exits 0 having possibly matched nothing. Completion is `modal app list` showing the app `stopped` with 0 tasks **and** the `ARCHIVE PASS COMPLETE` marker present.

- [ ] **Step 5: Commit**

```bash
git add speaker_embed.py
git commit -F- <<'EOF'
feat: embed the archive once, and refuse if the world moved

One episode per container, capped at 8 -- the same courtesy cut_one_episode
observes toward a single podcast host, and the same protection against a
rate-limit midway through a run whose partial failures then look like rot.

The expected episode and crop counts are asserted inside the script rather
than checked by eye, so a corpus that moved since the measurement causes a
refusal instead of an unknown-size run against someone else's bandwidth.

Audio is fetched, embedded and discarded. A failed crop is recorded with its
error rather than dropped, because a silently shorter vector set would move
every centroid it touches without saying so.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KhAPYARf38HUujE4PfF4G
EOF
```

---

## Task 6: Assignment, conflicts, and the checkpoint histogram

The measurement that decides whether Ship 2 gets built.

**Files:**
- Create: `assign_speakers.py`
- Writes: `speakers/assignments.json` (carrying the (show, episode_number, date) triple Task 8 filters on)

- [ ] **Step 1: Pre-register the Ship 2 threshold, before running anything**

Append to the spec's §1.3 and commit **first**:

> **Checkpoint pre-registration, 2026-09-02.** Ship 2 is built if the
> unmatched centroids yield **at least 15 groups appearing in 5 or more
> distinct episodes**. Below that the recurring-unnamed-voice population is
> too thin to justify archive-wide grouping, and Ship 1's five identities
> stand as the deliverable. Fixed before the histogram is seen.

- [ ] **Step 2: Write the assignment script**

Create `assign_speakers.py`:

```python
"""Assign each cluster to an enrolled voice, then count what is left over.

Two outputs. The assignments Ship 1 backfills, and the histogram that decides
whether Ship 2 is worth building -- which is the question the second half of
the design assumes and nobody has measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from corpus.embedding import weighted_centroid
from corpus.enrolment import assign, build_centroids, co_membership_conflicts
from corpus.identity import parse_transcript_filename

THRESHOLD = 0.55
MARGIN = 0.05


def centroids_by_cluster(records: list[dict]) -> dict[tuple[str, str], list[float]]:
    """One centroid per (episode, cluster), from that cluster's crops.

    Weighted by TRUE turn duration, not by crop length -- the distinction
    corpus/embedding.py exists to keep.
    """
    grouped: dict[tuple[str, str], list[tuple[list[float], float]]] = {}
    for record in records:
        if record.get("vector") is None:
            continue
        key = (record["episode"], record["cluster"])
        grouped.setdefault(key, []).append(
            (record["vector"], record["turn_duration_s"])
        )
    return {
        key: weighted_centroid([v for v, _ in items], [w for _, w in items])
        for key, items in grouped.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign clusters to enrolled voices and size Ship 2."
    )
    parser.add_argument(
        "clip_vectors",
        help="the winning speakers/clip_vectors_<embedder>.json from Task 4",
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--margin", type=float, default=MARGIN)
    args = parser.parse_args()

    labels = json.loads(Path("speakers/labels.json").read_text(encoding="utf-8"))
    records = json.loads(
        Path("speakers/archive_vectors.json").read_text(encoding="utf-8")
    )
    clip_vectors = json.loads(
        Path(args.clip_vectors).read_text(encoding="utf-8")
    )

    centroids = build_centroids(labels, clip_vectors)
    print(f"enrolled={sorted(centroids)}")
    if not centroids:
        print("REFUSING: no enrolment centroids. Check the clip vector keys.")
        return 2

    clusters = centroids_by_cluster(records)
    print(f"clusters with a centroid: {len(clusters)}")

    assignments = []
    for (episode, cluster), vector in sorted(clusters.items()):
        parsed = parse_transcript_filename(episode)
        if parsed is None:
            print(f"REFUSING: cannot parse episode name {episode!r}")
            return 2
        show, episode_number, date_str = parsed
        assignments.append(
            {
                "episode": episode,
                # The triple the backfill filters on. Chunk metadata has NO
                # `episode` key (corpus/chunking.py:66-80); an episode is
                # (show, episode_number, date), and carrying it here means
                # Task 8 never re-derives it.
                "show": show,
                "episode_number": episode_number,
                "date": date_str,
                "cluster": cluster,
                "person": assign(
                    vector,
                    centroids,
                    threshold=args.threshold,
                    margin=args.margin,
                ),
            }
        )

    matched = [a for a in assignments if a["person"] is not None]
    print(f"assigned={len(matched)} of {len(assignments)}")
    for person, count in Counter(a["person"] for a in matched).most_common():
        print(f"  {person}: {count} clusters")

    conflicts = co_membership_conflicts(assignments)
    print(f"\nco-membership conflicts: {len(conflicts)}")
    for conflict in conflicts[:20]:
        print(f"  {conflict['episode']}: {conflict['person']} {conflict['clusters']}")

    # --- the checkpoint -------------------------------------------------
    unmatched = [a for a in assignments if a["person"] is None]
    print(f"\nunmatched clusters: {len(unmatched)}")
    print(
        "Ship 2's population is measured by clustering these; run the "
        "histogram step next and compare against the pre-registered "
        "threshold of 15 groups in 5+ episodes."
    )

    Path("speakers/assignments.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "margin": args.margin,
                "assignments": assignments,
                "conflicts": conflicts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("ASSIGN COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run it and read the conflicts**

```bash
py assign_speakers.py speakers/clip_vectors_<winner>.json
```

Expected: `enrolled=['Jacob Shapiro', 'Marko Papic', 'Matt Gertken', 'Rob Larity', 'Tobias Harris']`, a per-person cluster count, a conflict list, and `ASSIGN COMPLETE`.

**Read the conflicts before proceeding.** Each is either diarisation over-segmenting one person (correct to merge) or two people collapsing into one identity (the error that matters). If conflicts exceed 5% of matched clusters, stop and raise the threshold rather than backfilling.

- [ ] **Step 4: Commit**

```bash
git add assign_speakers.py speakers/assignments.json docs/superpowers/specs/2026-09-02-speaker-grouping-design.md
git commit -F- <<'EOF'
feat: assign clusters to enrolled voices, and count what is left

Assignment is threshold AND margin, so a cluster that resembles two enrolled
voices equally gets neither name. The co-membership conflicts are printed
before anything is written, because each one is either diarisation
over-segmenting a person (correct to merge) or two people collapsing into one
identity (the error with no external symptom).

The leftover count is the checkpoint. Ship 2's threshold -- 15 groups in 5 or
more episodes -- was pre-registered and committed before this ran.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KhAPYARf38HUujE4PfF4G
EOF
```

---

## Task 7: The output contract — four render states

Ships **before** the backfill, so the consumer is ready when identities land.

**Files:**
- Modify: `corpus/rendering.py`
- Modify: `tests/test_rendering.py`
- Modify: `mcp_server.py:120-146` (result dict), `mcp_server.py:194-231` (`search_podcasts`)

**Interfaces:**
- Produces: `format_result(result: dict, *, include_relevance: bool = True, names: dict[str, str] | None = None) -> str`; `display_speaker(result: dict, names: dict[str, str] | None) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rendering.py`:

```python
from corpus.rendering import display_speaker, format_result

BASE = {
    "text": "[SPEAKER_01] the thing he said",
    "speaker": "SPEAKER_01",
    "date": "2026-01-01",
    "show": "The Jacob Shapiro Podcast",
    "episode_number": "300",
    "episode_title": "A Title",
    "start_time": 12.0,
    "relevance_score": 0.9,
}
NAMES = {"Rob Larity": "Rob Larity"}


def test_named_identity_renders_the_name():
    result = {**BASE, "speaker_id": "Rob Larity"}
    assert display_speaker(result, NAMES) == "Rob Larity"


def test_unnamed_group_never_renders_a_bare_token():
    # An opaque grp_007 alone is a token an LLM consumer will invent meaning
    # for. It appears only as a parenthetical after words that say what it is.
    result = {**BASE, "speaker_id": "grp_007"}
    rendered = display_speaker(result, {})
    assert "grp_007" in rendered
    assert rendered != "grp_007"
    assert "unidentified voice" in rendered


def test_absent_speaker_id_renders_unidentified():
    assert display_speaker(BASE, NAMES) == "unidentified"


def test_unknown_speaker_renders_unidentified():
    result = {**BASE, "speaker": "UNKNOWN", "text": "[UNKNOWN] words"}
    assert display_speaker(result, NAMES) == "unidentified"


def test_prefix_is_stripped_with_the_cluster_label_not_the_display_name():
    # THE trap: chunking.py bakes [SPEAKER_01] into the document and the strip
    # must key off the raw cluster label. Stripping with "Rob Larity" would
    # leave the cluster label visible in the text it was meant to replace.
    result = {**BASE, "speaker_id": "Rob Larity"}
    rendered = format_result(result, names=NAMES)
    assert "[SPEAKER_01]" not in rendered
    assert "speaker: Rob Larity" in rendered
    assert "the thing he said" in rendered


def test_rendering_is_unchanged_when_no_speaker_id_exists_anywhere():
    # Every record in Chroma is in this state until the backfill runs, so this
    # is the compatibility test: the change must be inert before it is useful.
    assert format_result(BASE) == format_result(BASE, names={})
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_rendering.py -v
```

Expected: `ImportError: cannot import name 'display_speaker'`.

- [ ] **Step 3: Implement**

In `corpus/rendering.py`, add below `UNKNOWN_SPEAKER`:

```python
# What a consumer reads when nothing identifies the voice. Two different
# causes -- assignment declined, and no label ever existed -- render
# identically on purpose: the distinction is real but is not the reader's
# business, and inventing two phrasings would imply a difference they could
# act on.
UNIDENTIFIED = "unidentified"


def display_speaker(result: dict, names: dict[str, str] | None) -> str:
    """The identity a caller reads, from `speaker_id` and the names map.

    Never returns a bare group id. An opaque `grp_007` alone is a token an LLM
    consumer will invent meaning for, so it appears only as a parenthetical
    after words that say what it is.
    """
    speaker_id = result.get("speaker_id")
    if not speaker_id:
        return UNIDENTIFIED
    resolved = (names or {}).get(speaker_id)
    if resolved:
        return resolved
    return f"{UNIDENTIFIED} voice ({speaker_id})"
```

Then change `format_result`:

```python
def format_result(
    result: dict,
    *,
    include_relevance: bool = True,
    names: dict[str, str] | None = None,
) -> str:
    """One search hit, as the caller reads it.

    The prefix is stripped with the RAW CLUSTER LABEL, never the display name.
    `chunking.py:84` bakes `[SPEAKER_01] ` into the document text, so keying
    the strip off a resolved name would leave the cluster label visible in the
    very text the name was meant to replace.
    """
    cluster = result.get("speaker") or UNKNOWN_SPEAKER
    text = _strip_speaker_prefix(result.get("text", ""), cluster)
    speaker = display_speaker(result, names)

    header = (
        f"[{result['date']}] {result['show']} - "
        f"Episode {result['episode_number']} ({result['episode_title']}) "
        f"@ {result['start_time']:.1f}s | speaker: {speaker}"
    )
    if include_relevance:
        header += f" [relevance: {result['relevance_score']}]"

    return f"{header}\n{text}\n"
```

And `format_results`:

```python
def format_results(
    results: list[dict],
    *,
    include_relevance: bool = True,
    names: dict[str, str] | None = None,
) -> str:
    """The whole response body."""
    return "\n---\n".join(
        format_result(r, include_relevance=include_relevance, names=names)
        for r in results
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_rendering.py -v
uv run ruff check corpus/rendering.py tests/test_rendering.py
uv run mypy
```

Expected: all pass, including the pre-existing rendering tests — the compatibility test proves the change is inert before the backfill.

- [ ] **Step 5: Wire the MCP layer**

In `mcp_server.py`, add `"speaker_id": meta.get("speaker_id"),` to the result dict beside `"speaker"` (around line 131), and add a `speaker` filter to `search_podcasts`:

```python
    def search_podcasts(
        query: str,
        n_results: int = 10,
        show: str = None,
        after_date: str = None,
        before_date: str = None,
        speaker: str = None,
    ) -> str:
        """
        Search podcast transcripts semantically. Returns relevant chunks with
        speaker attribution, episode metadata, and publication date.
        Use this whenever the user asks about topics, opinions, or statements
        from the podcasts. Results are sorted most-recent first so recency is
        always visible — factor in publication date when reasoning about whether
        views may have changed.

        Speaker identity spans shows: filtering by a person returns them
        wherever they appear, including as a guest on another feed, which is
        what makes "has this person's view changed" answerable.

        Args:
            query: What to search for, e.g. "China Taiwan" or "Fed interest rates"
            n_results: Number of results to return (default 10)
            show: Filter to a specific show name (optional)
            after_date: Only include episodes after this date, YYYY-MM-DD (optional)
            before_date: Only include episodes before this date, YYYY-MM-DD (optional)
            speaker: Filter to one person, e.g. "Jacob Shapiro" (optional)
        """
```

Pass `speaker=speaker` into `searcher.search.remote(...)`, and in `search` add the clause beside the existing filters:

```python
        if speaker:
            filters.append({"speaker_id": {"$eq": speaker}})
```

**The names map ships as `corpus/names.py`, not `speakers/names.json`, and is
read in the ASGI factory — both of those are corrections, each with a reason:**

- **Not a new data file.** `mcp_server.py:32` ends the image chain with
  `.add_local_python_source("corpus")` and its own comment says it must stay
  LAST because a rebuild there can silently change the host-header behaviour
  documented in `mcp_server()`. Appending an `add_local_file` for
  `speakers/names.json` would violate that and risk exactly the rebuild the
  comment warns about. A module inside the already-mounted `corpus` package
  ships for free and changes no image.
- **Not in `load()`.** `load()` is `@modal.enter()` on the `PodcastSearch`
  class (`mcp_server.py:43-45`), which runs in the **GPU container**.
  `format_results` is called in the **ASGI process** (`mcp_server()` at
  `:163`), so `self.names` is unreachable from where the rendering happens.

Create `corpus/names.py`:

```python
"""Group id to display name. The ONE place a name is written down.

Derived, never duplicated: Chroma holds a stable `speaker_id` and this maps it
to something a person reads, so naming a group is an edit here and nothing
else -- no Chroma write, no re-embed, and it propagates across all 433
episodes at once.

Ship 1's identities are people, so today this is close to an identity map. It
exists now anyway because Ship 2 writes `grp_NNN` keys into the same table and
the render path must not change shape between ships.
"""

from __future__ import annotations

NAMES: dict[str, str] = {
    "Jacob Shapiro": "Jacob Shapiro",
    "Rob Larity": "Rob Larity",
    "Marko Papic": "Marko Papic",
    "Tobias Harris": "Tobias Harris",
    "Matt Gertken": "Matt Gertken",
}
```

Then in `mcp_server.py`, import it beside the existing render import at line 4:

```python
from corpus.names import NAMES
from corpus.rendering import format_results
```

**and patch BOTH call sites — there are two.** `search_podcasts` at
`mcp_server.py:231` and `latest_on_topic` at `:252`. Patching only the first
leaves every `latest_on_topic` result rendering `unidentified`, which looks
like a backfill that failed rather than a render that was missed:

```python
        return format_results(results, names=NAMES)          # :231
        return format_results(results, include_relevance=False, names=NAMES)  # :252
```

- [ ] **Step 6: Commit**

```bash
git add corpus/rendering.py corpus/names.py tests/test_rendering.py mcp_server.py
git commit -F- <<'EOF'
feat: render four speaker states, and strip with the cluster label

The deliverable is identity in MCP results, so the output line is part of the
design rather than a detail. Four states are now specified and tested: named
identity, unnamed group, assignment declined, and never labelled -- the last
two rendering identically on purpose, since the distinction is real but is not
the reader's business.

A bare grp_007 is never emitted alone. An opaque token is one an LLM consumer
will invent meaning for, so it only ever appears parenthetically after words
that say what it is.

The prefix strip keys off the RAW CLUSTER LABEL, never the display name.
chunking.py:84 bakes [SPEAKER_01] into the document text, so stripping with a
resolved name would leave the cluster label visible in the text the name was
meant to replace -- the defect that shaped the whole additive-metadata design.

Ships before the backfill and is inert until it runs, which the compatibility
test pins.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KhAPYARf38HUujE4PfF4G
EOF
```

---

## Task 8: The backfill

Metadata only, additive, counts pre-registered.

**Files:**
- Create: `backfill_speaker_id.py`
- Create: `tests/test_backfill.py`

**Interfaces:**
- Consumes: `tests/conftest.py::FakeCollection` (models Cloud's caps and its **merge-on-upsert** metadata behaviour), `corpus.store.episode_where(show, episode_number, date_str)`, `corpus.identity.parse_transcript_filename(name) -> tuple[show, ep, date] | None`, `transcribe.get_chroma_collection(api_key, tenant, database)`.
- Produces: `plan_updates(collection, assignments: list[dict]) -> list[tuple[str, dict]]`; `apply_updates(collection, updates, *, expected: int) -> int`; Modal function `run_backfill(apply: bool) -> dict`.

> **VERIFIED BEFORE WRITING THIS TASK, because the first draft got both wrong
> and would have failed silently.**
>
> - **There is no `episode` metadata key.** `corpus/chunking.py:66-80` writes
>   `show`, `episode_number`, `episode_title`, `date`, `speaker`,
>   `start_time`, `date_ts`, `rules_version`, and `episode_guid` only when it
>   is not None. An episode is addressed by the **triple**, and
>   `corpus/store.py:53` already provides `episode_where` for it. A filter on
>   `{"episode": ...}` matches nothing, plans **zero** updates, and reports
>   success — the worst possible failure mode for a backfill.
> - **`mcp_server.py` has no importable collection helper.** It constructs a
>   `chromadb.CloudClient` inside the Modal class's `load()`
>   (`mcp_server.py:45-77`). The importable one is
>   `transcribe.py:137 get_chroma_collection(chroma_api_key, chroma_tenant,
>   chroma_database)` — three arguments, from the Modal secret. **So this
>   backfill runs as a Modal function**, not as a local script: the Chroma
>   credentials live in `podcast-secrets` and are not on the developer's
>   machine.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backfill.py`:

```python
"""The backfill writes ONE new metadata key and nothing else.

Chroma merges metadata on upsert (measured on Cloud, see conftest), so a key
this writes cannot later be removed by rewriting the episode -- only by
deleting the record. That makes "add, never overwrite" a correctness
requirement rather than a style preference.
"""

from __future__ import annotations

import pytest

from backfill_speaker_id import apply_updates, plan_updates


SHOW = "The Jacob Shapiro Podcast"
EP1 = f"{SHOW} - Episode 1 - 2022-05-02.txt"
EP2 = f"{SHOW} - Episode 2 - 2022-05-06.txt"


@pytest.fixture
def seeded(collection):
    # The metadata written by corpus/chunking.py:66-80. There is NO "episode"
    # key -- an episode is the (show, episode_number, date) triple, which is
    # what corpus/store.py:53 episode_where selects on.
    collection.upsert(
        ids=["ep1::0", "ep1::1", "ep2::0"],
        documents=["[SPEAKER_00] a", "[SPEAKER_01] b", "[SPEAKER_00] c"],
        metadatas=[
            {
                "show": SHOW,
                "episode_number": "1",
                "date": "2022-05-02",
                "speaker": "SPEAKER_00",
            },
            {
                "show": SHOW,
                "episode_number": "1",
                "date": "2022-05-02",
                "speaker": "SPEAKER_01",
            },
            {
                "show": SHOW,
                "episode_number": "2",
                "date": "2022-05-06",
                "speaker": "SPEAKER_00",
            },
        ],
    )
    return collection


def _record(episode, number, date, cluster, person):
    return {
        "episode": episode,
        "show": SHOW,
        "episode_number": number,
        "date": date,
        "cluster": cluster,
        "person": person,
    }


ASSIGNMENTS = [
    _record(EP1, "1", "2022-05-02", "SPEAKER_00", "Jacob Shapiro"),
    _record(EP1, "1", "2022-05-02", "SPEAKER_01", None),
    _record(EP2, "2", "2022-05-06", "SPEAKER_00", "Rob Larity"),
]


def test_an_assignment_without_the_triple_raises():
    # The defect this test exists for, and the one that broke the first draft
    # of this task: a filter that matches nothing plans zero updates and
    # reports success. Refuse instead.
    from backfill_speaker_id import plan_updates as plan

    with pytest.raises(ValueError, match="missing"):
        plan(None, [{"episode": "x.txt", "cluster": "SPEAKER_00", "person": "X"}])


def test_plans_one_update_per_matching_chunk(seeded):
    updates = plan_updates(seeded, ASSIGNMENTS)
    assert {chunk_id for chunk_id, _ in updates} == {"ep1::0", "ep2::0"}


def test_declined_clusters_get_no_key_at_all(seeded):
    # person: null is ground truth. Writing "unknown" would be a value a
    # filter could match, which is exactly the false confidence to avoid.
    updates = plan_updates(seeded, ASSIGNMENTS)
    assert "ep1::1" not in {chunk_id for chunk_id, _ in updates}


def test_the_update_carries_only_speaker_id(seeded):
    updates = plan_updates(seeded, ASSIGNMENTS)
    for _, metadata in updates:
        assert set(metadata) == {"speaker_id"}


def test_speaker_is_never_overwritten(seeded):
    apply_updates(seeded, plan_updates(seeded, ASSIGNMENTS), expected=2)
    got = seeded.get(ids=["ep1::0"], include=["metadatas"])
    assert got["metadatas"][0]["speaker"] == "SPEAKER_00"
    assert got["metadatas"][0]["speaker_id"] == "Jacob Shapiro"


def test_documents_are_not_rewritten(seeded):
    # HONEST ABOUT ITS OWN REACH. FakeCollection.upsert preserves a document
    # by construction when `documents` is None, so this pins the CALLER (it
    # passes no documents) and cannot prove Cloud behaves the same way. The
    # real behaviour of a metadata-only upsert against Cloud is spec UNKNOWN
    # 6.3 and is measured in Step 5 against a scratch collection, not here.
    before = seeded.get(ids=["ep1::0"], include=["documents"])["documents"][0]
    apply_updates(seeded, plan_updates(seeded, ASSIGNMENTS), expected=2)
    after = seeded.get(ids=["ep1::0"], include=["documents"])["documents"][0]
    assert before == after == "[SPEAKER_00] a"


def test_refuses_when_the_count_moved(seeded):
    # A collection that changed since the plan was made must cause a refusal,
    # not an unknown-size write.
    with pytest.raises(RuntimeError, match="expected 99"):
        apply_updates(seeded, plan_updates(seeded, ASSIGNMENTS), expected=99)


def test_batches_stay_under_the_cloud_cap(seeded):
    for i in range(700):
        seeded.upsert(
            ids=[f"big::{i}"],
            documents=["[SPEAKER_00] x"],
            metadatas=[
                {
                    "show": SHOW,
                    "episode_number": "999",
                    "date": "2026-01-01",
                    "speaker": "SPEAKER_00",
                }
            ],
        )
    big = f"{SHOW} - Episode 999 - 2026-01-01.txt"
    many = plan_updates(
        seeded, [_record(big, "999", "2026-01-01", "SPEAKER_00", "Jacob Shapiro")]
    )
    # Both the paging in plan_updates and the batching in apply_updates are
    # under test: get() silently truncates at 300 and upsert() raises above it.
    assert len(many) == 700
    assert apply_updates(seeded, many, expected=700) == 700
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_backfill.py -v
```

Expected: `ModuleNotFoundError: No module named 'backfill_speaker_id'`.

- [ ] **Step 3: Implement**

Create `backfill_speaker_id.py`:

```python
"""Write `speaker_id` into Chroma. Add a key; never touch another.

WHY ADDITIVE IS A CORRECTNESS REQUIREMENT, not a preference. `chunking.py:84`
bakes `[SPEAKER_01] ` into the document text and `rendering.py` strips it
using the metadata value, so overwriting `speaker` with a person's name leaves
the raw cluster label visible in the text it was meant to replace. And per
`corpus/writing.py`, Chroma MERGES metadata on upsert -- a key written here
can never be removed by rewriting the episode, only by deleting the record.
Both point the same way: add `speaker_id`, leave everything else alone.

This never writes a document and never writes an embedding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from corpus.store import BATCH, episode_where

# BATCH comes from corpus/store.py:20 (250), under a measured Cloud cap of
# 300. A second constant that happened to differ would be a silent
# inconsistency between two writers against the same collection.


def plan_updates(collection, assignments: list[dict]) -> list[tuple[str, dict]]:
    """Every (chunk_id, {"speaker_id": person}) this backfill would write.

    A cluster with `person: None` yields NOTHING. Writing a placeholder would
    create a value a filter could match, which is the false confidence the
    whole refusal rule exists to avoid.
    """
    updates: list[tuple[str, dict]] = []
    for record in assignments:
        missing = {"show", "episode_number", "date"} - set(record)
        if missing:
            # Refuse rather than skip. A record without the triple would build
            # a filter that selects nothing, plan zero updates, and report
            # success -- the failure mode with no symptom, and the one that
            # broke the first draft of this task.
            raise ValueError(
                f"assignment is missing {sorted(missing)}: {record.get('episode')!r}"
            )
        person = record.get("person")
        if person is None:
            continue
        where = episode_where(
            record["show"], record["episode_number"], record["date"]
        )
        where = {"$and": [*where["$and"], {"speaker": {"$eq": record["cluster"]}}]}
        offset = 0
        while True:
            got = collection.get(
                where=where,
                limit=BATCH,
                offset=offset,
                include=["metadatas"],
            )
            ids = got["ids"]
            if not ids:
                break
            updates.extend((chunk_id, {"speaker_id": person}) for chunk_id in ids)
            if len(ids) < BATCH:
                break
            offset += BATCH
    return updates


def apply_updates(collection, updates: list[tuple[str, dict]], *, expected: int) -> int:
    """Write the planned metadata, refusing if the count moved.

    The count is asserted rather than trusted: a collection that changed since
    the plan was built should cause a refusal, not an unknown-size write.
    """
    if len(updates) != expected:
        raise RuntimeError(
            f"REFUSING: expected {expected} updates, planned {len(updates)}. "
            f"The collection moved since the plan was built; re-plan rather "
            f"than write."
        )
    written = 0
    for start in range(0, len(updates), BATCH):
        batch = updates[start : start + BATCH]
        collection.upsert(
            ids=[chunk_id for chunk_id, _ in batch],
            metadatas=[metadata for _, metadata in batch],
        )
        written += len(batch)
    return written


# Runs ON MODAL, not locally. The Chroma credentials live in the
# `podcast-secrets` Modal secret and are not on the developer's machine, and
# `mcp_server.py` exposes no importable collection helper -- it builds a
# CloudClient inside its Modal class's load(). The importable one is
# transcribe.py:137, which takes three arguments.
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "chromadb==1.5.9"
).add_local_python_source("corpus")

app = modal.App("speaker-backfill")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=1800,
)
def run_backfill(assignments: list[dict], apply: bool) -> dict:
    from transcribe import get_chroma_collection

    collection = get_chroma_collection(
        os.environ["CHROMA_API_KEY"],
        os.environ["CHROMA_TENANT"],
        os.environ["CHROMA_DATABASE"],
    )
    print(f"collection holds {collection.count()} records")

    updates = plan_updates(collection, assignments)
    print(f"planned {len(updates)} chunk updates")
    if not apply:
        print("DRY RUN — nothing written.")
        print("BACKFILL PLAN COMPLETE")
        return {"planned": len(updates), "written": 0}

    written = apply_updates(collection, updates, expected=len(updates))
    print(f"wrote {written}")
    print("BACKFILL COMPLETE")
    return {"planned": len(updates), "written": written}


@app.local_entrypoint()
def main(apply: bool = False) -> None:
    payload = json.loads(
        Path("speakers/assignments.json").read_text(encoding="utf-8")
    )
    print(run_backfill.remote(payload["assignments"], apply))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_backfill.py -v
uv run ruff check backfill_speaker_id.py tests/test_backfill.py
uv run mypy
```

Expected: 7 passed.

- [ ] **Step 5: Settle UNKNOWN 6.3 against a scratch Cloud collection FIRST**

The unit tests pin what this code *sends*; they cannot pin what Chroma *does*
with it, because `FakeCollection` preserves a document by construction. The
one behaviour the whole additive design rests on — that a metadata-only
upsert leaves the document and the embedding untouched — is still unmeasured
against the real service. Measure it on a throwaway collection before touching
the live one:

```python
# scratch_metadata_probe.py, run on Modal with podcast-secrets
collection.upsert(
    ids=["probe::0"],
    documents=["[SPEAKER_00] hello"],
    metadatas=[{"show": "S", "episode_number": "1", "date": "2026-01-01"}],
)
before = collection.get(ids=["probe::0"], include=["documents", "embeddings"])
collection.upsert(ids=["probe::0"], metadatas=[{"speaker_id": "Someone"}])
after = collection.get(ids=["probe::0"], include=["documents", "embeddings"])
print("document preserved:", before["documents"] == after["documents"])
print("embedding preserved:", before["embeddings"] == after["embeddings"])
print("metadata merged:", after["metadatas"][0])
```

**If the document or the embedding does not survive, STOP.** The backfill would
then be destructive and Task 8 needs redesigning around `collection.update()`
instead. Record the result in the spec's §6.3 either way — this closes the
UNKNOWN rather than sidestepping it.

- [ ] **Step 6: Dry-run against the real collection, then apply**

```bash
modal run backfill_speaker_id.py
modal run backfill_speaker_id.py --apply
```

Between the two: check the planned count is plausible against the ~30k chunks in the collection and the per-person cluster counts from Task 6. A planned count of zero now means the (show, episode_number, date, speaker) filter matched nothing — **investigate rather than re-running with `--apply`.**

- [ ] **Step 7: Verify end to end through the real consumer**

Query the deployed MCP for a topic Jacob Shapiro discusses and confirm results read `speaker: Jacob Shapiro` with no `[SPEAKER_NN]` prefix leaking into the text, then repeat with the `speaker` filter set.

- [ ] **Step 8: Commit**

```bash
git add backfill_speaker_id.py tests/test_backfill.py
git commit -F- <<'EOF'
feat: backfill speaker_id, adding a key and touching nothing else

Additive is a correctness requirement here, not a preference. chunking.py
bakes [SPEAKER_01] into the document text and rendering.py strips it using the
metadata value, so overwriting speaker leaves the cluster label visible in the
text the name was meant to replace. And per corpus/writing.py Chroma MERGES
metadata on upsert, so a key written here cannot be removed by rewriting the
episode -- only by deleting the record.

A declined cluster gets no key at all. A placeholder would be a value a filter
could match, which is the false confidence the refusal rule exists to prevent.

The planned count is asserted before the write, so a collection that moved
since planning causes a refusal rather than an unknown-size operation.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KhAPYARf38HUujE4PfF4G
EOF
```

---

## Self-Review

**Spec coverage.** §0.4's exclusion rule → Task 1. §2.1's crop plan and two window lengths → Tasks 1 and 5. §2.2's additive metadata → Task 8. §2.3's output contract → Task 7. §2.4's backfill and pre-registered counts → Task 8. §3.1's threshold-and-margin and `null`-as-truth → Tasks 2, 6, 8. §3.3's embedder question → Tasks 3 and 4. §4.1b's co-membership flag → Tasks 2 and 6. §1.3's checkpoint → Task 6. §6.1 closed by Task 4; §6.6 closed by Task 6.

**Deliberate gaps, all Ship 2 (spec §1.3):** `corpus/grouping.py`, `speaker_tool.py`'s `groups`/`name`/`gate-plan`/`gate-score`, retiring the `who is this?` prompt at `speaker_tool.py:691`, and §4.2's 45-pair recognition gate. None is buildable before Task 6's histogram says whether Ship 2 happens.

**Still open and not silently closed:** §6.2 (the 44 clusters below the crop floor) is untouched — they simply produce no crops. §6.3 is **measured in Task 8 Step 5** against a scratch Cloud collection, because the unit tests can only pin what the caller sends and `FakeCollection` preserves documents by construction. §6.4 (whether community-1 is gated) and the `README.md:53-55` correction are untouched and remain recorded in the spec.

**Type consistency.** `weighted_centroid(vectors, weights)` is called with the same two-positional shape in `enrolment.build_centroids` and `assign_speakers.centroids_by_cluster`. `assign(vector, centroids, *, threshold, margin)` keeps its keyword-only form in Tasks 4 and 6. `format_result`'s new `names` parameter is keyword-only and defaulted, so every existing caller keeps working — which the compatibility test in Task 7 pins.

**One risk worth naming.** Task 6's `THRESHOLD = 0.55` and `MARGIN = 0.05` are the only unmeasured numbers in this plan; the spec leaves them deliberately unset (§6.7) because they must come from the observed score distribution. Task 4's bake-off produces that distribution, so **set them from its output before running Task 6**, and record the values in the spec when you do.
