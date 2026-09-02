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
        t for t in turns if t["speaker"] == speaker and t["duration_s"] >= min_turn_s
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


def weighted_centroid(vectors: list[list[float]], weights: list[float]) -> list[float]:
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
            raise ValueError(f"inconsistent dimension: {len(vector)} vs {dim}")
        for i, value in enumerate(vector):
            accumulator[i] += value * weight

    mean = [value / total for value in accumulator]
    norm = math.sqrt(sum(value * value for value in mean))
    if norm == 0.0:
        raise ValueError("centroid has zero norm")
    return [value / norm for value in mean]
