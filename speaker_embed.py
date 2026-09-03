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
clips_image = _layers.add_local_dir("clips", "/clips").add_local_python_source("corpus")

app = modal.App("speaker-embed")

model_cache = modal.Volume.from_name("whisperx-model-cache", create_if_missing=True)

MODEL_PATH = "/models"

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
        waveform, rate = torchaudio.load(path, frame_offset=offset, num_frames=frames)
        if rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, rate, self.sample_rate)
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
    volumes={MODEL_PATH: model_cache},
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
