"""THROWAWAY PROBE — settles Part 7 UNKNOWNs 5 and 6 of the speaker-id spec.

Two questions, one CPU container, no GPU:

  5. Does the Hugging Face acceptance recorded in README.md reach an EMBEDDING
     model, or only the two diarisation models named there?
  6. What does the DEPLOYED image actually expose? pyannote arrives
     transitively via whisperx==3.8.5, unpinned and frozen at build time.

The image spec below is copied VERBATIM from transcribe.py:9-27 so Modal's
content-addressing reuses the cache. Per docs/operations.md, build lines in
the output mean the spec has drifted and none of these numbers are the
deployed ones. That is the pre-registered tell; check for it first.

Every result is classified PRESENT / GATED / ABSENT / UNKNOWN, and the last
two are never collapsed: "the import failed" and "the download was refused"
send you to different fixes.

Run:  modal run probe_embedding.py
"""

import modal

# --- VERBATIM from transcribe.py:9-27. Do not tidy. ---------------------
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
# ------------------------------------------------------------------------

app = modal.App("speaker-embedding-probe", image=image)

model_cache = modal.Volume.from_name("whisperx-model-cache", create_if_missing=True)
MODEL_PATH = "/models"

# Candidates, in the order they would be preferred. The first is whatever the
# working diarisation pipeline already uses -- if that is reachable, the
# gating question is closed by a model already in production.
CANDIDATES = [
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    "pyannote/embedding",
]


def classify(exc: BaseException) -> tuple[str, str]:
    """Map an exception to (verdict, why). Never returns a bare failure.

    GATED and ABSENT are different worlds: one needs a click on a web page,
    the other needs a different image. Collapsing them into "it didn't work"
    is how a probe licenses the wrong fix.
    """
    name = type(exc).__name__
    text = f"{name}: {exc}"
    low = text.lower()

    if name in ("ImportError", "ModuleNotFoundError"):
        return "ABSENT", text
    if name in ("GatedRepoError", "RepositoryNotFoundError"):
        return "GATED", text
    if "401" in text or "403" in text:
        return "GATED", text
    if "gated" in low or "awaiting approval" in low or "accept the conditions" in low:
        return "GATED", text
    if "authoriz" in low or "authentic" in low:
        return "GATED", text
    return "UNKNOWN", text


@app.function(
    volumes={MODEL_PATH: model_cache},
    secrets=[modal.Secret.from_name("podcast-secrets")],
    timeout=1800,
)
def probe() -> None:
    import os
    import re
    import traceback

    print("=" * 70)
    print("SECTION 0 — what this image is")
    print("=" * 70)

    import importlib

    for mod in (
        "torch",
        "torchaudio",
        "whisperx",
        "pyannote.audio",
        "huggingface_hub",
        "transformers",
        "speechbrain",
        "onnxruntime",
    ):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "(no __version__)")
            print(f"  PRESENT  {mod:<22} {ver}")
        except Exception as e:  # noqa: BLE001 - reporting, not handling
            verdict, why = classify(e)
            print(f"  {verdict:<8} {mod:<22} {why}")

    hf_token = os.environ.get("HF_TOKEN")
    print(f"\n  HF_TOKEN present: {bool(hf_token)}  len={len(hf_token or '')}")
    try:
        print(f"  {MODEL_PATH} holds: {sorted(os.listdir(MODEL_PATH))[:20]}")
    except Exception as e:  # noqa: BLE001
        print(f"  {MODEL_PATH} unreadable: {e}")

    print()
    print("=" * 70)
    print("SECTION 1 — POSITIVE CONTROL: the pipeline that already works")
    print("=" * 70)
    print("Same construction as transcribe.py:415, on CPU. If this fails,")
    print("every verdict below is UNKNOWN rather than negative.")

    pipeline = None
    try:
        from whisperx.diarize import DiarizationPipeline

        diarize = DiarizationPipeline(token=hf_token, device="cpu")
        print("  CONTROL OK — token, network and image all work.")
        pipeline = getattr(diarize, "model", None)
        print(f"  whisperx wrapper: {type(diarize).__name__}")
        print(f"  inner pipeline:   {type(pipeline).__name__ if pipeline else None}")
    except Exception as e:  # noqa: BLE001
        verdict, why = classify(e)
        print(f"  CONTROL FAILED [{verdict}] — {why}")
        traceback.print_exc()

    print()
    print("=" * 70)
    print("SECTION 2 — does the working pipeline ALREADY carry an embedder?")
    print("=" * 70)
    print("pyannote diarisation clusters by voice, so it must embed. If that")
    print("embedder is reachable, UNKNOWN 5 is closed by a model already in")
    print("production, already cached, already accepted.")

    embedder = None
    if pipeline is not None:
        # Redact before printing: pipeline configs carry the HF token inline,
        # and a probe that writes a secret into its own log is a new problem.
        def _redact(text: str) -> str:
            return re.sub(r"hf_[A-Za-z0-9]+", "hf_<REDACTED>", text)

        attrs = sorted(a for a in vars(pipeline) if not a.startswith("__"))
        print(f"  pipeline attributes: {attrs}")
        for attr in ("embedding", "_embedding", "embedding_model", "_segmentation"):
            val = getattr(pipeline, attr, None)
            if val is not None:
                print(f"  .{attr}: {type(val).__name__}  {_redact(str(val))[:200]}")
                if "embedding" in attr and embedder is None:
                    embedder = val
        for cfg in ("_params", "params", "config"):
            val = getattr(pipeline, cfg, None)
            if val is not None:
                print(f"  .{cfg}: {_redact(str(val))[:400]}")
    else:
        print("  SKIPPED — no pipeline to introspect (control failed).")

    print()
    print("=" * 70)
    print("SECTION 3 — can each embedding candidate be LOADED?")
    print("=" * 70)

    loaded = {}
    for repo in CANDIDATES:
        try:
            from pyannote.audio import Model

            m = Model.from_pretrained(repo, use_auth_token=hf_token)
            loaded[repo] = m
            dim = getattr(m, "dimension", "(no .dimension)")
            print(f"  PRESENT  {repo}  dim={dim}  class={type(m).__name__}")
        except Exception as e:  # noqa: BLE001
            verdict, why = classify(e)
            print(f"  {verdict:<8} {repo}  {why[:300]}")

    print()
    print("=" * 70)
    print("SECTION 4 — does it produce a VECTOR? (the API-shape question)")
    print("=" * 70)
    print("A model that loads but whose inference API differs is another late,")
    print("expensive failure. Synthetic audio is enough to prove the shape.")

    if not loaded:
        print("  SKIPPED — nothing loaded. This is UNKNOWN, not 'no embedder'.")
    else:
        import torch

        waveform = torch.randn(1, 16000 * 5) * 0.05  # 5s, the crop length we would use
        for repo, m in loaded.items():
            try:
                from pyannote.audio import Inference

                inference = Inference(m, window="whole")
                vec = inference({"waveform": waveform, "sample_rate": 16000})
                print(f"  OK  {repo}  shape={getattr(vec, 'shape', type(vec))}")
            except Exception as e:  # noqa: BLE001
                verdict, why = classify(e)
                print(f"  {verdict:<8} {repo}  {why[:300]}")

    print()
    print("PROBE COMPLETE")  # terminal marker; absence means it did not finish


@app.local_entrypoint()
def main() -> None:
    probe.remote()
