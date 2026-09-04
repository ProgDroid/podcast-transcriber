"""Cropping audio and turning it into a speaker vector. ONE definition.

This lives in `corpus/` rather than beside either caller because two things
now need it: the batch archive pass (`speaker_embed.py`) and the nightly
ingest (`transcribe.py`). A second copy of the frame arithmetic below would
be correct on the day it was written and would diverge the first time either
copy was touched -- and the bug this module's comments describe survived a
full 433-episode run precisely because nothing else computed the same thing
to disagree with it. One definition means `tests/test_embed_crop_window.py`
guards both callers.

Import-light on purpose: `torchaudio` and `pyannote.audio` are imported
inside the functions that need them, so `corpus` stays importable in a test
process that has neither.
"""

from __future__ import annotations

import os

# The embedder that produced the archive's own diarisation clusters, and the
# verified-ungated standalone fallback. `pyannote/embedding` is GATED (403)
# and `speechbrain/spkrec-ecapa-voxceleb` is absent from the image; both were
# probed on 2026-09-02 and are not candidates.
COMMUNITY = "pyannote/speaker-diarization-community-1"
WESPEAKER = "pyannote/wespeaker-voxceleb-resnet34-LM"


class Embedder:
    """Wraps whichever model was loaded behind one `.embed` call."""

    def __init__(self, inference, sample_rate: int = 16000) -> None:
        self._inference = inference
        self.sample_rate = sample_rate

    def embed(self, path: str, start_s: float, length_s: float) -> list[float]:
        import torchaudio

        # `frame_offset` and `num_frames` are counted in the FILE's native
        # rate, never in the rate we want out. Computing them against a
        # hardcoded 16000 seeks to start_s * 16000/native -- a third of the
        # intended timestamp on 48kHz audio -- and returns a third of the
        # intended duration, silently, with a perfectly plausible vector at
        # the end of it. That shipped, and it made the whole archive pass
        # embed the wrong moments (found 2026-09-04; the ratio scan peaked at
        # exactly 16000/48000). There is no single rate to assume: the
        # archive is measured mixed, 48kHz for 83% of episodes and 44.1kHz
        # for the rest, so the file has to be asked.
        info = torchaudio.info(path)
        native = int(info.sample_rate)
        offset = int(start_s * native)
        frames = int(length_s * native)
        waveform, rate = torchaudio.load(path, frame_offset=offset, num_frames=frames)
        if int(rate) != native:
            # The offset above was computed from `info`. If `load` disagrees
            # about the rate then it was computed against the wrong scale and
            # this crop is not the window that was asked for.
            raise RuntimeError(
                f"{path}: info reports {native}Hz but load returned {rate}Hz; "
                f"the crop at {start_s}s is not the requested window."
            )
        if rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, rate, self.sample_rate)

        # The check that turns this class of bug from silent into loud. A
        # crop may legitimately come back short only when its window runs off
        # the end of the file; any other shortfall means the seek landed
        # somewhere other than where it was sent.
        got_s = waveform.shape[-1] / self.sample_rate
        available_s = max(0.0, info.num_frames / native - start_s)
        want_s = min(length_s, available_s)
        if got_s < want_s * 0.9:
            raise RuntimeError(
                f"{path}: asked for {length_s}s at {start_s}s and got {got_s:.2f}s "
                f"({want_s:.2f}s was available). The seek did not land where it "
                f"was sent; the vector would describe the wrong audio."
            )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        vector = self._inference(
            {"waveform": waveform, "sample_rate": self.sample_rate}
        )
        return [float(v) for v in vector.reshape(-1)]


def load_embedder(name: str, hf_token: str, *, model_home: str) -> Embedder:
    """Return an embedder for `name`, or raise with the reason.

    Never returns a degraded object. A model that silently failed to load
    would produce vectors that are wrong rather than absent, and wrong vectors
    make confident wrong identities -- the failure with no external symptom.

    HF_HOME is pointed at the mounted model volume HERE rather than via
    `image.env(...)`, because changing the image would change its content
    address and force the rebuild the verbatim image spec exists to avoid.
    Without it every container re-downloads the model: the probe on
    2026-09-02 found /models holds the whisper and bge weights and NO pyannote
    weights, so diarisation has been paying that cost per cold start already.

    `model_home` is a parameter rather than a module constant because both
    calling apps mount the same volume but own their own mount path.
    """
    os.environ.setdefault("HF_HOME", model_home)

    from pyannote.audio import Inference, Model

    if name == COMMUNITY:
        # The embedder ships INSIDE the diarisation repo, in an `embedding`
        # subfolder -- probed 2026-09-02 from the live pipeline's config.
        model = Model.from_pretrained(
            name, subfolder="embedding", use_auth_token=hf_token
        )
    else:
        model = Model.from_pretrained(name, use_auth_token=hf_token)
    return Embedder(Inference(model, window="whole"))
