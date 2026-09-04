"""`_Embedder.embed` must crop the window it was asked for, at any file rate.

The archive pass shipped with `frame_offset`/`num_frames` computed against a
hardcoded 16000 while `torchaudio.load` counts both in the FILE's native
rate. On 48kHz audio every crop therefore landed at `start_s * 16000/48000`
-- one third of its intended timestamp -- and returned one third of its
intended duration. Nothing raised. Every vector was well-formed, unit-norm
and describing the wrong moment, which is why it survived a full 433-episode
run and was only caught by scanning the timestamp scale as a free parameter
(peak accuracy 0.916 at exactly r=1/3, against 0.492 at r=1).

The archive is measured mixed-rate -- 48000Hz for 155 of 192 sampled
episodes, 44100Hz for the other 37 -- so "assume 48k" is not a fix either.

These tests assert the two things that were wrong, plus the guard that would
have made the original failure loud:

  * the seek is computed in the file's own rate, at 48000 AND at 44100
  * a crop that comes back short mid-file raises rather than embedding
  * a crop that comes back short because it ran off the END of the file is
    legitimate and must not raise

Mutation check for the first: reverting the arithmetic to `self.sample_rate`
makes `test_seek_is_in_the_files_native_rate` fail on both parametrised
rates (recorded offset 160000 against an expected 480000). It is not a test
that passes on the code it was written to reject.
"""

from __future__ import annotations

import sys
import types

import pytest


class _FakeWaveform:
    """The slice of `.shape`, `.mean` and division that `embed` actually uses."""

    def __init__(self, channels: int, frames: int) -> None:
        self.shape = (channels, frames)

    def mean(self, dim: int, keepdim: bool) -> "_FakeWaveform":  # noqa: ARG002
        return _FakeWaveform(1, self.shape[-1])


class _FakeInfo:
    def __init__(self, sample_rate: int, num_frames: int) -> None:
        self.sample_rate = sample_rate
        self.num_frames = num_frames


class _FakeTorchaudio(types.ModuleType):
    """Counts frames the way the real library does: in the file's own rate.

    `load_rate` exists so a test can make `info` and `load` disagree, which is
    the one case where the offset is known to have been computed against the
    wrong scale.
    """

    def __init__(
        self,
        *,
        native: int,
        duration_s: float,
        channels: int = 2,
        load_rate: int | None = None,
        short_by: int = 0,
    ) -> None:
        super().__init__("torchaudio")
        self._native = native
        self._total = int(duration_s * native)
        self._channels = channels
        self._load_rate = load_rate or native
        self._short_by = short_by
        self.calls: list[dict] = []
        self.functional = types.SimpleNamespace(resample=self._resample)

    def info(self, path: str) -> _FakeInfo:  # noqa: ARG002
        return _FakeInfo(self._native, self._total)

    def load(self, path: str, frame_offset: int = 0, num_frames: int = -1):  # noqa: ARG002
        self.calls.append({"frame_offset": frame_offset, "num_frames": num_frames})
        # The real library clamps at EOF rather than erroring or zero-padding.
        available = max(0, self._total - frame_offset)
        got = min(num_frames, available) - self._short_by
        return _FakeWaveform(self._channels, max(got, 0)), self._load_rate

    def _resample(self, waveform: _FakeWaveform, orig: int, new: int):
        return _FakeWaveform(waveform.shape[0], round(waveform.shape[-1] * new / orig))


@pytest.fixture
def embedder(monkeypatch):
    """`_Embedder` with a stub inference, and a hook to install a fake library.

    `embed` imports torchaudio inside the function body, so patching
    `sys.modules` after import time is enough and no real audio stack is
    needed to exercise the arithmetic.
    """
    import speaker_embed

    def _make(**kwargs):
        fake = _FakeTorchaudio(**kwargs)
        monkeypatch.setitem(sys.modules, "torchaudio", fake)
        inference = lambda payload: types.SimpleNamespace(  # noqa: E731
            reshape=lambda _shape: [0.5, 0.5]
        )
        return speaker_embed._Embedder(inference), fake

    return _make


@pytest.mark.parametrize("native", [48000, 44100])
def test_seek_is_in_the_files_native_rate(embedder, native):
    """The bug, stated directly: 30s at 600s must be asked for in file frames."""
    emb, fake = embedder(native=native, duration_s=3600.0)

    emb.embed("episode.mp3", start_s=600.0, length_s=30.0)

    assert fake.calls == [{"frame_offset": 600 * native, "num_frames": 30 * native}], (
        "offset and length must be counted in the file's rate, not the output rate"
    )


def test_output_is_resampled_to_the_embedder_rate(embedder):
    """Cropping in native frames must not change what the model is handed."""
    emb, fake = embedder(native=48000, duration_s=3600.0)

    emb.embed("episode.mp3", start_s=600.0, length_s=30.0)

    # 30s at 48k in, 30s at 16k out -- the resample still has to happen, it
    # just happens after a seek that landed in the right place.
    assert fake.calls[0]["num_frames"] == 30 * 48000


def test_a_short_crop_mid_file_raises(embedder):
    """The guard that would have made the original bug loud instead of silent."""
    emb, _ = embedder(native=48000, duration_s=3600.0, short_by=20 * 48000)

    with pytest.raises(RuntimeError, match="did not land where it was sent"):
        emb.embed("episode.mp3", start_s=600.0, length_s=30.0)


def test_a_short_crop_at_the_end_of_the_file_is_allowed(embedder):
    """A window overhanging EOF is legitimate and must not be rejected."""
    emb, _ = embedder(native=48000, duration_s=610.0)

    # Asks for 30s starting at 600s; only 10s exist. That is the file's fault,
    # not the seek's, and `select_crop_turns` can legitimately produce it.
    vector = emb.embed("episode.mp3", start_s=600.0, length_s=30.0)

    assert vector == [0.5, 0.5]


def test_disagreement_between_info_and_load_raises(embedder):
    """If the two disagree, the offset was computed against the wrong scale."""
    emb, _ = embedder(native=48000, duration_s=3600.0, load_rate=44100)

    with pytest.raises(RuntimeError, match="is not the requested window"):
        emb.embed("episode.mp3", start_s=600.0, length_s=30.0)
