"""Who decides where Hugging Face caches its weights, and when.

`load_embedder` sets `HF_HOME` as its first statement, before it imports
pyannote and before anything that can raise. That has always been more than a
convenience for the voice embedder: `HF_HOME` is process-wide, so it also
redirects `DiarizationPipeline`, which `transcribe.py` constructs per episode
LATER in the same container and which has nothing to do with speaker identity.

The consequence was that an unrelated subsystem's cache root depended on WHERE
inside enrolment a failure happened -- a missing mount raised before
`load_embedder` was reached and left `HF_HOME` unset, while a failed model
download raised after `setdefault` had already run. Both print the same
"ENROLMENT UNAVAILABLE" line, and nothing distinguished them.

The fix makes the caller own it: `process_episodes` and `repair_identity` set
`HF_HOME` explicitly and early, as a container-wide choice. These tests pin
the property that makes that fix STICK -- `setdefault` semantics, so an
explicit value the caller already chose is never overwritten -- and the
fallback that keeps `speaker_embed.py` working, since it mounts its own path.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def fake_pyannote(monkeypatch):
    """A stand-in for `pyannote.audio`, so no model or token is needed.

    `load_embedder` imports pyannote INSIDE the function body, so putting the
    fake in `sys.modules` after import time is enough -- the same trick
    `tests/test_embed_crop_window.py` uses for torchaudio.
    """
    loaded: dict = {}

    def from_pretrained(name, **kwargs):
        loaded["name"] = name
        loaded["kwargs"] = kwargs
        return object()

    module = types.ModuleType("pyannote.audio")
    module.Model = types.SimpleNamespace(from_pretrained=from_pretrained)
    module.Inference = lambda model, window: types.SimpleNamespace(  # noqa: ARG005
        model=model, window=window
    )
    package = types.ModuleType("pyannote")
    package.audio = module
    monkeypatch.setitem(sys.modules, "pyannote", package)
    monkeypatch.setitem(sys.modules, "pyannote.audio", module)
    return loaded


def test_an_explicitly_set_hf_home_is_never_overwritten(monkeypatch, fake_pyannote):
    """THE PROPERTY THE FIX RELIES ON.

    `process_episodes` sets HF_HOME before it loads anything. If this were a
    plain assignment rather than `setdefault`, that deliberate container-wide
    choice would be silently replaced by whatever `model_home` the enrolment
    load happened to pass -- and the fix would be undone by the very call it
    exists to constrain.
    """
    from corpus.voice import COMMUNITY, load_embedder

    monkeypatch.setenv("HF_HOME", "/chosen-by-the-caller")

    load_embedder(COMMUNITY, "token", model_home="/models")

    import os

    assert os.environ["HF_HOME"] == "/chosen-by-the-caller"


def test_hf_home_is_still_set_when_the_caller_chose_nothing(monkeypatch, fake_pyannote):
    """The fallback `speaker_embed.py` depends on.

    It mounts its own model path and does NOT set HF_HOME itself, so removing
    the setdefault entirely would send it back to re-downloading the weights
    on every cold start.
    """
    from corpus.voice import COMMUNITY, load_embedder

    monkeypatch.delenv("HF_HOME", raising=False)

    load_embedder(COMMUNITY, "token", model_home="/models")

    import os

    assert os.environ["HF_HOME"] == "/models"


def test_the_community_embedder_is_loaded_from_the_embedding_subfolder(fake_pyannote):
    """Pins the one detail that makes COMMUNITY different from WESPEAKER.

    The embedder ships INSIDE the diarisation repo, in an `embedding`
    subfolder. Dropping the subfolder silently loads the diarisation model
    instead, which produces vectors that are wrong rather than absent.
    """
    from corpus.voice import COMMUNITY, load_embedder

    load_embedder(COMMUNITY, "token", model_home="/models")

    assert fake_pyannote["name"] == COMMUNITY
    assert fake_pyannote["kwargs"]["subfolder"] == "embedding"


def test_a_standalone_embedder_is_loaded_without_a_subfolder(fake_pyannote):
    from corpus.voice import WESPEAKER, load_embedder

    load_embedder(WESPEAKER, "token", model_home="/models")

    assert fake_pyannote["name"] == WESPEAKER
    assert "subfolder" not in fake_pyannote["kwargs"]
