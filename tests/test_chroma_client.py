import pytest

from corpus import chroma_client
from corpus.chroma_client import ChromaConfigError, connection_settings

TOKEN = "t" * 64


def test_https_url_gives_tls_on_443_with_a_bearer_header():
    s = connection_settings(
        {"CHROMA_URL": "https://chroma.example.test", "CHROMA_TOKEN": TOKEN}
    )
    assert s == {
        "host": "chroma.example.test",
        "port": 443,
        "ssl": True,
        "headers": {"Authorization": f"Bearer {TOKEN}"},
    }


def test_an_explicit_port_is_kept():
    s = connection_settings(
        {"CHROMA_URL": "https://example.test:8443", "CHROMA_TOKEN": TOKEN}
    )
    assert (s["port"], s["ssl"]) == (8443, True)


def test_plain_http_is_allowed_only_for_a_local_server():
    s = connection_settings(
        {"CHROMA_URL": "http://localhost:8000", "CHROMA_TOKEN": TOKEN}
    )
    assert (s["host"], s["port"], s["ssl"]) == ("localhost", 8000, False)
    with pytest.raises(ChromaConfigError, match="https"):
        connection_settings(
            {"CHROMA_URL": "http://chroma.example.test", "CHROMA_TOKEN": TOKEN}
        )


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"CHROMA_URL": "https://chroma.example.test"},
        {"CHROMA_TOKEN": TOKEN},
        {"CHROMA_URL": "", "CHROMA_TOKEN": TOKEN},
        {"CHROMA_URL": "https://chroma.example.test", "CHROMA_TOKEN": ""},
    ],
)
def test_missing_settings_are_refused_naming_the_secret(env):
    # No silent fallback to Chroma Cloud: after 2026-11-22 it is read-only,
    # and a writer that quietly reached it would fail far from the cause.
    with pytest.raises(ChromaConfigError, match="chroma-selfhosted"):
        connection_settings(env)


def test_a_url_without_a_host_is_refused():
    with pytest.raises(ChromaConfigError):
        connection_settings({"CHROMA_URL": "https://", "CHROMA_TOKEN": TOKEN})


def test_the_error_never_contains_the_token():
    with pytest.raises(ChromaConfigError) as exc:
        connection_settings(
            {"CHROMA_URL": "http://chroma.example.test", "CHROMA_TOKEN": TOKEN}
        )
    assert TOKEN not in str(exc.value)


def test_make_client_builds_an_http_client_from_the_settings(monkeypatch):
    seen = {}

    class FakeChromadb:
        @staticmethod
        def HttpClient(**kwargs):
            seen.update(kwargs)
            return "client"

    monkeypatch.setitem(__import__("sys").modules, "chromadb", FakeChromadb)
    env = {"CHROMA_URL": "https://chroma.example.test", "CHROMA_TOKEN": TOKEN}
    assert chroma_client.make_client(env) == "client"
    assert seen == connection_settings(env)
