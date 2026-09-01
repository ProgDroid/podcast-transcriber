from corpus.feed import entry_guid


def test_a_plain_guid_is_returned():
    assert (
        entry_guid({"id": "1c45dbd9-0dc3-4d07", "guidislink": False})
        == "1c45dbd9-0dc3-4d07"
    )


def test_a_link_derived_guid_is_refused():
    # RSS <guid isPermaLink> defaults to true. A link-derived id is a URL, and
    # this publisher rewrites URLs, so it is not an identity.
    assert entry_guid({"id": "https://example.com/ep/1", "guidislink": True}) is None


def test_a_missing_guid_is_none():
    assert entry_guid({}) is None
    assert entry_guid({"id": "", "guidislink": False}) is None


def test_guidislink_absent_is_treated_as_a_permalink():
    # feedparser omits the key for some feeds. RSS defaults isPermaLink to
    # true, so absence must be refused, never silently accepted.
    assert entry_guid({"id": "https://example.com/ep/2"}) is None
