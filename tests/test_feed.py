from corpus.feed import entry_guid, episode_number_of


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


def test_itunes_episode_wins_when_present():
    assert episode_number_of({"itunes_episode": "7", "title": "Episode 6: x"}) == "7"


def test_the_title_fallback_matches_the_abbreviated_forms():
    assert episode_number_of({"title": "Ep 5: something"}) == "5"
    assert episode_number_of({"title": "Ep. 5: something"}) == "5"
    assert episode_number_of({"title": "Ep5: something"}) == "5"


def test_the_title_fallback_cannot_match_the_word_episode():
    # Deliberate, not a bug. Widening it would collide with the publisher's
    # own itunes numbering, which disagrees with its titles by one -- two
    # different "Episode 4"s. See episode_number_of's docstring.
    assert episode_number_of({"title": "Episode 5: something"}) == "Unknown"


def test_an_entry_with_no_number_anywhere_is_unknown():
    assert episode_number_of({"title": "A trailer"}) == "Unknown"
    assert episode_number_of({}) == "Unknown"


def test_the_number_is_always_a_string():
    # The filename builder interpolates it directly, and an int would render
    # identically while comparing unequal to the parsed filename's str.
    assert episode_number_of({"itunes_episode": 12}) == "12"
