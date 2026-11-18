import math

import pytest

from semantic_cache.cache import ResponseCache, cosine, tokens


def test_tokenization_unicode():
    assert tokens("Äpfel ÄPFEL") == {"äpfel": 2}


def test_cosine():
    assert cosine(tokens("a b"), tokens("b a")) == pytest.approx(1)
    assert cosine(tokens("a"), tokens("b")) == 0
    assert cosine(tokens(""), tokens("b")) == 0


def make(clock=lambda: 100):
    c = ResponseCache(clock=clock)
    c.put(scope="a", model="m", prompt="hello world", answer="yes", ttl=10)
    return c


def test_hit_and_miss():
    c = make()
    assert c.lookup(scope="a", model="m", prompt="HELLO WORLD")["answer"] == "yes"
    assert c.lookup(scope="a", model="m", prompt="different words") is None


def test_scope_and_model_are_isolated():
    c = make()
    assert c.lookup(scope="b", model="m", prompt="hello world") is None
    assert c.lookup(scope="a", model="n", prompt="hello world") is None


def test_expiry_at_boundary():
    t = [100]
    c = make(lambda: t[0])
    t[0] = 110
    assert c.lookup(scope="a", model="m", prompt="hello world") is None
    assert c.purge_expired() == 1


def test_invalidation_is_scoped():
    c = make()
    c.put(scope="b", model="m", prompt="hello world", answer="other", ttl=10)
    assert c.invalidate(scope="a", model="m") == 1
    assert c.lookup(scope="b", model="m", prompt="hello world")["answer"] == "other"


def test_tie_prefers_newest():
    c = make()
    c.put(scope="a", model="m", prompt="world hello", answer="new", ttl=10)
    assert c.lookup(scope="a", model="m", prompt="world hello")["answer"] == "new"


def test_persistence(tmp_path):
    path = str(tmp_path / "cache.sqlite")
    c = ResponseCache(path, clock=lambda: 1)
    c.put(scope="a", model="m", prompt="hello", answer="ok", ttl=10)
    c.close()
    d = ResponseCache(path, clock=lambda: 2)
    assert d.lookup(scope="a", model="m", prompt="hello")["answer"] == "ok"


@pytest.mark.parametrize("ttl", [0, -1, math.inf, math.nan])
def test_invalid_ttl(ttl):
    with pytest.raises(ValueError):
        ResponseCache().put(scope="a", model="m", prompt="x", answer="y", ttl=ttl)


@pytest.mark.parametrize("threshold", [0, -1, 1.1, math.nan])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        make().lookup(scope="a", model="m", prompt="hello world", threshold=threshold)


def test_blank_query_does_not_hit():
    assert make().lookup(scope="a", model="m", prompt="!!!") is None


def test_threshold_allows_partial_match():
    c = make()
    assert c.lookup(scope="a", model="m", prompt="hello", threshold=0.7)["answer"] == "yes"
    assert c.lookup(scope="a", model="m", prompt="hello", threshold=0.9) is None


def test_known_false_positive_word_order():
    # The baseline intentionally ignores order: callers must not treat it as semantic safety.
    c = ResponseCache()
    c.put(scope="a", model="m", prompt="dog bites man", answer="one", ttl=10)
    assert c.lookup(scope="a", model="m", prompt="man bites dog") is not None
