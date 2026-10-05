"""Levenshtein self-scoring sanity (used by Eval 3 reporting)."""

from cybertyping.tasks._levenshtein import levenshtein


def test_basic():
    assert levenshtein("", "") == 0
    assert levenshtein("a", "") == 1
    assert levenshtein("", "abc") == 3
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("hello world", "hello world") == 0
    assert levenshtein("hello world", "hello vorld") == 1
