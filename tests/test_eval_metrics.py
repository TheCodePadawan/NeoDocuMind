"""Unit tests for evaluation metric functions."""

from eval.evaluate import keyword_recall, reciprocal_rank


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a.md", "b.md"], "a.md") == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank(["x.md", "a.md", "b.md"], "a.md") == 0.5


def test_reciprocal_rank_missing():
    assert reciprocal_rank(["x.md", "y.md"], "a.md") == 0.0


def test_keyword_recall_full_and_partial():
    assert keyword_recall("The limit is 60 USD per meal.", ["60", "meal"]) == 1.0
    assert keyword_recall("The limit is 60 USD.", ["60", "meal"]) == 0.5
    assert keyword_recall("nothing here", []) == 0.0
