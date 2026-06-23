"""Unit tests for evaluation metric functions."""

from eval.evaluate import keyword_recall, reciprocal_rank
from eval.judge import parse_judge_response


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


def test_parse_judge_response_plain_json():
    out = parse_judge_response('{"faithfulness": 5, "relevance": 4, "reason": "ok"}')
    assert out["faithfulness"] == 5
    assert out["relevance"] == 4
    assert out["reason"] == "ok"


def test_parse_judge_response_with_code_fence_and_prose():
    text = 'Here is my verdict:\n```json\n{"faithfulness": 3, "relevance": 5}\n```'
    out = parse_judge_response(text)
    assert out["faithfulness"] == 3
    assert out["relevance"] == 5


def test_parse_judge_response_clamps_and_handles_garbage():
    assert parse_judge_response('{"faithfulness": 9, "relevance": 0}')["faithfulness"] == 5
    assert parse_judge_response('{"faithfulness": 9, "relevance": 0}')["relevance"] == 1
    bad = parse_judge_response("not json at all")
    assert bad["faithfulness"] is None
    assert bad["relevance"] is None
