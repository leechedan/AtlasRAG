import pytest

from app.models.reranker import get_reranker

pytestmark = pytest.mark.slow


def test_score_ranks_relevant_text_higher():
    reranker = get_reranker()
    scores = reranker.score(
        "苹果的产地在哪里？",
        ["苹果主要产自中国、美国等国家。", "今天天气晴朗，适合出门跑步。"],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_get_reranker_is_singleton():
    assert get_reranker() is get_reranker()
