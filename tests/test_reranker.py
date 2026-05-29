import unittest
from src.input_method.generator.base import Candidate
from src.input_method.reranker.frequency_reranker import FrequencyReranker
from src.input_method.reranker.model_reranker import ModelReranker


class TestFrequencyReranker(unittest.TestCase):
    """测试词频排序器"""

    def test_sort_by_score_descending(self) -> None:
        """测试按分数降序排列"""
        reranker = FrequencyReranker()
        candidates = [
            Candidate(text="low", composing_covered="test", score=10.0),
            Candidate(text="high", composing_covered="test", score=100.0),
            Candidate(text="mid", composing_covered="test", score=50.0),
        ]
        sorted_cands = reranker.rerank(context_before="", composing="test", candidates=candidates)
        self.assertEqual(sorted_cands[0].text, "high")
        self.assertEqual(sorted_cands[1].text, "mid")
        self.assertEqual(sorted_cands[2].text, "low")

    def test_empty_candidates(self) -> None:
        """测试空候选列表"""
        reranker = FrequencyReranker()
        result = reranker.rerank(context_before="", composing="", candidates=[])
        self.assertEqual(result, [])


class TestModelReranker(unittest.TestCase):
    """测试模型排序器"""

    def test_fallback_when_no_model(self) -> None:
        """测试无模型时自动退避到词频排序"""
        reranker = ModelReranker()  # 无 model_path
        self.assertFalse(reranker.model_loaded)
        
        candidates = [
            Candidate(text="low", composing_covered="test", score=10.0),
            Candidate(text="high", composing_covered="test", score=100.0),
        ]
        result = reranker.rerank(context_before="", composing="test", candidates=candidates)
        # 应该按词频排序（退避行为）
        self.assertEqual(result[0].text, "high")

    def test_stub_context_scoring(self) -> None:
        """测试桩类的上下文加权打分"""
        reranker = ModelReranker()
        reranker.model_loaded = True
        
        candidates = [
            Candidate(text="golf", composing_covered="gf", score=10.0),
            Candidate(text="go", composing_covered="go", score=20.0),
        ]
        result = reranker.rerank(context_before="parameter", composing="gf", candidates=candidates)
        # "golf" 在 "parameter" 上下文下获得 3.0 倍加权 (30.0)，应超越 "go" (20.0)
        self.assertEqual(result[0].text, "golf")

    def test_nonexistent_model_path(self) -> None:
        """测试不存在的模型路径"""
        reranker = ModelReranker(model_path="/nonexistent/model.pt")
        self.assertFalse(reranker.model_loaded)
        self.assertIn("not found", reranker.model_info)


if __name__ == "__main__":
    unittest.main()
