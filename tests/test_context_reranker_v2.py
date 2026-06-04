import unittest
from collections import Counter

from training.context_reranker_v2 import (
    build_pair_text,
    candidate_generation_audit,
    candidate_order_indices,
    cross_split_overlap_counts,
    finalize_product_metrics,
    memory_baseline,
    parse_sample,
    signature_raw_input_target,
    target_index_for_order,
    topk_from_rank,
    topk_from_static_rank,
)


def make_sample(
    *,
    sample_id: str,
    source_doc_key: str,
    composing: str = "nihao",
    target_index: int = 1,
    context_before: str = "今天我想说",
    context_after: str = "，谢谢",
):
    raw = {
        "sample_id": sample_id,
        "source_doc_key": source_doc_key,
        "context_before": context_before,
        "context_after": context_after,
        "composing": composing,
        "candidates": [
            {"text": "拟好", "freq": 10, "source": "system_dict", "match_type": "same_pinyin", "static_rank": 1},
            {"text": "你好", "freq": 9, "source": "system_dict", "match_type": "exact_pinyin", "static_rank": 2},
            {"text": "泥嚎", "freq": 1, "source": "system_dict", "match_type": "same_pinyin", "static_rank": 3},
        ],
        "target_index": target_index,
    }
    sample, warnings = parse_sample(raw)
    assert sample is not None, warnings
    return sample


class TestContextRerankerV2Audit(unittest.TestCase):
    def test_context_mode_online_excludes_future_context(self) -> None:
        online = build_pair_text(
            context_before="上文ABC",
            context_after="未来XYZ",
            composing="zg",
            candidate="中国",
            context_mode="online",
        )
        edit = build_pair_text(
            context_before="上文ABC",
            context_after="未来XYZ",
            composing="zg",
            candidate="中国",
            context_mode="edit",
        )
        none = build_pair_text(
            context_before="上文ABC",
            context_after="未来XYZ",
            composing="zg",
            candidate="中国",
            context_mode="none",
        )

        self.assertIn("上文ABC", online)
        self.assertNotIn("未来XYZ", online)
        self.assertIn("未来XYZ", edit)
        self.assertNotIn("上文ABC", none)
        self.assertNotIn("未来XYZ", none)

    def test_static_baselines_are_separate_from_candidate_zero(self) -> None:
        sample = make_sample(sample_id="s1", source_doc_key="doc1")
        candidate0 = topk_from_rank([sample.target_index], [len(sample.candidates)])
        static_rank = topk_from_static_rank([sample])

        self.assertEqual(candidate0["top1"], 0.0)
        self.assertEqual(static_rank["top1"], 0.0)
        self.assertEqual(static_rank["top3"], 1.0)

    def test_memory_baseline_uses_train_majority_target(self) -> None:
        train = [make_sample(sample_id="tr1", source_doc_key="doc1")]
        val = [make_sample(sample_id="val1", source_doc_key="doc2")]
        result = memory_baseline(train, val, lambda sample: sample.composing)

        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["top1_all"], 1.0)

    def test_split_overlap_detects_raw_input_target_reuse(self) -> None:
        train = [make_sample(sample_id="tr1", source_doc_key="doc1")]
        val = [make_sample(sample_id="val1", source_doc_key="doc2")]
        result = cross_split_overlap_counts(
            {"train": train, "val": val},
            signature_raw_input_target,
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(sorted(result["examples"][0]["splits"]), ["train", "val"])

    def test_candidate_shuffle_preserves_target_mapping(self) -> None:
        sample = make_sample(sample_id="shuffle-case", source_doc_key="doc1")
        order = candidate_order_indices(sample, "shuffle")
        mapped_target = target_index_for_order(sample, order, "normal")

        self.assertEqual(sorted(order), [0, 1, 2])
        self.assertEqual(order[mapped_target], sample.target_index)
        self.assertIn(target_index_for_order(sample, order, "random"), [0, 1, 2])

    def test_candidate_generation_audit_counts_hard_negatives(self) -> None:
        sample = make_sample(sample_id="s1", source_doc_key="doc1")
        result = candidate_generation_audit([sample])

        self.assertGreater(result["hard_negative_candidate_ratio"], 0.0)
        self.assertEqual(result["samples_without_hard_negative_rate"], 0.0)
        self.assertEqual(result["target_first_rate"], 0.0)

    def test_parse_sample_accepts_recommended_candidate_meta(self) -> None:
        sample, warnings = parse_sample(
            {
                "sample_id": "meta1",
                "source_doc_key": "doc1",
                "context_before": "上文",
                "composing": "zg",
                "candidates": ["中国", "中过"],
                "target_index": 0,
                "candidate_meta": [
                    {"freq": 99, "source": "system_dict", "match_type": "exact_pinyin", "original_rank": 1},
                    {"freq": 1, "source": "system_dict", "match_type": "same_pinyin", "original_rank": 2},
                ],
            }
        )

        self.assertEqual(warnings, [])
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.candidate_meta[0].freq, 99.0)
        self.assertEqual(sample.candidate_meta[0].original_rank, 1)

    def test_product_metrics_reject_when_harm_exceeds_fix(self) -> None:
        metrics = finalize_product_metrics(
            Counter(
                {
                    "original_top1_correct_total": 10,
                    "original_top1_wrong_total": 10,
                    "keep_when_original_top1_correct": 8,
                    "harm_when_original_top1_correct": 2,
                    "fix_when_original_top1_wrong": 1,
                }
            )
        )

        self.assertEqual(metrics["harm_when_original_top1_correct"], 0.2)
        self.assertEqual(metrics["fix_when_original_top1_wrong"], 0.1)
        self.assertEqual(metrics["default_enable_recommendation"], "reject_default_enable")


if __name__ == "__main__":
    unittest.main()
