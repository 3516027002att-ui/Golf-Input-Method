from typing import List, Optional
from .base import BaseReranker
from ..generator.base import Candidate
from ..user_memory import UserMemory


class FrequencyReranker(BaseReranker):
    """传统 baseline 排序器：静态召回分 + 匹配特征 + 用户记忆。"""

    def __init__(self, user_memory: Optional[UserMemory] = None):
        self.user_memory = user_memory

    def rerank(self, context_before: str, composing: str, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return []

        input_key = composing.lower().strip()
        scored_candidates: List[Candidate] = []
        for cand in candidates:
            score = float(cand.score)
            source = cand.source

            if "exact_pinyin" in source or source in ("ja_exact", "vocab_exact"):
                score *= 1.20
            elif "exact_short" in source:
                score *= 1.05
            elif "prefix" in source:
                score *= 0.92
            elif "segmented" in source:
                score *= 0.82

            if len(cand.text) >= 2 and source.startswith("dict"):
                score *= 1.04
            if len(cand.text) > 8:
                score *= 0.95

            if self.user_memory is not None and input_key:
                score += self.user_memory.get_user_weight(cand.text, input_key) * 10000.0

            scored_candidates.append(
                Candidate(
                    text=cand.text,
                    composing_covered=cand.composing_covered,
                    score=score,
                    source=cand.source,
                )
            )

        return sorted(scored_candidates, key=lambda x: x.score, reverse=True)
