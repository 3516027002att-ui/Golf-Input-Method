from typing import List
from .base import BaseReranker
from ..generator.base import Candidate

class FrequencyReranker(BaseReranker):
    """基于静态词频或召回分数对候选词进行重排的排序器"""

    def rerank(self, context_before: str, composing: str, candidates: List[Candidate]) -> List[Candidate]:
        # 如果没有候选词，直接返回
        if not candidates:
            return []
            
        # 按照 Candidate 自身的分数进行降序排列
        # 分数计算已经由各自的 Generator 初步确定
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        return sorted_candidates
