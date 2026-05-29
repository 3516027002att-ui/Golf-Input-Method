from abc import ABC, abstractmethod
from typing import List
from ..generator.base import Candidate

class BaseReranker(ABC):
    """候选词重排器的抽象基类"""

    @abstractmethod
    def rerank(self, context_before: str, composing: str, candidates: List[Candidate]) -> List[Candidate]:
        """
        对已召回的候选词列表进行打分并重新排序。
        
        Args:
            context_before: 光标之前的已上屏历史文本 (作为预测排序的上下文)
            composing: 当前输入的原始字母串 (例如 "nihhao")
            candidates: 输入的候选词列表
            
        Returns:
            重排并更新了分数后的 Candidate 列表，按分数从高到低排列。
        """
        pass
