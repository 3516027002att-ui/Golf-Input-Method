from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Candidate:
    # 候选字词上屏文本 (例如 "你好" 或 "the")
    text: str
    
    # 匹配输入缓冲区的 composing 子段 (例如 "nihhao" 或 "th")
    composing_covered: str
    
    # 静态分数或召回分数 (值越大排序越靠前)
    score: float = 0.0
    
    # 候选词来源 (例如 "dict", "association", "tokenizer", "user")
    source: str = "dict"

    def __repr__(self) -> str:
        return f"Candidate(text='{self.text}', composing='{self.composing_covered}', score={self.score:.4f}, source='{self.source}')"


class BaseCandidateGenerator(ABC):
    """候选字词召回器的抽象基类"""
    
    @abstractmethod
    def generate_candidates(self, context_before: str, composing: str) -> List[Candidate]:
        """
        根据当前上下文和输入缓冲区生成候选词列表。
        
        Args:
            context_before: 光标之前的已上屏历史文本 (可用于联想词生成)
            composing: 当前输入的原始字母串 (例如 "nihhao")
            
        Returns:
            Candidate 对象列表
        """
        pass

    @staticmethod
    def _deduplicate_and_truncate(candidates: List[Candidate], max_count: int) -> List[Candidate]:
        """去重（保留同词中分数最高者）并截断到 max_count 条"""
        unique: Dict[str, Candidate] = {}
        for cand in candidates:
            if cand.text not in unique or cand.score > unique[cand.text].score:
                unique[cand.text] = cand
        return list(unique.values())[:max_count]

