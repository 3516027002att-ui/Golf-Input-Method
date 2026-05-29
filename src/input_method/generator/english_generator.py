from typing import List, Dict, Tuple
import logging
import os
from .base import BaseCandidateGenerator, Candidate

logger = logging.getLogger(__name__)

# 内置高频英文单词：(单词, 静态词频)
BUILTIN_ENGLISH_WORDS: List[Tuple[str, float]] = [
    ("the", 10000), ("and", 8000), ("to", 7000), ("of", 6500), ("a", 6000),
    ("in", 5500), ("is", 5000), ("that", 4800), ("for", 4500), ("it", 4300),
    ("on", 4000), ("was", 3800), ("as", 3600), ("with", 3500), ("be", 3400),
    ("by", 3300), ("at", 3200), ("an", 3100), ("this", 3000), ("are", 2900),
    ("from", 2800), ("have", 2700), ("not", 2600), ("but", 2500), ("they", 2400),
    ("or", 2300), ("which", 2200), ("you", 2100), ("we", 2000), ("would", 1900),
    ("him", 1800), ("been", 1700), ("has", 1600), ("more", 1500), ("their", 1450),
    ("there", 1400), ("what", 1350), ("one", 1300), ("out", 1250), ("my", 1200),
    ("about", 1150), ("who", 1100), ("will", 1050), ("get", 1000), ("into", 950),
    ("just", 900), ("like", 850), ("time", 800), ("some", 750), ("them", 700),
    ("other", 650), ("people", 600), ("your", 580), ("only", 500),
    ("hello", 400), ("world", 380), ("test", 350), ("framework", 300), ("input", 250),
    ("method", 200), ("model", 180), ("rerank", 150), ("parameter", 100), ("golf", 90)
]

# 英文上下文联想词映射
ENGLISH_ASSOC_DICT: Dict[str, List[str]] = {
    "the": ["same", "first", "other", "government", "company", "world", "input"],
    "and": ["the", "then", "also", "more", "now", "here"],
    "to": ["be", "have", "do", "say", "get", "make", "use"],
    "you": ["can", "are", "have", "will", "do", "know", "see"],
    "i": ["have", "am", "will", "would", "think", "know", "want"],
    "we": ["are", "have", "can", "will", "need", "should"],
    "would": ["be", "like", "have", "rather", "say"],
    "input": ["method", "buffer", "generator", "framework"],
    "parameter": ["golf"],
}


class EnglishCandidateGenerator(BaseCandidateGenerator):
    """英文前缀补全召回器，支持基于 Tokenizer 词表与内置词典的补全"""

    def __init__(self, tokenizer_path: str = None, max_recall: int = 100):
        self.max_recall = max_recall
        self.vocab: List[Tuple[str, float]] = list(BUILTIN_ENGLISH_WORDS)
        
        # 尝试从 Tokenizer 载入词表资产
        if tokenizer_path:
            self._load_tokenizer_vocab(tokenizer_path)

    def _load_tokenizer_vocab(self, tokenizer_path: str) -> None:
        # 寻找对应的 .vocab 文件 (与 .model 往往在同一目录下)
        vocab_path = tokenizer_path
        if vocab_path.endswith(".model"):
            vocab_path = vocab_path[:-6] + ".vocab"

        if os.path.exists(vocab_path):
            try:
                loaded_tokens = []
                with open(vocab_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            token, score_str = parts[0], parts[1]
                            try:
                                score = float(score_str)
                            except ValueError:
                                score = 0.0
                            
                            # 替换 SentencePiece 特殊空格字符 ▁ (通常是 ▁, \u2581 或 _)
                            # 将其清洗为普通空格或空前缀
                            clean_token = token.replace("\u2581", "").replace("▁", "").strip()
                            if clean_token and clean_token.isalpha():
                                # 词表中 score 越大 (负值越接近0) 概率越高，将其映射到正向分数区间
                                loaded_tokens.append((clean_token.lower(), score))
                
                # 将载入的 tokens 加入到我们的词表
                # 与内置词表结合，分数统一映射到与 BUILTIN_ENGLISH_WORDS 相近的量级
                for token, score in loaded_tokens:
                    # Tokenizer 分数一般为负值到 0 之间，偏移并缩放到 0~2000
                    base_freq = (score + 10.0) * 100.0
                    self.vocab.append((token, max(0.0, base_freq)))
            except Exception:
                logger.warning("Tokenizer 词表加载失败 (path=%s)，降级使用内置词典", vocab_path, exc_info=True)

    def generate_candidates(self, context_before: str, composing: str) -> List[Candidate]:
        candidates: List[Candidate] = []
        composing = composing.lower().strip()

        # 场景 A: Composing 缓冲区为空 -> 触发上下文联想召回
        if not composing:
            if context_before:
                words_before = context_before.lower().split()
                if words_before:
                    last_word = words_before[-1]
                    if last_word in ENGLISH_ASSOC_DICT:
                        assoc_words = ENGLISH_ASSOC_DICT[last_word]
                        for idx, word in enumerate(assoc_words):
                            score = max(0.01, 1.0 - idx * 0.1)
                            candidates.append(Candidate(
                                text=word,
                                composing_covered="",
                                score=score,
                                source="association"
                            ))
            return candidates

        # 场景 B: Composing 不为空 -> 匹配单词前缀
        for word, score in self.vocab:
            if word.startswith(composing):
                # 如果是精确匹配，给予更高的加权
                if word == composing:
                    final_score = score * 1.5
                    src = "vocab_exact"
                else:
                    ratio = len(composing) / len(word)
                    final_score = score * 0.8 * ratio
                    src = "vocab_prefix"

                candidates.append(Candidate(
                    text=word,
                    composing_covered=composing,
                    score=final_score,
                    source=src
                ))

        # 去重并截断（排序由 Reranker 统一负责）
        return self._deduplicate_and_truncate(candidates, self.max_recall)
