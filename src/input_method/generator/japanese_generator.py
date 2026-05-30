import logging
from typing import List, Dict, Tuple
from .base import BaseCandidateGenerator, Candidate

logger = logging.getLogger(__name__)

# Romaji -> Hiragana 的核心音节对照表 (极简版，满足原型召回即可)
ROMAJI_MAP: Dict[str, str] = {
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "sa": "さ", "shi": "し", "su": "す", "se": "せ", "so": "そ",
    "ta": "た", "chi": "ち", "tsu": "つ", "te": "て", "to": "と",
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "fu": "ふ", "he": "へ", "ho": "ほ",
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "ya": "や", "yu": "ゆ", "yo": "よ",
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "wa": "わ", "wo": "を", "n": "ん"
}

# 常见日语单词召回：(罗马字, 假名/汉字, 分数)
JAPANESE_WORDS: List[Tuple[str, str, float]] = [
    ("nihongo", "日本語", 10.0),
    ("nihongo", "にほんご", 8.0),
    ("nihon", "日本", 9.0),
    ("nihon", "にほん", 7.0),
    ("arigatou", "ありがとう", 9.5),
    ("sayounara", "さようなら", 8.5),
    ("golf", "ゴルフ", 9.0),
    ("konnichiha", "こんにちは", 9.0),
    ("sumimasen", "すみません", 8.5)
]

# 日语上下文联想词映射
JA_ASSOC_DICT: Dict[str, List[str]] = {
    "日本語": ["の", "勉強", "入力", "難しい"],
    "日本": ["の", "食", "桜", "東京"],
    "ありがとう": ["ございます", "ございました"],
}


class JapaneseCandidateGenerator(BaseCandidateGenerator):
    """
    日语罗马字输入候选召回器 (原型骨架)。
    
    支持简易的罗马音转假名及部分高频日语词汇，旨在打通日语输入链路。
    """

    def __init__(self, max_recall: int = 100):
        self.max_recall = max_recall

    def generate_candidates(self, context_before: str, composing: str) -> List[Candidate]:
        candidates: List[Candidate] = []
        composing = composing.lower().strip()

        # 场景 A: Composing 缓冲区为空 -> 触发联想词
        if not composing:
            if context_before:
                # 提取末尾词汇做联想
                for length in (4, 3, 2, 1):
                    if len(context_before) >= length:
                        tail = context_before[-length:]
                        if tail in JA_ASSOC_DICT:
                            for idx, word in enumerate(JA_ASSOC_DICT[tail]):
                                score = max(0.01, 1.0 - idx * 0.1)
                                candidates.append(Candidate(
                                    text=word,
                                    composing_covered="",
                                    score=score,
                                    source="association"
                                ))
                            break
            return candidates

        # 场景 B: Composing 不为空 -> 召回罗马音假名与常用词
        # 1. 召回假名音节转换结果 (如输入 "ka" -> "か")
        if composing in ROMAJI_MAP:
            candidates.append(Candidate(
                text=ROMAJI_MAP[composing],
                composing_covered=composing,
                score=10.0,
                source="romaji_kana"
            ))

        # 2. 召回单词字典匹配 (包含精确与前缀匹配)
        for romaji, kana, base_score in JAPANESE_WORDS:
            if romaji == composing:
                candidates.append(Candidate(
                    text=kana,
                    composing_covered=composing,
                    score=base_score * 1.5,
                    source="ja_exact"
                ))
            elif romaji.startswith(composing):
                ratio = len(composing) / len(romaji)
                candidates.append(Candidate(
                    text=kana,
                    composing_covered=composing,
                    score=base_score * 0.8 * ratio,
                    source="ja_prefix"
                ))

        # 3. 提示：告知这是原型骨架
        candidates.append(Candidate(
            text="[提示: 日语输入处于原型阶段]",
            composing_covered=composing,
            score=0.01,
            source="system_tip"
        ))

        # 去重并截断
        return self._deduplicate_and_truncate(candidates, self.max_recall)
