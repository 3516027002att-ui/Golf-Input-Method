import time
from typing import List, Optional
from .base import BaseReranker
from ..generator.base import Candidate
from ..user_memory import UserMemory


class FrequencyReranker(BaseReranker):
    """传统 baseline 排序器：静态召回分 + 匹配特征 + 用户记忆 + 可解释打分。"""

    # 所有排序权重集中管理，便于调参
    WEIGHTS = {
        "exact_pinyin": 1.20,
        "exact_short": 1.05,
        "prefix": 0.92,
        "segmented": 0.82,
        "word_len_bonus": 1.04,       # 2字以上词语加分
        "long_text_penalty": 0.95,    # 超长文本惩罚
        "user_memory_scale": 10000.0, # 用户记忆权重缩放
        "recent_use_bonus": 5000.0,   # 最近使用加权（1小时内）
        "recent_use_window_s": 3600,  # 最近使用时间窗口（秒）
        "domain_weight": 1.0,         # 领域词库权重（预留）
    }

    def __init__(self, user_memory: Optional[UserMemory] = None, debug_mode: bool = False):
        self.user_memory = user_memory
        self.debug_mode = debug_mode

    def rerank(self, context_before: str, composing: str, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return []

        input_key = composing.lower().strip()
        now = time.time()
        scored_candidates: List[Candidate] = []
        W = self.WEIGHTS

        for cand in candidates:
            score = float(cand.score)
            source = cand.source
            debug_parts = []

            base_score = score
            if self.debug_mode:
                debug_parts.append(f"base={base_score:.1f}")

            # 匹配类型加权
            if "exact_pinyin" in source or source in ("ja_exact", "vocab_exact"):
                score *= W["exact_pinyin"]
                if self.debug_mode:
                    debug_parts.append(f"exact_py*{W['exact_pinyin']}")
            elif "exact_short" in source:
                score *= W["exact_short"]
                if self.debug_mode:
                    debug_parts.append(f"exact_short*{W['exact_short']}")
            elif "prefix" in source:
                score *= W["prefix"]
                if self.debug_mode:
                    debug_parts.append(f"prefix*{W['prefix']}")
            elif "segmented" in source:
                score *= W["segmented"]
                if self.debug_mode:
                    debug_parts.append(f"segmented*{W['segmented']}")

            # 词长加权
            if len(cand.text) >= 2 and source.startswith("dict"):
                score *= W["word_len_bonus"]
                if self.debug_mode:
                    debug_parts.append(f"len_bonus*{W['word_len_bonus']}")
            if len(cand.text) > 8:
                score *= W["long_text_penalty"]
                if self.debug_mode:
                    debug_parts.append(f"long_pen*{W['long_text_penalty']}")

            # 领域词库权重预留
            if "domain" in source:
                score *= W["domain_weight"]
                if self.debug_mode:
                    debug_parts.append(f"domain*{W['domain_weight']}")

            # 用户记忆加权
            if self.user_memory is not None and input_key:
                user_w = self.user_memory.get_user_weight(cand.text, input_key)
                if user_w > 0:
                    score += user_w * W["user_memory_scale"]
                    if self.debug_mode:
                        debug_parts.append(f"user_mem+{user_w * W['user_memory_scale']:.0f}")

                # 最近使用加权
                last_used = self.user_memory.get_last_used_at(cand.text, input_key)
                if last_used > 0 and (now - last_used) < W["recent_use_window_s"]:
                    score += W["recent_use_bonus"]
                    if self.debug_mode:
                        debug_parts.append(f"recent+{W['recent_use_bonus']:.0f}")

            debug_info = " | ".join(debug_parts) if self.debug_mode else ""

            scored_candidates.append(
                Candidate(
                    text=cand.text,
                    composing_covered=cand.composing_covered,
                    score=score,
                    source=cand.source,
                    debug_info=debug_info,
                )
            )

        return sorted(scored_candidates, key=lambda x: x.score, reverse=True)
