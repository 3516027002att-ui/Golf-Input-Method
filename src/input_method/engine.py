from typing import List
import logging
import time
from .config import InputMethodConfig
from .generator.base import Candidate, BaseCandidateGenerator
from .generator.pinyin_generator import PinyinCandidateGenerator
from .generator.english_generator import EnglishCandidateGenerator
from .generator.japanese_generator import JapaneseCandidateGenerator
from .reranker.base import BaseReranker
from .reranker.frequency_reranker import FrequencyReranker
from .reranker.model_reranker import ModelReranker
from .user_memory import UserMemory

logger = logging.getLogger(__name__)

_MAX_HISTORY_LEN = 100


class InputMethodEngine:
    """输入法引擎核心：管理 composing、候选召回、排序、提交与用户记忆。"""

    def __init__(self, config: InputMethodConfig):
        self.config = config
        self.config.validate()

        self.composing = ""
        self.committed_history = ""
        self.candidates: List[Candidate] = []
        self.page_index = 0
        self.last_query_latency_ms = 0.0

        self.user_memory = UserMemory(self.config.user_dict_path)
        self.generator: BaseCandidateGenerator = self._build_generator()
        self.reranker: BaseReranker = self._build_reranker()
        self.refresh_candidates()

    def _build_generator(self) -> BaseCandidateGenerator:
        if self.config.mode == "pinyin":
            return PinyinCandidateGenerator(
                dict_path=self.config.dict_path,
                max_recall=self.config.max_recall,
            )
        if self.config.mode == "english":
            return EnglishCandidateGenerator(
                tokenizer_path=self.config.tokenizer_path,
                max_recall=self.config.max_recall,
            )
        return JapaneseCandidateGenerator(max_recall=self.config.max_recall)

    def _build_reranker(self) -> BaseReranker:
        if self.config.use_model_rerank:
            return ModelReranker(
                model_path=self.config.model_path,
                tokenizer_path=self.config.tokenizer_path,
            )
        return FrequencyReranker(user_memory=self.user_memory)

    def refresh_candidates(self) -> None:
        """公开刷新入口：调用召回和排序流程更新候选列表，并重置页码。"""
        start_time = time.perf_counter()
        raw_cands = self.generator.generate_candidates(
            context_before=self.committed_history,
            composing=self.composing,
        )
        self.candidates = self.reranker.rerank(
            context_before=self.committed_history,
            composing=self.composing,
            candidates=raw_cands,
        )
        self.page_index = 0
        self.last_query_latency_ms = (time.perf_counter() - start_time) * 1000.0

    # 兼容旧调用；新代码应使用 refresh_candidates。
    def _update_candidates(self) -> None:
        self.refresh_candidates()

    def handle_char(self, char: str) -> bool:
        if len(char) != 1 or char in ("\r", "\n", "\b", " "):
            return False
        if self.config.mode in ("pinyin", "japanese") and not (char.isascii() and char.isalpha()):
            return False

        self.composing += char.lower()
        self.refresh_candidates()
        return True

    def handle_backspace(self) -> bool:
        if self.composing:
            self.composing = self.composing[:-1]
            self.refresh_candidates()
            return True
        return False

    def handle_enter(self) -> bool:
        if self.composing:
            raw_text = self.composing
            self.commit_text(raw_text)
            return True
        return False

    def handle_space(self) -> bool:
        current_page = self.get_current_page_candidates()
        if current_page:
            self.select_candidate_on_page(0)
            return True
        if self.composing:
            return False
        self.commit_text(" ")
        return True

    def handle_candidate_select(self, key_num: int) -> bool:
        return self.select_candidate_on_page(key_num - 1)

    def handle_page_next(self) -> bool:
        if self.page_index < self.total_pages() - 1:
            self.page_index += 1
            return True
        return False

    def handle_page_prev(self) -> bool:
        if self.page_index > 0:
            self.page_index -= 1
            return True
        return False

    def switch_mode(self, new_mode: str) -> None:
        self.config.mode = new_mode
        self.config.validate()
        self.generator = self._build_generator()
        self.reranker = self._build_reranker()
        self.composing = ""
        self.candidates = []
        self.page_index = 0
        self.refresh_candidates()
        logger.info("输入模式已切换为: %s", new_mode)

    def set_model_rerank_enabled(self, enabled: bool) -> None:
        """开启或关闭模型重排；当前真实模型未接入时仍会退避。"""
        self.config.use_model_rerank = enabled
        self.reranker = self._build_reranker()
        self.refresh_candidates()

    def clear_user_memory(self) -> None:
        self.user_memory.clear()
        self.refresh_candidates()

    def commit_text(self, text: str) -> None:
        self.committed_history += text
        if len(self.committed_history) > _MAX_HISTORY_LEN:
            self.committed_history = self.committed_history[-_MAX_HISTORY_LEN:]
        self.composing = ""
        self.refresh_candidates()

    def select_candidate_on_page(self, index_on_page: int) -> bool:
        current_page = self.get_current_page_candidates()
        if 0 <= index_on_page < len(current_page):
            selected_cand = current_page[index_on_page]
            input_key = self.composing
            if input_key:
                self.user_memory.record_selection(selected_cand.text, input_key)
            self.commit_text(selected_cand.text)
            return True
        return False

    def get_current_page_candidates(self) -> List[Candidate]:
        start = self.page_index * self.config.page_size
        end = start + self.config.page_size
        return self.candidates[start:end]

    def total_pages(self) -> int:
        if not self.candidates:
            return 0
        return (len(self.candidates) + self.config.page_size - 1) // self.config.page_size

    def clear(self) -> None:
        self.composing = ""
        self.committed_history = ""
        self.candidates = []
        self.page_index = 0
        self.refresh_candidates()
