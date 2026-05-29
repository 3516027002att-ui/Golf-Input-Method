from typing import List, Optional
import logging
import time
from .config import InputMethodConfig
from .generator.base import Candidate, BaseCandidateGenerator
from .generator.pinyin_generator import PinyinCandidateGenerator
from .generator.english_generator import EnglishCandidateGenerator
from .reranker.base import BaseReranker
from .reranker.frequency_reranker import FrequencyReranker
from .reranker.model_reranker import ModelReranker

logger = logging.getLogger(__name__)

# committed_history 滑动窗口最大长度（字符数）
_MAX_HISTORY_LEN = 100


class InputMethodEngine:
    """
    输入法引擎核心。
    
    驱动输入缓冲区的状态机，控制字符追加、删除、回车上屏、候选词选择和分页。
    """

    def __init__(self, config: InputMethodConfig):
        self.config = config
        self.config.validate()

        # 状态变量
        self.composing = ""           # 输入缓冲区 raw 文本 (如 "nihao")
        self.committed_history = ""    # 已提交历史文本
        self.candidates: List[Candidate] = []  # 当前候选词列表
        self.page_index = 0           # 候选词翻页索引 (0-based)
        
        # 调试/耗时统计
        self.last_query_latency_ms = 0.0

        # 初始化 Generator
        self.generator: BaseCandidateGenerator
        if self.config.mode == "pinyin":
            self.generator = PinyinCandidateGenerator(max_recall=self.config.max_recall)
        else:
            self.generator = EnglishCandidateGenerator(
                tokenizer_path=self.config.tokenizer_path,
                max_recall=self.config.max_recall
            )

        # 初始化 Reranker
        self.reranker: BaseReranker
        if self.config.use_model_rerank:
            self.reranker = ModelReranker(
                model_path=self.config.model_path,
                tokenizer_path=self.config.tokenizer_path
            )
        else:
            self.reranker = FrequencyReranker()

        # 初始时，触发一次空缓冲区的联想词更新
        self._update_candidates()

    def _update_candidates(self) -> None:
        """调用召回和排序流程更新候选列表，并重置页码"""
        start_time = time.time()
        
        # 1. 召回
        raw_cands = self.generator.generate_candidates(
            context_before=self.committed_history,
            composing=self.composing
        )
        
        # 2. 重排
        self.candidates = self.reranker.rerank(
            context_before=self.committed_history,
            composing=self.composing,
            candidates=raw_cands
        )
        
        self.page_index = 0
        self.last_query_latency_ms = (time.time() - start_time) * 1000.0

    # --- 输入事件处理器 ---

    def handle_char(self, char: str) -> bool:
        """
        处理普通字符输入。
        返回 True 表示状态已更新（通常需要重绘界面）。
        """
        # 只接收合法字符 (拼音模式下多为 a-z，英文模式下为 a-z, A-Z, 标点等)
        # 不处理回车、空格、退格等，这些由专用 handle 处理
        if len(char) != 1 or char in ("\r", "\n", "\b", " "):
            return False

        # 中文拼音模式下只允许小写英文字母
        if self.config.mode == "pinyin" and not char.isalpha():
            return False

        self.composing += char
        self._update_candidates()
        return True

    def handle_backspace(self) -> bool:
        """
        处理退格键。
        返回 True 表示状态已更新。
        """
        if self.composing:
            self.composing = self.composing[:-1]
            self._update_candidates()
            return True
        return False

    def handle_enter(self) -> bool:
        """
        处理回车键：将输入缓冲区的 raw 文本直接上屏。
        返回 True 表示状态已更新。
        """
        if self.composing:
            # 拼音模式下上屏 raw 拼音字母，英文模式上屏 raw 字母
            raw_text = self.composing
            self.commit_text(raw_text)
            return True
        return False

    def handle_space(self) -> bool:
        """
        处理空格键：
        若有候选词，上屏首选词；若无候选词，则将空格作为普通字符上屏。
        返回 True 表示状态已更新。
        """
        current_page = self.get_current_page_candidates()
        if current_page:
            # 默认上屏当前页的第一顺位候选词
            self.select_candidate_on_page(0)
            return True
        else:
            # 无候选，直接提交空格
            # 拼音模式下，若 composing 不为空，空格其实也是选择首词
            if self.composing:
                return False
            self.commit_text(" ")
            return True

    def handle_candidate_select(self, key_num: int) -> bool:
        """
        按数字键（1-9）选择候选词。
        key_num: 用户按下的数字（1 表示第1个，9 表示第9个）。
        返回 True 表示选择成功且已更新状态。
        """
        # 1-indexed to 0-indexed
        index_on_page = key_num - 1
        return self.select_candidate_on_page(index_on_page)

    def handle_page_next(self) -> bool:
        """翻下页，返回 True 表示翻页成功"""
        if self.page_index < self.total_pages() - 1:
            self.page_index += 1
            return True
        return False

    def handle_page_prev(self) -> bool:
        """翻上页，返回 True 表示翻页成功"""
        if self.page_index > 0:
            self.page_index -= 1
            return True
        return False

    # --- 辅助方法 ---

    def switch_mode(self, new_mode: str) -> None:
        """切换输入模式（pinyin / english），重建 generator 和 reranker，清空输入状态。"""
        self.config.mode = new_mode
        self.config.validate()

        # 重建 Generator
        if self.config.mode == "pinyin":
            self.generator = PinyinCandidateGenerator(max_recall=self.config.max_recall)
        else:
            self.generator = EnglishCandidateGenerator(
                tokenizer_path=self.config.tokenizer_path,
                max_recall=self.config.max_recall
            )

        # 重建 Reranker
        if self.config.use_model_rerank:
            self.reranker = ModelReranker(
                model_path=self.config.model_path,
                tokenizer_path=self.config.tokenizer_path
            )
        else:
            self.reranker = FrequencyReranker()

        # 清空输入状态（保留 committed_history 以便联想）
        self.composing = ""
        self.candidates = []
        self.page_index = 0
        self._update_candidates()
        logger.info("输入模式已切换为: %s", new_mode)

    def commit_text(self, text: str) -> None:
        """将文字提交上屏，清空输入缓冲区，触发联想"""
        self.committed_history += text
        # 保持滑动窗口，只保留尾部
        if len(self.committed_history) > _MAX_HISTORY_LEN:
            self.committed_history = self.committed_history[-_MAX_HISTORY_LEN:]
        self.composing = ""
        self._update_candidates()  # 触发联想词更新

    def select_candidate_on_page(self, index_on_page: int) -> bool:
        """选择当前页中的某个候选词并上屏"""
        current_page = self.get_current_page_candidates()
        if 0 <= index_on_page < len(current_page):
            selected_cand = current_page[index_on_page]
            
            # 部分拼音匹配的进阶逻辑 (在此做极简且安全的上屏处理):
            # 将候选词上屏，清空 composing，触发下一轮联想
            self.commit_text(selected_cand.text)
            return True
        return False

    def get_current_page_candidates(self) -> List[Candidate]:
        """获取当前页的候选词列表"""
        start = self.page_index * self.config.page_size
        end = start + self.config.page_size
        return self.candidates[start:end]

    def total_pages(self) -> int:
        """获取候选词的总页数"""
        if not self.candidates:
            return 0
        return (len(self.candidates) + self.config.page_size - 1) // self.config.page_size

    def clear(self) -> None:
        """清空引擎状态"""
        self.composing = ""
        self.committed_history = ""
        self.candidates = []
        self.page_index = 0
        self._update_candidates()
