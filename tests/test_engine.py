import unittest
from src.input_method.config import InputMethodConfig
from src.input_method.engine import InputMethodEngine
from src.input_method.generator.base import Candidate

class TestInputMethodEngine(unittest.TestCase):
    """测试输入法引擎状态机及交互行为"""

    def setUp(self) -> None:
        self.config = InputMethodConfig(mode="pinyin", page_size=5)
        self.engine = InputMethodEngine(self.config)

    def test_initial_state(self) -> None:
        """测试初始状态是否正确"""
        self.assertEqual(self.engine.composing, "")
        self.assertEqual(self.engine.committed_history, "")
        self.assertEqual(self.engine.page_index, 0)
        # 初始时可能有根据空上下文触发的联想词候选
        self.assertTrue(isinstance(self.engine.candidates, list))

    def test_handle_char(self) -> None:
        """测试输入字符缓冲及候选词更新"""
        # 输入 'n'
        changed = self.engine.handle_char("n")
        self.assertTrue(changed)
        self.assertEqual(self.engine.composing, "n")
        self.assertTrue(len(self.engine.candidates) > 0)
        
        # 输入 'i'
        self.engine.handle_char("i")
        self.assertEqual(self.engine.composing, "ni")
        
        # 在拼音模式下，不允许输入非英文字母
        changed_bad = self.engine.handle_char("1")
        self.assertFalse(changed_bad)
        self.assertEqual(self.engine.composing, "ni")

    def test_handle_backspace(self) -> None:
        """测试退格键删除逻辑"""
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        self.assertEqual(self.engine.composing, "ni")
        
        changed = self.engine.handle_backspace()
        self.assertTrue(changed)
        self.assertEqual(self.engine.composing, "n")
        
        self.engine.handle_backspace()
        self.assertEqual(self.engine.composing, "")
        
        # 缓冲区已空，退格不应该发生改变
        changed_empty = self.engine.handle_backspace()
        self.assertFalse(changed_empty)

    def test_handle_enter(self) -> None:
        """测试回车提交原始字母"""
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        
        changed = self.engine.handle_enter()
        self.assertTrue(changed)
        self.assertEqual(self.engine.committed_history, "ni")
        self.assertEqual(self.engine.composing, "")

    def test_handle_space(self) -> None:
        """测试空格确认首选词"""
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        
        # 默认首选词应该是 "你"
        current_page = self.engine.get_current_page_candidates()
        first_cand = current_page[0].text
        
        changed = self.engine.handle_space()
        self.assertTrue(changed)
        self.assertEqual(self.engine.committed_history, first_cand)
        self.assertEqual(self.engine.composing, "")

        # 当 composing 为空且没有候选联想词时，按下空格应直接将空格符提交上屏
        self.engine.clear()
        self.engine.handle_space()
        self.assertEqual(self.engine.committed_history, " ")

    def test_candidate_selection_by_number(self) -> None:
        """测试按数字键选择候选词"""
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        
        # 获取第 2 个候选词
        current_page = self.engine.get_current_page_candidates()
        self.assertTrue(len(current_page) >= 2)
        second_cand = current_page[1].text
        
        # 选择第 2 个候选词 (按键 '2')
        changed = self.engine.handle_candidate_select(2)
        self.assertTrue(changed)
        self.assertEqual(self.engine.committed_history, second_cand)
        self.assertEqual(self.engine.composing, "")

    def test_paging(self) -> None:
        """测试翻页逻辑"""
        # 输入 'a' 召回许多词，测试翻页
        self.engine.handle_char("y")
        total_cands = len(self.engine.candidates)
        
        if total_cands > 5:
            # 第一页
            self.assertEqual(self.engine.page_index, 0)
            self.assertTrue(self.engine.handle_page_next())
            self.assertEqual(self.engine.page_index, 1)
            
            # 翻回前页
            self.assertTrue(self.engine.handle_page_prev())
            self.assertEqual(self.engine.page_index, 0)
            
            # 不能继续往前翻页
            self.assertFalse(self.engine.handle_page_prev())
            self.assertEqual(self.engine.page_index, 0)

    def test_association_trigger(self) -> None:
        """测试上屏后联想词的触发"""
        self.engine.commit_text("我")
        # 应该自动触发联想词，如 "们"、"觉得"
        cands = self.engine.candidates
        self.assertTrue(len(cands) > 0)
        self.assertTrue(any(c.source == "association" for c in cands))

    def test_model_fallback(self) -> None:
        """测试 ML 排词模型未加载时自动退避"""
        fallback_config = InputMethodConfig(
            mode="pinyin",
            use_model_rerank=True,
            model_path="non_existent_model_file.pt"
        )
        engine = InputMethodEngine(fallback_config)
        # 验证未报错，成功初始化
        self.assertFalse(engine.reranker.model_loaded)
        self.assertTrue("not found" in engine.reranker.model_info)
        
        # 仍然能正常打字和获取候选词
        engine.handle_char("n")
        self.assertTrue(len(engine.candidates) > 0)

    def test_switch_mode(self) -> None:
        """测试 switch_mode 切换输入模式（仅中/英双路）"""
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        self.assertEqual(self.engine.composing, "ni")

        # 切换到英文模式
        self.engine.switch_mode("english")
        self.assertEqual(self.engine.config.mode, "english")
        self.assertEqual(self.engine.composing, "")  # composing 应被清空

        # 英文模式：直通不拦截 (handle_char 返回 False)
        self.assertFalse(self.engine.handle_char("t"))
        self.assertEqual(self.engine.composing, "")  # composing 仍为空

        # 切换回拼音模式
        self.engine.switch_mode("pinyin")
        self.assertEqual(self.engine.config.mode, "pinyin")

    def test_switch_mode_rejects_japanese(self) -> None:
        """测试 switch_mode 不允许切换到日语（UI 层面隐藏）"""
        with self.assertRaises(ValueError):
            self.engine.switch_mode("japanese")

    def test_switch_language_ja(self) -> None:
        """测试 switch_language('ja') 内部接口可切换到日语（预留）"""
        self.engine.switch_language("ja")
        self.assertEqual(self.engine.config.mode, "japanese")
        # 确认日语 generator 可用
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        self.engine.handle_char("h")
        self.engine.handle_char("o")
        self.engine.handle_char("n")
        cands = self.engine.get_current_page_candidates()
        # 日语模式应该能召回候选
        self.assertTrue(len(cands) > 0)

    def test_handle_enter_empty(self) -> None:
        """测试空缓冲区时按回车"""
        changed = self.engine.handle_enter()
        self.assertFalse(changed)
        self.assertEqual(self.engine.committed_history, "")

    def test_candidate_select_out_of_range(self) -> None:
        """测试选择不存在的候选词"""
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        # 尝试选择第 99 个候选（肯定不存在）
        changed = self.engine.handle_candidate_select(99)
        self.assertFalse(changed)

    def test_committed_history_sliding_window(self) -> None:
        """测试 committed_history 不会无限增长"""
        # 提交大量文本
        for i in range(200):
            self.engine.commit_text("字")
        # 应该被截断到合理长度
        self.assertLessEqual(len(self.engine.committed_history), 100)
        # 末尾应该仍然是 "字"
        self.assertTrue(self.engine.committed_history.endswith("字"))

    def test_english_mode_passthrough(self) -> None:
        """测试英文模式：输入直通，不进入 IME 候选流程"""
        eng_config = InputMethodConfig(mode="english", page_size=5)
        eng_engine = InputMethodEngine(eng_config)

        # 英文模式下 handle_char 返回 False（不拦截）
        self.assertFalse(eng_engine.handle_char("t"))
        self.assertFalse(eng_engine.handle_char("h"))
        # composing 为空，不会产生候选
        self.assertEqual(eng_engine.composing, "")

        # 空格在无 composing 时提交空格字符
        eng_engine.handle_space()
        self.assertIn(" ", eng_engine.committed_history)

    def test_config_validate_invalid_mode(self) -> None:
        """测试非法 mode 值应抛出 ValueError"""
        with self.assertRaises(ValueError):
            bad_config = InputMethodConfig(mode="wubi")
            bad_config.validate()

    def test_config_validate_invalid_page_size(self) -> None:
        """测试 page_size <= 0 应抛出 ValueError"""
        with self.assertRaises(ValueError):
            bad_config = InputMethodConfig(page_size=0)
            bad_config.validate()


if __name__ == "__main__":
    unittest.main()
