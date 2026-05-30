import os
import json
import tempfile
import unittest

from src.input_method.config import InputMethodConfig
from src.input_method.engine import InputMethodEngine
from src.input_method.generator.base import Candidate
from src.input_method.lexicon import LexiconLoader
from src.input_method.user_memory import UserMemory


class TestEnhancedEngine(unittest.TestCase):
    """测试引擎增强功能"""

    def setUp(self) -> None:
        self.config = InputMethodConfig(mode="pinyin", dict_path=None)
        self.engine = InputMethodEngine(self.config)

    def test_handle_escape(self) -> None:
        """测试 Escape 清空 composing"""
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        self.assertEqual(self.engine.composing, "ni")
        result = self.engine.handle_escape()
        self.assertTrue(result)
        self.assertEqual(self.engine.composing, "")
        self.assertEqual(len(self.engine.candidates), 0)

    def test_handle_escape_empty(self) -> None:
        """测试空 composing 时 Escape 返回 False"""
        result = self.engine.handle_escape()
        self.assertFalse(result)

    def test_handle_delete(self) -> None:
        """测试 Delete 键行为"""
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        result = self.engine.handle_delete()
        self.assertTrue(result)
        self.assertEqual(self.engine.composing, "n")

    def test_learning_disabled(self) -> None:
        """测试关闭学习功能后不记录用户选择"""
        # 使用独立的临时文件避免共享状态
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            config = InputMethodConfig(mode="pinyin", dict_path=None, user_dict_path=tmp_path)
            engine = InputMethodEngine(config)
            engine.set_learning_enabled(False)
            for ch in "nihao":
                engine.handle_char(ch)
            cands = engine.get_current_page_candidates()
            if cands:
                engine.select_candidate_on_page(0)
            # 因为学习关闭了，用户记忆应该为空
            weight = engine.user_memory.get_user_weight("你好", "nihao")
            self.assertEqual(weight, 0.0)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_debug_scores(self) -> None:
        """测试调试分数输出"""
        self.engine.handle_char("n")
        self.engine.handle_char("i")
        scores = self.engine.get_debug_scores()
        self.assertIsInstance(scores, list)
        if scores:
            self.assertIn("text", scores[0])
            self.assertIn("score", scores[0])
            self.assertIn("source", scores[0])


class TestEnhancedConfig(unittest.TestCase):
    """测试增强配置系统"""

    def test_config_from_json_file(self) -> None:
        """测试从 JSON 文件加载配置"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({
                "mode": "english",
                "page_size": 10,
                "learning_enabled": False,
                "log_level": "DEBUG",
            }, f)
            f.flush()
            path = f.name

        try:
            config = InputMethodConfig.from_json_file(path)
            self.assertEqual(config.mode, "english")
            self.assertEqual(config.page_size, 10)
            self.assertFalse(config.learning_enabled)
            self.assertEqual(config.log_level, "DEBUG")
        finally:
            os.unlink(path)

    def test_config_from_nonexistent_file(self) -> None:
        """测试不存在的文件回退到默认配置"""
        config = InputMethodConfig.from_json_file("/nonexistent/path.json")
        self.assertEqual(config.mode, "pinyin")
        self.assertEqual(config.page_size, 5)

    def test_config_validate_enhanced(self) -> None:
        """测试增强的验证"""
        config = InputMethodConfig(log_level="INVALID")
        with self.assertRaises(ValueError):
            config.validate()

        config2 = InputMethodConfig(candidate_layout="invalid")
        with self.assertRaises(ValueError):
            config2.validate()


class TestEnhancedUserMemory(unittest.TestCase):
    """测试用户记忆增强功能"""

    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        self.tmp.close()
        self.memory = UserMemory(self.tmp.name)

    def tearDown(self) -> None:
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_delete_word(self) -> None:
        """测试删除用户词"""
        self.memory.record_selection("你好", "nihao")
        self.assertTrue(self.memory.get_user_weight("你好", "nihao") > 0)
        result = self.memory.delete_word("你好", "nihao")
        self.assertTrue(result)
        self.assertEqual(self.memory.get_user_weight("你好", "nihao"), 0.0)

    def test_delete_nonexistent(self) -> None:
        """测试删除不存在的词返回 False"""
        result = self.memory.delete_word("不存在", "bcs")
        self.assertFalse(result)

    def test_disable_word(self) -> None:
        """测试禁用用户词"""
        self.memory.record_selection("你好", "nihao")
        result = self.memory.disable_word("你好", "nihao")
        self.assertTrue(result)
        self.assertEqual(self.memory.get_user_weight("你好", "nihao"), 0.0)

    def test_export_import(self) -> None:
        """测试导出和导入"""
        self.memory.record_selection("你好", "nihao")
        self.memory.record_selection("世界", "shijie")

        export_path = self.tmp.name + ".export.json"
        try:
            count = self.memory.export_to_file(export_path)
            self.assertGreater(count, 0)

            new_memory = UserMemory(self.tmp.name + ".new.json")
            added = new_memory.import_from_file(export_path)
            self.assertGreater(added, 0)
            self.assertTrue(new_memory.get_user_weight("你好", "nihao") > 0)
        finally:
            for p in [export_path, self.tmp.name + ".new.json"]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_get_all_words(self) -> None:
        """测试获取所有用户词"""
        self.memory.record_selection("你好", "nihao")
        all_words = self.memory.get_all_words()
        self.assertIn("nihao", all_words)
        self.assertIn("你好", all_words["nihao"])


class TestLexiconBadLineTolerance(unittest.TestCase):
    """测试词库坏行容错"""

    def test_bad_lines_skipped(self) -> None:
        """测试坏行被跳过并计数"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            # 有效行
            f.write(json.dumps({"word": "好", "pinyin": "hao", "freq": 100}) + "\n")
            # 坏行：JSON 解析失败
            f.write("invalid json\n")
            # 坏行：缺少必需字段
            f.write(json.dumps({"pinyin": "no_word"}) + "\n")
            # 有效行
            f.write(json.dumps({"word": "你", "pinyin": "ni", "freq": 200}) + "\n")
            f.flush()
            path = f.name

        try:
            entries, bad_count = LexiconLoader.load_from_jsonl(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(bad_count, 2)
        finally:
            os.unlink(path)


class TestEndToEndSmoke(unittest.TestCase):
    """端到端冒烟测试：验证常见输入场景"""

    def test_jintian(self) -> None:
        """测试 'jintian' -> '今天'"""
        config = InputMethodConfig(mode="pinyin", dict_path=None)
        engine = InputMethodEngine(config)
        for ch in "jintian":
            engine.handle_char(ch)
        cands = engine.candidates
        texts = [c.text for c in cands]
        # 今天在内置词库中
        self.assertIn("今天", texts)

    def test_xiangyao(self) -> None:
        """测试 'xiangyao' -> '想要'"""
        config = InputMethodConfig(mode="pinyin", dict_path=None)
        engine = InputMethodEngine(config)
        for ch in "xiangyao":
            engine.handle_char(ch)
        cands = engine.candidates
        texts = [c.text for c in cands]
        self.assertIn("想要", texts)

    def test_nihao(self) -> None:
        """测试 'nihao' -> '你好'"""
        config = InputMethodConfig(mode="pinyin", dict_path=None)
        engine = InputMethodEngine(config)
        for ch in "nihao":
            engine.handle_char(ch)
        cands = engine.candidates
        texts = [c.text for c in cands]
        self.assertIn("你好", texts)
        # 你好应该排在首位
        self.assertEqual(texts[0], "你好")


if __name__ == "__main__":
    unittest.main()
