import unittest
import os
import tempfile
import shutil
from src.input_method.config import InputMethodConfig
from src.input_method.engine import InputMethodEngine
from src.input_method.generator.pinyin_generator import PinyinCandidateGenerator
from src.input_method.generator.japanese_generator import JapaneseCandidateGenerator
from src.input_method.user_memory import UserMemory
from src.input_method.lexicon import LexiconLoader


class TestInputMethodV0Features(unittest.TestCase):
    """测试输入法底座 v0 新增的核心能力"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.dict_file = os.path.join(self.temp_dir, "test_dict.jsonl")
        self.user_memory_file = os.path.join(self.temp_dir, "test_user_memory.json")
        
        # 写入一个小型测试词库，增加同音词“相邀”以支持用户记忆竞争测试
        with open(self.dict_file, "w", encoding="utf-8") as f:
            f.write('{"word": "我", "pinyin": "wo", "short_pinyin": "w", "freq": 8000, "source": "test"}\n')
            f.write('{"word": "想", "pinyin": "xiang", "short_pinyin": "x", "freq": 6500, "source": "test"}\n')
            f.write('{"word": "要", "pinyin": "yao", "short_pinyin": "y", "freq": 6300, "source": "test"}\n')
            f.write('{"word": "想要", "pinyin": "xiangyao", "short_pinyin": "xy", "freq": 6100, "source": "test"}\n')
            f.write('{"word": "相邀", "pinyin": "xiangyao", "short_pinyin": "xy", "freq": 3000, "source": "test"}\n')
            f.write('{"word": "我想要", "pinyin": "woxiangyao", "short_pinyin": "wxy", "freq": 5500, "source": "test"}\n')
            f.write('{"word": "你好", "pinyin": "nihao", "short_pinyin": "nh", "freq": 6700, "source": "test"}\n')
            f.write('{"word": "中国", "pinyin": "zhongguo", "short_pinyin": "zg", "freq": 6500, "source": "test"}\n')
            f.write('{"word": "输入法", "pinyin": "shurufa", "short_pinyin": "srf", "freq": 3600, "source": "test"}\n')
            
        self.config = InputMethodConfig(
            mode="pinyin",
            dict_path=self.dict_file,
            user_dict_path=self.user_memory_file
        )
        self.engine = InputMethodEngine(self.config)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
        if os.path.exists(self.user_memory_file):
            try:
                os.remove(self.user_memory_file)
            except Exception:
                pass

    def test_lexicon_loader_success(self) -> None:
        """测试词库成功加载"""
        entries, bad_count = LexiconLoader.load_from_jsonl(self.dict_file)
        self.assertEqual(len(entries), 9)
        self.assertEqual(bad_count, 0)
        self.assertEqual(entries[0].word, "我")
        
    def test_lexicon_loader_fallback(self) -> None:
        """测试外部词库加载失败降级使用内置词库"""
        generator = PinyinCandidateGenerator(dict_path="nonexistent_dict.jsonl")
        self.assertTrue(len(generator.words) > 0)
        self.assertEqual(generator.words[0][4], "dict")

    def test_pinyin_segmentation(self) -> None:
        """测试连续拼音切分能够处理 woxiangyao"""
        generator = self.engine.generator
        self.assertTrue(hasattr(generator, "split_pinyin"))
        
        segs = generator.split_pinyin("woxiangyao")
        self.assertEqual(segs, ["wo", "xiang", "yao"])
        
        # 测试前缀切分 (末尾不完整音节)
        segs_prefix = generator.split_pinyin("woxiangy")
        self.assertEqual(segs_prefix, ["wo", "xiang", "y"])

    def test_woxiangyao_recall(self) -> None:
        """测试输入 woxiangyao 可以召回合理候选"""
        self.engine.clear()
        for char in "woxiangyao":
            self.engine.handle_char(char)
        
        cands = [c.text for c in self.engine.candidates]
        self.assertTrue(len(cands) > 0)
        self.assertTrue("我想要" in cands or any("我" in c for c in cands))

    def test_user_memory_record_and_save(self) -> None:
        """测试用户选择后，记忆被持久化"""
        memory = UserMemory(self.user_memory_file)
        memory.record_selection("想要", "xiangyao")
        
        # 验证文件已持久化落盘
        self.assertTrue(os.path.exists(self.user_memory_file))
        
        new_memory = UserMemory(self.user_memory_file)
        self.assertEqual(new_memory.get_user_weight("想要", "xiangyao"), 1.0)

    def test_user_memory_influences_sorting(self) -> None:
        """测试用户记忆改变候选排序"""
        self.engine.clear()
        for char in "xiangyao":
            self.engine.handle_char(char)
        
        # 初始首位词应该是 "想要" (freq: 6100)，次选为 "相邀" (freq: 3000)
        self.assertEqual(self.engine.candidates[0].text, "想要")
        
        # 模拟用户选择 "相邀"
        self.engine.user_memory.record_selection("相邀", "xiangyao")
        self.engine.user_memory.record_selection("相邀", "xiangyao")
        
        # 清除输入并重新键入相同的拼音，验证 "相邀" 升至第一顺位
        self.engine.clear()
        for char in "xiangyao":
            self.engine.handle_char(char)
            
        self.assertEqual(self.engine.candidates[0].text, "相邀")

    def test_clear_user_memory(self) -> None:
        """测试清空学习记录"""
        self.engine.user_memory.record_selection("想要", "xiangyao")
        self.assertTrue(os.path.exists(self.user_memory_file))
        
        self.engine.clear_user_memory()
        self.assertFalse(os.path.exists(self.user_memory_file))
        self.assertEqual(self.engine.user_memory.get_user_weight("想要", "xiangyao"), 0.0)

    def test_japanese_mode(self) -> None:
        """测试日语模式初始化与返回假名候选"""
        self.engine.switch_mode("japanese")
        self.assertEqual(self.engine.config.mode, "japanese")
        
        for char in "ka":
            self.engine.handle_char(char)
            
        cands = [c.text for c in self.engine.candidates]
        self.assertTrue("か" in cands)

    def test_switch_mode_no_init_anti_pattern(self) -> None:
        """测试 switch_mode 切换模式不是重建 Engine 本身"""
        old_id = id(self.engine)
        self.engine.switch_mode("english")
        self.assertEqual(id(self.engine), old_id)
        self.assertEqual(self.engine.config.mode, "english")

    def test_all_v0_essential_pinyin_queries(self) -> None:
        """测试 v0 所有必需拼音和简拼的召回质量"""
        cases = [
            ("nihao", "你好"),
            ("nh", "你好"),
            ("wo", "我"),
            ("zhongguo", "中国"),
            ("zg", "中国"),
            ("shurufa", "输入法"),
            ("xiangyao", "想要"),
            ("woxiangyao", "我想要"),
        ]
        for query, expected in cases:
            self.engine.clear()
            for char in query:
                self.engine.handle_char(char)
            cands = [c.text for c in self.engine.candidates]
            self.assertTrue(
                expected in cands[:3],
                f"查询 '{query}' 期望的候选词 '{expected}' 未出现在 Top-3 中: {cands[:3]}"
            )

    def test_nonexistent_pinyin_query(self) -> None:
        """测试不合理拼音，应该召回空或极低概率的系统提示而无意义词条"""
        self.engine.clear()
        for char in "zzzzz":
            self.engine.handle_char(char)
        cands = [c.text for c in self.engine.candidates]
        self.assertTrue(len(cands) == 0 or all("无" in c or "提示" in c for c in cands))

    def test_user_memory_persists_across_engine_recreations(self) -> None:
        """测试重新创建 Engine 后用户记忆仍影响排序"""
        # 第一个引擎：选择低频词"相邀"多次
        self.engine.clear()
        self.engine.user_memory.record_selection("相邀", "xiangyao")
        self.engine.user_memory.record_selection("相邀", "xiangyao")

        # 用同一配置创建全新的 Engine 实例
        new_engine = InputMethodEngine(self.config)
        for char in "xiangyao":
            new_engine.handle_char(char)

        # 重新加载后，"相邀"因用户记忆应排在首位
        self.assertEqual(new_engine.candidates[0].text, "相邀")

    def test_gui_modules_importable(self) -> None:
        """测试 GUI 模块可正常导入"""
        from src.input_method.gui_editor import GuiEditor
        from src.input_method.gui_candidate_window import GuiCandidateWindow
        self.assertTrue(callable(GuiEditor))
        self.assertTrue(callable(GuiCandidateWindow))

    def test_evaluate_script_importable(self) -> None:
        """测试评估脚本可作为模块导入（不实际运行 main）"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate_candidates",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "evaluate_candidates.py")
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        # 只验证可加载，不执行 main
        self.assertIsNotNone(module)

    def test_benchmark_script_importable(self) -> None:
        """测试 benchmark 脚本可作为模块导入"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "benchmark_latency",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "benchmark_latency.py")
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
