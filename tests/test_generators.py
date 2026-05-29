import unittest
from src.input_method.generator.base import Candidate
from src.input_method.generator.pinyin_generator import PinyinCandidateGenerator
from src.input_method.generator.english_generator import EnglishCandidateGenerator

class TestGeneratorsAndRerankers(unittest.TestCase):
    """测试拼音/英文召回器与排序器的正确性"""

    def test_pinyin_generator_exact_pinyin(self) -> None:
        """测试全拼精确匹配"""
        gen = PinyinCandidateGenerator()
        
        # 输入 "nihao" 应该召回 "你好" 和 "你好啊" (前缀)
        cands = gen.generate_candidates(context_before="", composing="nihao")
        self.assertTrue(len(cands) > 0)
        self.assertEqual(cands[0].text, "你好")
        self.assertEqual(cands[0].source, "dict_exact_pinyin")

    def test_pinyin_generator_exact_short(self) -> None:
        """测试简拼精确匹配"""
        gen = PinyinCandidateGenerator()
        
        # 输入 "nh" 应该召回 "你好"
        cands = gen.generate_candidates(context_before="", composing="nh")
        self.assertTrue(len(cands) > 0)
        self.assertTrue(any(c.text == "你好" for c in cands))

    def test_pinyin_generator_prefix_matching(self) -> None:
        """测试全拼与简拼前缀匹配"""
        gen = PinyinCandidateGenerator()
        
        # 全拼前缀：输入 "nih" 应匹配 "nihao" ("你好")
        cands_pfx = gen.generate_candidates(context_before="", composing="nih")
        self.assertTrue(any("你好" in c.text for c in cands_pfx))
        
        # 简拼前缀：输入 "z" 应匹配 "zg" ("中国")
        cands_short_pfx = gen.generate_candidates(context_before="", composing="z")
        self.assertTrue(any("中国" in c.text for c in cands_short_pfx))

    def test_english_generator_prefix(self) -> None:
        """测试英文前缀补全"""
        gen = EnglishCandidateGenerator()
        
        # 输入 "th" 应该匹配 "the", "that", "this", "they", "there", "their" 等
        cands = gen.generate_candidates(context_before="", composing="th")
        self.assertTrue(len(cands) > 0)
        self.assertTrue(any(c.text == "the" for c in cands))
        self.assertTrue(any(c.text == "that" for c in cands))

    def test_english_generator_tokenizer_load_resilient(self) -> None:
        """测试英文生成器对 Tokenizer 路径的不崩溃容错"""
        # 给一个假路径，应该自动降级到内置词库，不崩溃
        gen = EnglishCandidateGenerator(tokenizer_path="invalid_path_to_tokenizer.model")
        cands = gen.generate_candidates(context_before="", composing="para")
        self.assertTrue(any("parameter" in c.text for c in cands))

    def test_pinyin_no_match(self) -> None:
        """测试无匹配的拼音输入"""
        gen = PinyinCandidateGenerator()
        cands = gen.generate_candidates(context_before="", composing="zzzzz")
        self.assertEqual(len(cands), 0)

    def test_english_no_match(self) -> None:
        """测试无匹配的英文输入"""
        gen = EnglishCandidateGenerator()
        cands = gen.generate_candidates(context_before="", composing="xyzqwk")
        self.assertEqual(len(cands), 0)

    def test_pinyin_association(self) -> None:
        """测试拼音联想词生成"""
        gen = PinyinCandidateGenerator()
        cands = gen.generate_candidates(context_before="我", composing="")
        self.assertTrue(len(cands) > 0)
        self.assertTrue(all(c.source == "association" for c in cands))
        self.assertTrue(any(c.text == "们" for c in cands))

    def test_english_association(self) -> None:
        """测试英文联想词生成"""
        gen = EnglishCandidateGenerator()
        cands = gen.generate_candidates(context_before="the", composing="")
        self.assertTrue(len(cands) > 0)
        self.assertTrue(all(c.source == "association" for c in cands))

    def test_empty_composing_empty_context(self) -> None:
        """测试空 composing 和空 context_before"""
        gen_py = PinyinCandidateGenerator()
        gen_en = EnglishCandidateGenerator()
        self.assertEqual(gen_py.generate_candidates(context_before="", composing=""), [])
        self.assertEqual(gen_en.generate_candidates(context_before="", composing=""), [])


if __name__ == "__main__":
    unittest.main()
