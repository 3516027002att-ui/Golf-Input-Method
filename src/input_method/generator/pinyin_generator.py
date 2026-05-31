from typing import Dict, List, Tuple
import logging
import os
from .base import BaseCandidateGenerator, Candidate
from ..lexicon import LexiconLoader

logger = logging.getLogger(__name__)

# 预内置 fallback 常用词库：(词语, 全拼, 简拼, 静态词频, 来源)
RAW_WORDS: List[Tuple[str, str, str, float, str]] = [
    ("的", "de", "d", 9900, "dict"),
    ("是", "shi", "s", 9500, "dict"),
    ("了", "le", "l", 9000, "dict"),
    ("在", "zai", "z", 8500, "dict"),
    ("我", "wo", "w", 8000, "dict"),
    ("你", "ni", "n", 7900, "dict"),
    ("他", "ta", "t", 7000, "dict"),
    ("她", "ta", "t", 6500, "dict"),
    ("它", "ta", "t", 6000, "dict"),
    ("们", "men", "m", 6800, "dict"),
    ("我们", "women", "wm", 7800, "dict"),
    ("你们", "nimen", "nm", 7300, "dict"),
    ("他们", "tamen", "tm", 7100, "dict"),
    ("有", "you", "y", 6800, "dict"),
    ("个", "ge", "g", 6600, "dict"),
    ("好", "hao", "h", 6500, "dict"),
    ("你好", "nihao", "nh", 6700, "dict"),
    ("你好啊", "nihaoa", "nha", 4500, "dict"),
    ("这", "zhe", "z", 6400, "dict"),
    ("想", "xiang", "x", 6500, "dict"),
    ("要", "yao", "y", 6300, "dict"),
    ("想要", "xiangyao", "xy", 6100, "dict"),
    ("国", "guo", "g", 6200, "dict"),
    ("中国", "zhongguo", "zg", 6500, "dict"),
    ("人", "ren", "r", 6100, "dict"),
    ("和", "he", "h", 6000, "dict"),
    ("用", "yong", "y", 5900, "dict"),
    ("作", "zuo", "z", 5800, "dict"),
    ("时", "shi", "s", 5700, "dict"),
    ("去", "qu", "q", 5600, "dict"),
    ("来", "lai", "l", 5500, "dict"),
    ("会", "hui", "h", 5400, "dict"),
    ("能", "neng", "n", 5300, "dict"),
    ("对", "dui", "d", 5200, "dict"),
    ("都", "dou", "d", 5100, "dict"),
    ("多", "duo", "d", 5000, "dict"),
    ("少", "shao", "s", 4500, "dict"),
    ("多少", "duoshao", "ds", 4700, "dict"),
    ("没", "mei", "m", 4400, "dict"),
    ("没有", "meiyou", "my", 4800, "dict"),
    ("怎么", "zenme", "zm", 4300, "dict"),
    ("什么", "shenme", "sm", 4600, "dict"),
    ("为什么", "weishenme", "wsm", 4200, "dict"),
    ("觉得", "juede", "jd", 4100, "dict"),
    ("知道", "zhidao", "zd", 4050, "dict"),
    ("可以", "keyi", "ky", 4000, "dict"),
    ("现在", "xianzai", "xz", 3900, "dict"),
    ("今天", "jintian", "jt", 3800, "dict"),
    ("明天", "mingtian", "mt", 3700, "dict"),
    ("输入法", "shurufa", "srf", 3600, "dict"),
    ("自动排词", "zidongpaici", "zdpc", 3500, "dict"),
    ("模型", "moxing", "mx", 3400, "dict"),
    ("测试", "ceshi", "cs", 3300, "dict"),
    ("正确", "zhengque", "zq", 3200, "dict"),
    ("优秀", "youxiu", "yx", 3100, "dict"),
    ("框架", "kuangjia", "kj", 3000, "dict"),
    ("非常", "feichang", "fc", 2900, "dict"),
    ("开发", "kaifa", "kf", 2800, "dict"),
    ("代码", "daima", "dm", 2700, "dict"),
    ("程序", "chengxu", "cx", 2600, "dict"),
    ("系统", "xitong", "xt", 2500, "dict"),
    ("苹果", "pingguo", "pg", 2400, "dict"),
    ("香蕉", "xiangjiao", "xj", 2300, "dict"),
    ("西瓜", "xigua", "xg", 2200, "dict"),
    ("飞机", "feiji", "fj", 2100, "dict"),
    ("火车", "huoche", "hc", 2000, "dict"),
    ("汽车", "qiche", "qc", 1900, "dict"),
    ("天气", "tianqi", "tq", 1800, "dict"),
    ("下雨", "xiayu", "xy", 1700, "dict"),
    ("晴天", "qingtian", "qt", 1600, "dict"),
    ("高兴", "gaoxing", "gx", 1500, "dict"),
    ("快乐", "kuaile", "kl", 1400, "dict"),
    ("谢谢", "xiexie", "xx", 1300, "dict"),
    ("再见", "zaijian", "zj", 1200, "dict"),
    ("学习", "xuexi", "xx", 1100, "dict"),
    ("工作", "gongzuo", "gz", 1000, "dict"),
    ("生活", "shenghuo", "sh", 900, "dict"),
    ("时间", "shijian", "sj", 800, "dict"),
    ("地方", "difang", "df", 700, "dict"),
    ("朋友", "pengyou", "py", 600, "dict"),
    ("大家", "dajia", "dj", 500, "dict"),
    ("世界", "shijie", "sj", 400, "dict"),
    ("希望", "xiwang", "xw", 300, "dict"),
    ("成功", "chenggong", "cg", 200, "dict"),
    ("失败", "shibai", "sb", 100, "dict"),
]

ASSOCIATION_DICT: Dict[str, List[str]] = {
    "我": ["们", "觉得", "喜欢", "在写代码", "去过了", "知道"],
    "你": ["好", "们", "在干嘛", "去哪里", "觉得", "可以"],
    "我们": ["一起", "去", "觉得", "开始", "是", "大家"],
    "你们": ["觉得", "去哪里", "在干嘛", "大家"],
    "他们": ["觉得", "去哪里", "大家"],
    "中国": ["人", "历史", "文化", "制造", "科学"],
    "今天": ["天气", "真开心", "去吃什么", "星期几", "工作"],
    "谢谢": ["你", "大家", "您的支持"],
    "自动": ["排词", "控制", "生成"],
    "输入": ["法", "核心", "缓冲区"],
    "优秀": ["的", "框架", "作品"],
}

PINYIN_SYLLABLES = {
    "a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao", "bei", "ben",
    "beng", "bi", "bian", "biao", "bie", "bin", "bing", "bo", "bu", "ca", "cai",
    "can", "cang", "cao", "ce", "cen", "ceng", "cha", "chai", "chan", "chang",
    "chao", "che", "chen", "cheng", "chi", "chong", "chou", "chu", "chuai",
    "chuan", "chuang", "chui", "chun", "chuo", "ci", "cong", "cou", "cu",
    "cuan", "cui", "cun", "cuo", "da", "dai", "dan", "dang", "dao", "de", "dei",
    "den", "deng", "di", "dia", "dian", "diao", "die", "ding", "diu", "dong",
    "dou", "du", "duan", "dui", "dun", "duo", "e", "ei", "en", "eng", "er",
    "fa", "fan", "fang", "fei", "fen", "feng", "fo", "fou", "fu", "ga", "gai",
    "gan", "gang", "gao", "ge", "gei", "gen", "geng", "gong", "gou", "gu", "gua",
    "guai", "guan", "guang", "gui", "gun", "guo", "ha", "hai", "han", "hang",
    "hao", "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua", "huai",
    "huan", "huang", "hui", "hun", "huo", "ji", "jia", "jian", "jiang", "jiao",
    "jie", "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun", "ka", "kai",
    "kan", "kang", "kao", "ke", "ken", "keng", "kong", "kou", "ku", "kua",
    "kuai", "kuan", "kuang", "kui", "kun", "kuo", "la", "lai", "lan", "lang",
    "lao", "le", "lei", "leng", "li", "lia", "lian", "liang", "liao", "lie",
    "lin", "ling", "liu", "lo", "long", "lou", "lu", "luan", "lue", "lun", "luo",
    "lv", "ma", "mai", "man", "mang", "mao", "me", "mei", "men", "meng", "mi",
    "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu", "na", "nai",
    "nan", "nang", "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang",
    "niao", "nie", "nin", "ning", "niu", "nong", "nou", "nu", "nuan", "nue",
    "nuo", "nv", "o", "ou", "pa", "pai", "pan", "pang", "pao", "pei", "pen",
    "peng", "pi", "pian", "piao", "pie", "pin", "ping", "po", "pou", "pu", "qi",
    "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu", "qu",
    "quan", "que", "qun", "ran", "rang", "rao", "re", "ren", "reng", "ri", "rong",
    "rou", "ru", "ruan", "rui", "run", "ruo", "sa", "sai", "san", "sang", "sao",
    "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao", "she", "shei",
    "shen", "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan", "shuang",
    "shui", "shun", "shuo", "si", "song", "sou", "su", "suan", "sui", "sun",
    "suo", "ta", "tai", "tan", "tang", "tao", "te", "teng", "ti", "tian", "tiao",
    "tie", "ting", "tong", "tou", "tu", "tuan", "tui", "tun", "tuo", "wa", "wai",
    "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia", "xian", "xiang",
    "xiao", "xie", "xin", "xing", "xiong", "xiu", "xu", "xuan", "xue", "xun",
    "ya", "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong", "you",
    "yu", "yuan", "yue", "yun", "za", "zai", "zan", "zang", "zao", "ze", "zei",
    "zen", "zeng", "zha", "zhai", "zhan", "zhang", "zhao", "zhe", "zhei", "zhen",
    "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang",
    "zhui", "zhun", "zhuo", "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo",
}


class PinyinCandidateGenerator(BaseCandidateGenerator):
    """拼音候选词召回器，支持全拼、简拼、前缀和最小连续切分。"""

    def __init__(self, dict_path: str = None, max_recall: int = 100):
        self.max_recall = max_recall
        self.words: List[Tuple[str, str, str, float, str]] = list(RAW_WORDS)
        if dict_path and os.path.exists(dict_path):
            try:
                entries = LexiconLoader.load_from_jsonl(dict_path)
                self.words.extend(
                    (e.word, e.pinyin, e.short_pinyin, float(e.freq), "dict")
                    for e in entries
                )
            except Exception:
                logger.warning("外部拼音词库加载失败 (path=%s)，降级使用内置词库", dict_path, exc_info=True)

    def split_pinyin(self, text: str) -> List[str]:
        """正向最大匹配切分；末尾不完整片段作为前缀保留。"""
        text = text.lower().strip()
        result: List[str] = []
        i = 0
        while i < len(text):
            matched = ""
            max_end = min(len(text), i + 6)
            for end in range(max_end, i, -1):
                part = text[i:end]
                if part in PINYIN_SYLLABLES:
                    matched = part
                    break
            if matched:
                result.append(matched)
                i += len(matched)
            else:
                result.append(text[i:])
                break
        return result

    def _words_for_exact_pinyin(self, pinyin_key: str) -> List[Tuple[str, float]]:
        matches: List[Tuple[str, float]] = []
        for word, pinyin, _, freq, _ in self.words:
            if pinyin == pinyin_key:
                matches.append((word, float(freq)))
        return sorted(matches, key=lambda item: item[1], reverse=True)[:3]

    def _segmented_candidates(self, composing: str) -> List[Candidate]:
        segments = self.split_pinyin(composing)
        if len(segments) < 2 or any(len(seg) <= 1 for seg in segments):
            return []

        options = [self._words_for_exact_pinyin(seg) for seg in segments]
        if any(not opt for opt in options):
            return []

        candidates: List[Candidate] = []

        def build(index: int, text_parts: List[str], score_parts: List[float]) -> None:
            if index == len(options):
                phrase = "".join(text_parts)
                avg_score = sum(score_parts) / max(len(score_parts), 1)
                candidates.append(
                    Candidate(
                        text=phrase,
                        composing_covered=composing,
                        score=avg_score * 0.72,
                        source="dict_segmented_pinyin",
                    )
                )
                return
            for word, freq in options[index][:2]:
                build(index + 1, text_parts + [word], score_parts + [freq])

        build(0, [], [])
        return candidates

    # ── 键位误触纠错映射 ──
    CORRECTION_MAP = {
        "o": ["i"],   # o 键位于 i 键右侧，用户可能误触
        "u": ["i"],   # u 键位于 i 键右侧，用户可能误触
    }

    def _correction_candidates(self, composing: str) -> List[Candidate]:
        """对明显键位误触 (i/o/u) 生成纠错候选。
        纠错候选来源标记为 'correction'，便于测试和 UI 标记。"""
        composing = composing.lower().strip()
        candidates: List[Candidate] = []

        # 检查是否包含可纠错字符
        needs_correction = any(ch in self.CORRECTION_MAP for ch in composing)
        if not needs_correction or len(composing) < 2:
            return candidates

        # 生成所有纠错变体
        corrected_keys = set()
        for pos, ch in enumerate(composing):
            if ch in self.CORRECTION_MAP:
                for replacement in self.CORRECTION_MAP[ch]:
                    corrected = composing[:pos] + replacement + composing[pos + 1:]
                    corrected_keys.add(corrected)

        # 对每个纠错后的拼音查找匹配词
        for corrected_pinyin in corrected_keys:
            for word, pinyin, short, freq, _source in self.words:
                freq_f = float(freq)
                # 纠错拼音精确匹配
                if pinyin == corrected_pinyin:
                    candidates.append(Candidate(
                        word, composing, freq_f * 1.0,
                        "correction_exact_pinyin",
                    ))
                # 纠错拼音前缀匹配
                elif pinyin.startswith(corrected_pinyin):
                    ratio = len(corrected_pinyin) / len(pinyin)
                    candidates.append(Candidate(
                        word, composing, freq_f * 0.7 * ratio,
                        "correction_prefix_pinyin",
                    ))
                # 纠错简拼匹配
                elif short == corrected_pinyin:
                    candidates.append(Candidate(
                        word, composing, freq_f * 0.9,
                        "correction_exact_short",
                    ))

        return candidates

    def generate_candidates(self, context_before: str, composing: str) -> List[Candidate]:
        candidates: List[Candidate] = []
        composing = composing.lower().strip()

        if not composing:
            if context_before:
                for length in (4, 3, 2, 1):
                    if len(context_before) >= length:
                        tail = context_before[-length:]
                        if tail in ASSOCIATION_DICT:
                            for idx, word in enumerate(ASSOCIATION_DICT[tail]):
                                candidates.append(
                                    Candidate(
                                        text=word,
                                        composing_covered="",
                                        score=max(0.01, 1.0 - idx * 0.1),
                                        source="association",
                                    )
                                )
                            break
            return candidates

        for word, pinyin, short, freq, _source in self.words:
            freq = float(freq)
            if pinyin == composing:
                candidates.append(
                    Candidate(word, composing, freq * 1.5, "dict_exact_pinyin")
                )
            elif short == composing:
                candidates.append(
                    Candidate(word, composing, freq * 1.2, "dict_exact_short")
                )
            elif pinyin.startswith(composing):
                ratio = len(composing) / len(pinyin)
                candidates.append(
                    Candidate(word, composing, freq * 0.8 * ratio, "dict_prefix_pinyin")
                )
            elif short.startswith(composing) and len(short) > 1:
                ratio = len(composing) / len(short)
                candidates.append(
                    Candidate(word, composing, freq * 0.6 * ratio, "dict_prefix_short")
                )

        candidates.extend(self._segmented_candidates(composing))

        # ── 基础纠错：i/o/u 键位误触 ──
        correction_candidates = self._correction_candidates(composing)

        # 去重 + 排序（普通候选在前）
        normal = self._deduplicate_and_truncate(candidates, self.max_recall)

        if not correction_candidates:
            return normal

        correction = self._deduplicate_and_truncate(correction_candidates, 20)

        # 规则：纠错候选不得挤占第 1 位（1-indexed）精确匹配候选
        # 默认将纠错候选放在第 4-8 位之间 (0-indexed: 3-7)
        # 若无普通候选且有高置信纠错，放在第 2-4 位之间 (0-indexed: 1-3)
        has_normal = len(normal) > 0
        if has_normal:
            insert_at = min(max(3, len(normal)), 7)
        else:
            insert_at = 1  # 无普通候选时从第 2 位(0-indexed:1)开始

        # 头部：insert_at 个普通候选（不足时从尾部补齐）
        head = list(normal[:insert_at])
        tail = list(normal[insert_at:])
        while len(head) < insert_at and tail:
            head.append(tail.pop(0))

        merged = head
        for corr in correction:
            if len(merged) >= self.max_recall:
                break
            if corr.text not in {c.text for c in merged}:
                merged.append(corr)
        for c in tail:
            if len(merged) >= self.max_recall:
                break
            if c.text not in {m.text for m in merged}:
                merged.append(c)

        return merged[:self.max_recall]
