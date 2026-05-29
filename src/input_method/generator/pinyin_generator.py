from typing import List, Dict, Tuple
import logging
import os
import json
from .base import BaseCandidateGenerator, Candidate

logger = logging.getLogger(__name__)

# 预内置的常用词库：(词语, 全拼, 简拼, 静态词频)
RAW_WORDS: List[Tuple[str, str, str, int]] = [
    ("的", "de", "d", 9900),
    ("是", "shi", "s", 9500),
    ("了", "le", "l", 9000),
    ("在", "zai", "z", 8500),
    ("我", "wo", "w", 8000),
    ("你", "ni", "n", 7900),
    ("他", "ta", "t", 7000),
    ("她", "ta", "t", 6500),
    ("它", "ta", "t", 6000),
    ("们", "men", "m", 6800),
    ("我们", "women", "wm", 7800),
    ("你们", "nimen", "nm", 7300),
    ("他们", "tamen", "tm", 7100),
    ("有", "you", "y", 6800),
    ("个", "ge", "g", 6600),
    ("好", "hao", "h", 6500),
    ("你好", "nihao", "nh", 6700),
    ("你好啊", "nihaoa", "nha", 4500),
    ("这", "zhe", "z", 6400),
    ("要", "yao", "y", 6300),
    ("国", "guo", "g", 6200),
    ("中国", "zhongguo", "zg", 6500),
    ("人", "ren", "r", 6100),
    ("和", "he", "h", 6000),
    ("用", "yong", "y", 5900),
    ("作", "zuo", "z", 5800),
    ("时", "shi", "s", 5700),
    ("去", "qu", "q", 5600),
    ("来", "lai", "l", 5500),
    ("会", "hui", "h", 5400),
    ("能", "neng", "n", 5300),
    ("对", "dui", "d", 5200),
    ("都", "dou", "d", 5100),
    ("多", "duo", "d", 5000),
    ("少", "shao", "s", 4500),
    ("多少", "duoshao", "ds", 4700),
    ("没", "mei", "m", 4400),
    ("没有", "meiyou", "my", 4800),
    ("怎么", "zenme", "zm", 4300),
    ("什么", "shenme", "sm", 4600),
    ("为什么", "weishenme", "wsm", 4200),
    ("觉得", "juede", "jd", 4100),
    ("知道", "zhidao", "zd", 4050),
    ("可以", "keyi", "ky", 4000),
    ("现在", "xianzai", "xz", 3900),
    ("今天", "jintian", "jt", 3800),
    ("明天", "mingtian", "mt", 3700),
    ("输入法", "shurufa", "srf", 3600),
    ("自动排词", "zidongpaici", "zdpc", 3500),
    ("模型", "moxing", "mx", 3400),
    ("测试", "ceshi", "cs", 3300),
    ("正确", "zhengque", "zq", 3200),
    ("优秀", "youxiu", "yx", 3100),
    ("框架", "kuangjia", "kj", 3000),
    ("非常", "feichang", "fc", 2900),
    ("开发", "kaifa", "kf", 2800),
    ("代码", "daima", "dm", 2700),
    ("程序", "chengxu", "cx", 2600),
    ("系统", "xitong", "xt", 2500),
    ("苹果", "pingguo", "pg", 2400),
    ("香蕉", "xiangjiao", "xj", 2300),
    ("西瓜", "xigua", "xg", 2200),
    ("飞机", "feiji", "fj", 2100),
    ("火车", "huoche", "hc", 2000),
    ("汽车", "qiche", "qc", 1900),
    ("天气", "tianqi", "tq", 1800),
    ("下雨", "xiayu", "xy", 1700),
    ("晴天", "qingtian", "qt", 1600),
    ("高兴", "gaoxing", "gx", 1500),
    ("快乐", "kuaile", "kl", 1400),
    ("谢谢", "xiexie", "xx", 1300),
    ("再见", "zaijian", "zj", 1200),
    ("学习", "xuexi", "xx", 1100),
    ("工作", "gongzuo", "gz", 1000),
    ("生活", "shenghuo", "sh", 900),
    ("时间", "shijian", "sj", 800),
    ("地方", "difang", "df", 700),
    ("朋友", "pengyou", "py", 600),
    ("大家", "dajia", "dj", 500),
    ("世界", "shijie", "sj", 400),
    ("希望", "xiwang", "xw", 300),
    ("成功", "chenggong", "cg", 200),
    ("失败", "shibai", "sb", 100),
]

# 上下文联想词映射 (当 composing 为空时，根据前一个字/词联想后续候选)
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


class PinyinCandidateGenerator(BaseCandidateGenerator):
    """拼音候选词召回器，支持全拼、首字母简拼和上下文联想"""

    def __init__(self, dict_path: str = None, max_recall: int = 100):
        self.max_recall = max_recall
        self.words = list(RAW_WORDS)
        
        # 如果提供了外部词库路径，尝试加载
        if dict_path and os.path.exists(dict_path):
            try:
                with open(dict_path, "r", encoding="utf-8") as f:
                    # 假定格式为 JSON 列表: [ [word, pinyin, short_pinyin, freq], ... ]
                    external_words = json.load(f)
                    self.words.extend([tuple(w) for w in external_words])
            except Exception:
                logger.warning("外部拼音词库加载失败 (path=%s)，降级使用内置词库", dict_path, exc_info=True)

    def generate_candidates(self, context_before: str, composing: str) -> List[Candidate]:
        candidates: List[Candidate] = []
        composing = composing.lower().strip()

        # 场景 A: Composing 缓冲区为空 -> 触发上下文联想召回
        if not composing:
            if context_before:
                # 尝试匹配 context_before 的最右端字/词
                for length in (4, 3, 2, 1):
                    if len(context_before) >= length:
                        tail = context_before[-length:]
                        if tail in ASSOCIATION_DICT:
                            assoc_words = ASSOCIATION_DICT[tail]
                            for idx, word in enumerate(assoc_words):
                                # 给联想词赋予稍高的分数，按其在联想列表里的顺序递减
                                score = max(0.01, 1.0 - idx * 0.1)
                                candidates.append(Candidate(
                                    text=word,
                                    composing_covered="",
                                    score=score,
                                    source="association"
                                ))
                            break
            return candidates

        # 场景 B: Composing 不为空 -> 匹配拼音
        for word, pinyin, short, freq in self.words:
            # 1. 精确全拼匹配
            if pinyin == composing:
                score = freq * 1.5
                candidates.append(Candidate(
                    text=word,
                    composing_covered=composing,
                    score=score,
                    source="dict_exact_pinyin"
                ))
            # 2. 精确简拼匹配
            elif short == composing:
                score = freq * 1.2
                candidates.append(Candidate(
                    text=word,
                    composing_covered=composing,
                    score=score,
                    source="dict_exact_short"
                ))
            # 3. 前缀全拼匹配 (例如输入 "nih" 匹配 "nihao" -> "你好")
            elif pinyin.startswith(composing):
                # 匹配比例作为衰减系数
                ratio = len(composing) / len(pinyin)
                score = freq * 0.8 * ratio
                candidates.append(Candidate(
                    text=word,
                    composing_covered=composing,
                    score=score,
                    source="dict_prefix_pinyin"
                ))
            # 4. 前缀简拼匹配 (例如输入 "z" 匹配 "zg" -> "中国")
            elif short.startswith(composing) and len(short) > 1:
                ratio = len(composing) / len(short)
                score = freq * 0.6 * ratio
                candidates.append(Candidate(
                    text=word,
                    composing_covered=composing,
                    score=score,
                    source="dict_prefix_short"
                ))

        # 去重并截断（排序由 Reranker 统一负责）
        return self._deduplicate_and_truncate(candidates, self.max_recall)
