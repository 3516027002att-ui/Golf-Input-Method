#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成合成拼音词库用于工业级压测 (10 万级)。

输出 JSONL 格式，每行一条记录，兼容 golf 标准词库格式：
{"word": "中国人", "pinyin": "zhongguoren", "short_pinyin": "zgr", "freq": 8523, "source": "synthetic"}
"""

import argparse
import json
import os
import random
import sys
import time


# 拼音音节表（覆盖常用拼音）
PINYIN_SYLLABLES = [
    "a", "ai", "an", "ang", "ao",
    "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng", "bi", "bian", "biao", "bie", "bin", "bing", "bo", "bu",
    "ca", "cai", "can", "cang", "cao", "ce", "cen", "ceng", "cha", "chai", "chan", "chang", "chao", "che", "chen", "cheng", "chi", "chong", "chou", "chu", "chua", "chuai", "chuan", "chuang", "chui", "chun", "chuo", "ci", "cong", "cou", "cu", "cuan", "cui", "cun", "cuo",
    "da", "dai", "dan", "dang", "dao", "de", "dei", "den", "deng", "di", "dian", "diao", "die", "ding", "diu", "dong", "dou", "du", "duan", "dui", "dun", "duo",
    "e", "ei", "en", "eng", "er",
    "fa", "fan", "fang", "fei", "fen", "feng", "fo", "fou", "fu",
    "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng", "gong", "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo",
    "ha", "hai", "han", "hang", "hao", "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun", "huo",
    "ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun",
    "ka", "kai", "kan", "kang", "kao", "ke", "kei", "ken", "keng", "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui", "kun", "kuo",
    "la", "lai", "lan", "lang", "lao", "le", "lei", "leng", "li", "lia", "lian", "liang", "liao", "lie", "lin", "ling", "liu", "long", "lou", "lu", "luan", "lun", "luo", "lv", "lve",
    "ma", "mai", "man", "mang", "mao", "me", "mei", "men", "meng", "mi", "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu",
    "na", "nai", "nan", "nang", "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang", "niao", "nie", "nin", "ning", "niu", "nong", "nou", "nu", "nuan", "nuo", "nv", "nve",
    "o", "ou",
    "pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng", "pi", "pian", "piao", "pie", "pin", "ping", "po", "pou", "pu",
    "qi", "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu", "qu", "quan", "que", "qun",
    "ran", "rang", "rao", "re", "ren", "reng", "ri", "rong", "rou", "ru", "rua", "ruan", "rui", "run", "ruo",
    "sa", "sai", "san", "sang", "sao", "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao", "she", "shei", "shen", "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan", "shuang", "shui", "shun", "shuo", "si", "song", "sou", "su", "suan", "sui", "sun", "suo",
    "ta", "tai", "tan", "tang", "tao", "te", "tei", "teng", "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui", "tun", "tuo",
    "wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu",
    "xi", "xia", "xian", "xiang", "xiao", "xie", "xin", "xing", "xiong", "xiu", "xu", "xuan", "xue", "xun",
    "ya", "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong", "you", "yu", "yuan", "yue", "yun",
    "za", "zai", "zan", "zang", "zao", "ze", "zei", "zen", "zeng", "zha", "zhai", "zhan", "zhang", "zhao", "zhe", "zhei", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang", "zhui", "zhun", "zhuo", "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo",
]

# 常用汉字池
COMMON_HANZI = list("的一是不了在有人我他这来之个们到说子大时中国要出会为上也以能对生学地得过就自和可下用天去那看还面发里没然小多都经把如要部工心好日方成体己事家行法民高力现实全明当已前点文正道开定其外重长关样义此意新比度理些主间么相去向同手产机头物活从进语气业部问水数法将外合并者使最本公已各第名及天化理")

# 双字常用搭配
TWO_CHAR_TEMPLATES = [
    "科技", "经济", "社会", "文化", "教育", "政治", "历史", "国家",
    "研究", "发展", "管理", "服务", "建设", "生产", "市场", "企业",
    "信息", "网络", "数据", "安全", "环境", "资源", "技术", "系统",
    "工程", "设计", "方案", "产品", "项目", "标准", "规范", "制度",
    "理论", "实践", "经验", "能力", "水平", "质量", "效率", "效果",
    "问题", "方法", "条件", "因素", "关系", "作用", "影响", "结果",
    "过程", "结构", "功能", "性能", "特征", "特点", "形式", "内容",
]

THREE_CHAR_TEMPLATES = [
    "计算机", "互联网", "数据库", "人工智能", "新能源", "半导体",
    "现代化", "工业化", "信息化", "自动化", "数字化", "智能化",
    "共和国", "博物馆", "图书馆", "体育馆", "科学院", "研究院",
    "企业家", "科学家", "教育家", "艺术家", "政治家",
    "奥运会", "世界杯", "锦标赛", "运动会", "文化节", "电影节",
]


def generate_syllable_combos(n_syllables, count, seed):
    """生成指定音节数的随机拼音组合，允许多词同音。"""
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        syllables = [rng.choice(PINYIN_SYLLABLES) for _ in range(n_syllables)]
        pinyin = "".join(syllables)
        short = "".join(s[0] for s in syllables)
        result.append((pinyin, short))
    return result


def generate_word_text(n_chars, rng):
    """根据字符数生成模拟中文词。"""
    if n_chars == 1:
        return rng.choice(COMMON_HANZI)
    elif n_chars == 2:
        if rng.random() < 0.4:
            return rng.choice(TWO_CHAR_TEMPLATES)
        return "".join(rng.choice(COMMON_HANZI) for _ in range(2))
    elif n_chars == 3:
        if rng.random() < 0.3:
            return rng.choice(THREE_CHAR_TEMPLATES)
        return "".join(rng.choice(COMMON_HANZI) for _ in range(3))
    else:
        return "".join(rng.choice(COMMON_HANZI) for _ in range(n_chars))


def main():
    parser = argparse.ArgumentParser(description="生成合成拼音词库用于工业级压测")
    parser.add_argument("--count", "-c", type=int, default=100000, help="生成词条总数")
    parser.add_argument("--output", "-o", type=str, default="lexicon_synthetic.jsonl", help="输出文件路径")
    parser.add_argument("--seed", "-s", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    total = args.count

    single = int(total * 0.08)
    two = int(total * 0.30)
    three = int(total * 0.30)
    four_plus = total - single - two - three

    print(f"生成合成词库: 总计 {total} 条")
    print(f"  单字: {single}, 双字: {two}, 三字: {three}, 四字及以上: {four_plus}")
    print(f"  随机种子: {args.seed}")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    start_time = time.time()
    written = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for label, n_syllables, word_len, cnt in [
            ("single", 1, 1, single),
            ("two", 2, 2, two),
            ("three", 3, 3, three),
        ]:
            combos = generate_syllable_combos(n_syllables, cnt, seed=args.seed + written)
            for pinyin, short in combos:
                word = generate_word_text(word_len, rng)
                freq = rng.randint(1, 10000)
                entry = {"word": word, "pinyin": pinyin, "short_pinyin": short, "freq": float(freq), "source": "synthetic"}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

        # 四字及以上
        for _ in range(four_plus):
            n_chars = rng.choices([4, 5, 6, 7, 8], weights=[50, 25, 15, 7, 3])[0]
            syllables = [rng.choice(PINYIN_SYLLABLES) for _ in range(n_chars)]
            pinyin = "".join(syllables)
            short = "".join(s[0] for s in syllables)
            word = generate_word_text(n_chars, rng)
            freq = rng.randint(1, 4000)
            entry = {"word": word, "pinyin": pinyin, "short_pinyin": short, "freq": float(freq), "source": "synthetic"}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"完成: 写入 {written} 条记录, 文件大小 {file_size_mb:.1f} MB, 耗时 {elapsed:.2f}s")
    print(f"输出: {args.output}")

    # 快速校验
    with open(args.output, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    if line_count != total:
        print(f"警告: 行数不匹配: 期望 {total}, 实际 {line_count}")
    else:
        print(f"校验通过: 行数 = {line_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
