# -*- coding: utf-8 -*-
"""合成词库生成器

生成 JSONL 格式的合成词库数据，用于开发和压力测试。
词频使用 Zipf 分布: freq = 10000 / (rank ** 0.8)。
"""

import argparse
import json
import os
import random
import sys

# 将项目根目录加入到 Python 模块查找路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 约 100 个常用拼音音节
PINYIN_SYLLABLES = [
    "a", "ai", "an", "ang", "ao",
    "ba", "bai", "ban", "bang", "bao", "bei", "ben", "bi", "bian", "bie", "bin", "bing", "bo", "bu",
    "ca", "cai", "can", "cao", "ce", "ceng", "cha", "chai", "chan", "chang", "chao", "che", "chen",
    "cheng", "chi", "chong", "chu", "ci", "cong", "cu", "cuo",
    "da", "dai", "dan", "dang", "dao", "de", "deng", "di", "dian", "ding", "dong", "dou", "du", "dui", "duo",
    "e", "en", "er",
    "fa", "fan", "fang", "fei", "fen", "feng", "fu",
    "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen", "gong", "gou", "gu", "gua", "guan", "guang", "gui", "guo",
    "ha", "hai", "han", "hang", "hao", "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun", "huo",
    "ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing", "jiu", "ju", "juan", "jue", "jun",
]

# 约 500 个常用汉字 (Unicode 0x4E00-0x9FFF)
COMMON_HANZI = [
    "的", "一", "是", "了", "我", "不", "人", "在", "他", "有",
    "这", "个", "上", "们", "来", "到", "时", "大", "地", "为",
    "子", "中", "你", "说", "生", "国", "年", "着", "就", "那",
    "和", "要", "她", "出", "也", "得", "里", "后", "自", "以",
    "会", "家", "可", "下", "而", "过", "天", "去", "能", "对",
    "小", "多", "然", "于", "心", "学", "么", "之", "都", "好",
    "看", "起", "发", "当", "没", "成", "只", "如", "事", "把",
    "还", "用", "第", "样", "道", "想", "作", "种", "开", "美",
    "总", "从", "无", "情", "已", "面", "最", "女", "但", "现",
    "前", "些", "所", "同", "日", "手", "又", "行", "意", "动",
    "方", "期", "它", "头", "经", "长", "儿", "回", "位", "分",
    "爱", "老", "因", "很", "给", "名", "法", "间", "斯", "知",
    "世", "什", "两", "次", "使", "身", "者", "被", "高", "已",
    "亲", "其", "进", "此", "话", "常", "与", "活", "正", "感",
    "见", "明", "问", "力", "理", "尔", "点", "文", "几", "定",
    "本", "公", "特", "做", "外", "孩", "相", "西", "果", "走",
    "将", "月", "十", "实", "向", "声", "车", "全", "信", "重",
    "三", "机", "工", "物", "气", "每", "并", "别", "真", "打",
    "太", "新", "比", "才", "便", "夫", "再", "书", "部", "水",
    "像", "眼", "少", "叫", "死", "呢", "电", "让", "系", "光",
    "华", "报", "条", "命", "加", "员", "吗", "找", "义", "反",
    "入", "万", "四", "民", "主", "治", "思", "运", "北", "战",
    "先", "海", "关", "白", "告", "更", "数", "传", "社", "教",
    "应", "求", "表", "五", "山", "王", "城", "觉", "内", "千",
    "接", "色", "路", "口", "带", "金", "阳", "场", "写", "黑",
    "今", "听", "花", "门", "立", "马", "百", "单", "军", "候",
    "变", "原", "深", "记", "满", "步", "快", "南", "持", "收",
    "早", "通", "九", "合", "故", "往", "吃", "客", "乐", "东",
    "风", "平", "直", "呀", "吧", "近", "坐", "落", "算", "张",
    "台", "红", "血", "认", "放", "决", "服", "许", "跟", "热",
    "送", "非", "答", "河", "六", "强", "造", "拿", "木", "笑",
    "根", "共", "连", "石", "八", "极", "影", "局", "夜", "脑",
    "足", "离", "费", "药", "型", "备", "星", "管", "研", "展",
    "何", "志", "音", "安", "错", "育", "观", "久", "居", "且",
    "青", "量", "设", "空", "考", "语", "选", "般", "仅", "楼",
    "黄", "究", "双", "节", "七", "越", "精", "令", "春", "秋",
    "冬", "夏", "助", "呼", "食", "医", "健", "康", "险", "银",
    "钱", "房", "产", "股", "市", "投", "商", "业", "图", "码",
    "技", "术", "科", "互", "联", "网", "脸", "微", "群", "消",
    "习", "校", "师", "读", "试", "验", "格", "题", "标", "划",
    "课", "段", "章", "词", "句", "式", "能", "件", "软", "硬",
    "盘", "器", "程", "模", "算", "源", "视", "频", "聊", "招",
    "歌", "舞", "画", "影", "飞", "船", "楼", "梦", "睡", "觉",
    "跑", "跳", "游", "泳", "球", "赛", "赢", "输", "场", "队",
    "员", "票", "价", "折", "店", "买", "卖", "货", "物", "品",
    "牌", "服", "装", "鞋", "帽", "衣", "裤", "镜", "表", "包",
    "饭", "菜", "肉", "鱼", "蛋", "奶", "茶", "酒", "米", "面",
    "糖", "盐", "油", "猫", "狗", "鸟", "虎", "龙", "蛇", "兔",
    "牛", "羊", "鸡", "鸭", "树", "草", "雨", "雪", "云", "冰",
    "火", "土", "林", "湖", "江", "河", "海", "洋", "岛", "桥",
]

DOMAINS = ["general", "tech", "medical", "education", "finance"]


def generate_word(hanzi_pool: list, syllable_pool: list, char_count: int) -> dict:
    """生成一个合成词条。"""
    word = "".join(random.choices(hanzi_pool, k=char_count))
    syllables = [random.choice(syllable_pool) for _ in range(char_count)]
    pinyin = "".join(syllables)
    short_pinyin = "".join(s[0] for s in syllables)
    return {
        "word": word,
        "pinyin": pinyin,
        "short_pinyin": short_pinyin,
        "syllables": syllables,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合成词库生成器 — 生成 JSONL 格式的合成词库用于开发和压测"
    )
    parser.add_argument(
        "--count", type=int, default=10000,
        help="生成词条数量 (默认: 10000)"
    )
    parser.add_argument(
        "--output", type=str, default=os.path.join(".smoke_data", "synthetic_lexicon.jsonl"),
        help="输出文件路径 (默认: .smoke_data/synthetic_lexicon.jsonl)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 词长分布: 单字 40%, 双字 35%, 三字 15%, 四字 10%
    length_weights = [
        (1, 0.40),
        (2, 0.35),
        (3, 0.15),
        (4, 0.10),
    ]
    lengths = [l for l, _ in length_weights]
    weights = [w for _, w in length_weights]

    print(f"开始生成合成词库: 目标 {args.count} 条")
    print(f"输出路径: {args.output}")
    print(f"随机种子: {args.seed}")
    print("-" * 50)

    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for rank in range(1, args.count + 1):
            char_count = random.choices(lengths, weights=weights, k=1)[0]
            entry = generate_word(COMMON_HANZI, PINYIN_SYLLABLES, char_count)

            # Zipf 分布词频: freq = 10000 / (rank ** 0.8)
            freq = 10000.0 / (rank ** 0.8)

            record = {
                "word": entry["word"],
                "pinyin": entry["pinyin"],
                "short_pinyin": entry["short_pinyin"],
                "freq": round(freq, 2),
                "source": "synthetic",
                "domain": random.choice(DOMAINS),
                "enabled": True,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            # 每 1 万条输出一次进度
            if written % 10000 == 0:
                print(f"  已生成: {written}/{args.count} 条")

    print("-" * 50)
    print(f"生成完成: 共 {written} 条, 写入 {args.output}")


if __name__ == "__main__":
    main()
