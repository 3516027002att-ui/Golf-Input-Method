import os
import sys
import time
from typing import List, Tuple

# 将项目根目录加入到 Python 模块查找路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.input_method.config import InputMethodConfig
from src.input_method.engine import InputMethodEngine


def run_evaluation() -> None:
    # 评测用例: (模式, 输入, 期望词)
    evaluation_cases: List[Tuple[str, str, str]] = [
        ("pinyin", "nihao", "你好"),
        ("pinyin", "wo", "我"),
        ("pinyin", "shurufa", "输入法"),
        ("pinyin", "zhongguo", "中国"),
        ("pinyin", "ceshi", "测试"),
        ("pinyin", "ta", "他"),
        ("pinyin", "xiangyao", "想要"),
        ("pinyin", "woxiangyao", "我想要"),
        ("pinyin", "nimen", "你们"),
        ("pinyin", "women", "我们"),
        ("english", "th", "the"),
        ("english", "an", "and"),
        ("english", "parame", "parameter"),
        ("english", "golf", "golf"),
        ("japanese", "nihon", "日本"),
        ("japanese", "ariga", "ありがとう"),
    ]

    print("=" * 60)
    print("      GOLF 输入法底座 v0 候选质量评估")
    print("=" * 60)

    # 按模式分组评估
    results_by_mode = {}

    for mode in ("pinyin", "english", "japanese"):
        # 为该模式创建一个引擎
        config = InputMethodConfig(
            mode=mode,
            dict_path=os.path.join("data", "lexicon", "dict.jsonl")
        )
        try:
            engine = InputMethodEngine(config)
        except Exception as e:
            print(f"警告: 初始化引擎失败 (mode={mode}): {e}")
            continue

        cases = [c for c in evaluation_cases if c[0] == mode]
        if not cases:
            continue

        total = len(cases)
        top1_hits = 0
        top3_hits = 0
        coverage_hits = 0

        print(f"\n开始评估 [{mode.upper()}] 模式 (样本数: {total})...")
        print("-" * 50)
        print(f"{'输入':<12} | {'期望词':<8} | {'首选 (Top1)':<8} | {'Top3 候选'}")
        print("-" * 50)

        for _, inp, expected in cases:
            # 输入字符
            engine.clear()
            for char in inp:
                engine.handle_char(char)

            # 获取所有候选
            cands = engine.candidates
            cand_texts = [c.text for c in cands]

            # 校验指标
            top1 = cand_texts[0] if cand_texts else "(空)"
            top3 = cand_texts[:3]

            is_top1 = (top1 == expected)
            is_top3 = (expected in top3)
            in_candidates = (expected in cand_texts)

            if is_top1:
                top1_hits += 1
            if is_top3:
                top3_hits += 1
            if in_candidates:
                coverage_hits += 1

            top3_str = ", ".join(top3)
            print(f"{inp:<12} | {expected:<8} | {top1:<8} | [{top3_str}]")

        results_by_mode[mode] = {
            "total": total,
            "top1_acc": top1_hits / total,
            "top3_acc": top3_hits / total,
            "coverage": coverage_hits / total
        }

    # 打印汇总表格
    print("\n" + "=" * 60)
    print("                      评估指标汇总")
    print("=" * 60)
    print(f"{'模式':<10} | {'总数':<5} | {'Top-1 准确率':<12} | {'Top-3 准确率':<12} | {'召回覆盖率':<10}")
    print("-" * 60)
    for mode, metrics in results_by_mode.items():
        print(
            f"{mode:<10} | {metrics['total']:<5} | "
            f"{metrics['top1_acc']*100:>.1f}%       | "
            f"{metrics['top3_acc']*100:>.1f}%       | "
            f"{metrics['coverage']*100:>.1f}%"
        )
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
