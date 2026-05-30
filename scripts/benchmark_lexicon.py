# -*- coding: utf-8 -*-
"""词库性能压测脚本

测量 PinyinCandidateGenerator 的加载时间、去重词条数、
查询延迟分位数 (P50/P95/P99) 和内存占用。
"""

import argparse
import os
import random
import sys
import time
from typing import List

# 将项目根目录加入到 Python 模块查找路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.input_method.generator.pinyin_generator import PinyinCandidateGenerator


# 用于随机查询测试的拼音样本
QUERY_SAMPLES = [
    "nihao", "wo", "shurufa", "zhongguo", "ceshi",
    "ta", "xiangyao", "woxiangyao", "nimen", "women",
    "py", "zg", "srf", "kaifa", "daima", "chengxu",
    "gaoxing", "kuaile", "xiexie", "xuexi", "gongzuo",
    "jintian", "mingtian", "keyi", "xianzai", "zhidao",
    "a", "b", "c", "d", "e", "f", "g", "h", "i",
    "shi", "de", "le", "zai", "you", "ge", "hao",
    "zhe", "guo", "ren", "he", "yong", "qu", "lai",
    "hui", "neng", "dui", "dou", "duo", "mei",
]


def calculate_percentile(data: List[float], q: float) -> float:
    """计算百分位数 (线性插值法)"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (n - 1) * q / 100.0
    idx_floor = int(idx)
    idx_ceil = min(n - 1, idx_floor + 1)
    if idx_floor == idx_ceil:
        return sorted_data[idx_floor]
    weight = idx - idx_floor
    return sorted_data[idx_floor] * (1.0 - weight) + sorted_data[idx_ceil] * weight


def estimate_memory(obj: object) -> int:
    """估算对象的内存占用 (字节)"""
    size = sys.getsizeof(obj)
    if hasattr(obj, "words"):
        size += sys.getsizeof(obj.words)
        for item in obj.words:
            size += sys.getsizeof(item)
            if isinstance(item, (list, tuple)):
                for sub in item:
                    size += sys.getsizeof(sub)
    return size


def run_benchmark(dict_path: str) -> None:
    print("=" * 60)
    print("      GOLF 输入法底座 v0 词库性能 Benchmark")
    print("=" * 60)

    # 1. 加载时间测量
    print("\n[1/4] 加载词库...")
    start_load = time.perf_counter()
    generator = PinyinCandidateGenerator(dict_path=dict_path)
    load_time_ms = (time.perf_counter() - start_load) * 1000.0
    print(f"  词库加载耗时: {load_time_ms:.3f} ms")

    # 2. 去重后词条数
    print("\n[2/4] 统计词条...")
    total_words = len(generator.words)
    unique_words = len(set(w[0] for w in generator.words))
    unique_pinyin = len(set(w[1] for w in generator.words))
    print(f"  总词条数: {total_words}")
    print(f"  去重后词条数 (按词语): {unique_words}")
    print(f"  去重后拼音数: {unique_pinyin}")

    # 3. 查询延迟测试 (1000 次随机查询)
    print("\n[3/4] 查询延迟测试 (1000 次随机查询)...")
    random.seed(42)
    query_count = 1000
    latencies_ms: List[float] = []

    for _ in range(query_count):
        query = random.choice(QUERY_SAMPLES)
        start_time = time.perf_counter()
        generator.generate_candidates(context_before="", composing=query)
        elapsed = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(elapsed)

    p50 = calculate_percentile(latencies_ms, 50.0)
    p95 = calculate_percentile(latencies_ms, 95.0)
    p99 = calculate_percentile(latencies_ms, 99.0)
    max_val = max(latencies_ms) if latencies_ms else 0.0
    avg_val = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    # 4. 内存估算
    print("\n[4/4] 内存估算...")
    mem_bytes = estimate_memory(generator)
    mem_kb = mem_bytes / 1024.0
    mem_mb = mem_kb / 1024.0

    # 输出汇总 (与 benchmark_latency.py 风格一致)
    print("\n" + "-" * 60)
    print(f" 词库路径          : {dict_path}")
    print(f" 词库加载耗时       : {load_time_ms:.3f} ms")
    print(f" 总词条数           : {total_words}")
    print(f" 去重后词条数       : {unique_words}")
    print(f" 查询次数           : {query_count} 次")
    print(f" 平均查询延迟       : {avg_val:.3f} ms")
    print(f" P50 (中位数) 延迟  : {p50:.3f} ms")
    print(f" P95 延迟           : {p95:.3f} ms")
    print(f" P99 延迟           : {p99:.3f} ms")
    print(f" 最大延迟           : {max_val:.3f} ms")
    print(f" 内存估算           : {mem_kb:.1f} KB ({mem_mb:.2f} MB)")
    print("=" * 60)
    print(" 性能结论:")
    if p99 < 5.0:
        print(" [优秀] 词库查询 P99 响应时间远低于 5ms，适合实时输入法使用。")
    elif p99 < 15.0:
        print(" [良好] 词库查询 P99 响应时间低于 15ms，体验流畅。")
    else:
        print(" [警告] 部分查询可能存在延迟，建议优化词表索引结构。")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GOLF 词库性能压测 — 加载时间/去重/查询延迟/内存"
    )
    parser.add_argument(
        "--dict-path", type=str,
        default=os.path.join("data", "lexicon", "dict.jsonl"),
        help="词库文件路径 (默认: data/lexicon/dict.jsonl)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dict_path):
        print(f"警告: 词库文件不存在 ({args.dict_path})，将使用内置 fallback 词库进行测试")

    run_benchmark(args.dict_path)


if __name__ == "__main__":
    main()
