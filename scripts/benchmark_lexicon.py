#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""词库加载和查询性能压测 (标准库实现，不强依赖 numpy)。

测试项：
- 词库文件加载 (JSONL 解析 + 内存占用估算)
- 召回查询延迟 P50/P95/P99 (全表扫描模拟)
- 10 万级词库性能判定
"""

import argparse
import json
import os
import sys
import time
import gc


def load_dict(dict_path):
    """加载 JSONL 词库。"""
    entries = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def benchmark_load(dict_path):
    """测试词库加载性能。"""
    start = time.time()
    entries = load_dict(dict_path)
    load_time = time.time() - start

    try:
        total_size = sum(sys.getsizeof(e) + sum(sys.getsizeof(str(v)) for v in e.values()) for e in entries)
        est_mb = total_size / (1024 * 1024)
    except Exception:
        est_mb = os.path.getsize(dict_path) / (1024 * 1024) * 3

    return {
        "count": len(entries),
        "load_time_s": load_time,
        "est_memory_mb": est_mb,
        "file_size_mb": os.path.getsize(dict_path) / (1024 * 1024),
    }


def benchmark_query(entries, num_queries=500):
    """测试查询延迟 (全表扫描)。"""
    import random
    rng = random.Random(42)
    sample_prefixes = []
    for _ in range(num_queries):
        entry = rng.choice(entries)
        pinyin = entry.get("pinyin", "")
        if pinyin and len(pinyin) >= 2:
            prefix_len = rng.randint(2, len(pinyin))
            sample_prefixes.append(pinyin[:prefix_len])
        else:
            sample_prefixes.append(pinyin)

    latencies = []
    for prefix in sample_prefixes:
        start = time.time()
        results = []
        for entry in entries:
            py = entry.get("pinyin", "")
            sp = entry.get("short_pinyin", entry.get("short", ""))
            if py.startswith(prefix) or sp.startswith(prefix):
                results.append(entry)
        results.sort(key=lambda e: e.get("freq", 0), reverse=True)
        results = results[:50]
        latencies.append((time.time() - start) * 1000)

    latencies.sort()
    n = len(latencies)
    return {
        "num_queries": num_queries,
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
        "max_ms": latencies[-1],
        "min_ms": latencies[0],
        "mean_ms": sum(latencies) / n,
    }


def main():
    parser = argparse.ArgumentParser(description="词库加载和查询性能压测")
    parser.add_argument("--dict-path", "-d", type=str, required=True, help="词库 JSONL 文件路径")
    parser.add_argument("--queries", "-q", type=int, default=500, help="查询样本数")
    args = parser.parse_args()

    if not os.path.exists(args.dict_path):
        print(f"错误: 文件不存在: {args.dict_path}")
        return 1

    print(f"词库性能压测: {args.dict_path}")
    print(f"  查询样本数: {args.queries}")
    print()

    # 1. 加载性能
    print("=== 加载性能 ===")
    load_result = benchmark_load(args.dict_path)
    print(f"  词条总数: {load_result['count']:,}")
    print(f"  文件大小: {load_result['file_size_mb']:.1f} MB")
    print(f"  加载耗时: {load_result['load_time_s']:.3f}s")
    print(f"  估算内存: {load_result['est_memory_mb']:.1f} MB")

    if load_result["count"] >= 100000:
        print(f"  ✅ 10 万级词库加载达标")
    elif load_result["count"] >= 50000:
        print(f"  ⚠️ 词库规模 < 10 万 (当前 {load_result['count']:,} 条)")
    else:
        print(f"  ⚠️ 词库规模较小 (当前 {load_result['count']:,} 条)")

    # 2. 查询延迟
    print()
    print("=== 查询延迟 ===")
    entries = load_dict(args.dict_path)
    query_result = benchmark_query(entries, num_queries=args.queries)
    print(f"  查询次数: {query_result['num_queries']}")
    print(f"  P50 延迟: {query_result['p50_ms']:.2f} ms")
    print(f"  P95 延迟: {query_result['p95_ms']:.2f} ms")
    print(f"  P99 延迟: {query_result['p99_ms']:.2f} ms")
    print(f"  平均延迟: {query_result['mean_ms']:.2f} ms")
    print(f"  最大延迟: {query_result['max_ms']:.2f} ms")
    print(f"  最小延迟: {query_result['min_ms']:.2f} ms")

    if query_result["p95_ms"] < 100:
        print(f"  ✅ P95 延迟 < 100ms，性能良好")
    elif query_result["p95_ms"] < 500:
        print(f"  ⚠️ P95 延迟 {query_result['p95_ms']:.1f}ms，可接受但需优化")
    else:
        print(f"  ❌ P95 延迟 {query_result['p95_ms']:.1f}ms，需要索引优化")

    # 3. 内存
    print()
    print("=== 内存估算 ===")
    gc.collect()
    entry_size = sum(sys.getsizeof(e) + sum(sys.getsizeof(str(v)) for v in e.values()) for e in entries[:1000]) / 1000
    total_est_mb = (entry_size * len(entries)) / (1024 * 1024)
    print(f"  词条数: {len(entries):,}")
    print(f"  估算总内存: {total_est_mb:.1f} MB")
    if total_est_mb < 500:
        print(f"  ✅ 内存占用合理 (< 500MB)")
    else:
        print(f"  ⚠️ 内存占用较高")

    print()
    print("压测完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
