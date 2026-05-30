import os
import sys
import time
import numpy as np
from typing import List

# 将项目根目录加入到 Python 模块查找路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.input_method.config import InputMethodConfig
from src.input_method.engine import InputMethodEngine


def run_benchmark() -> None:
    # 评测用拼音样本
    pinyin_queries = [
        "nihao", "wo", "shurufa", "zhongguo", "ceshi", 
        "ta", "xiangyao", "woxiangyao", "nimen", "women",
        "py", "zg", "srf", "kaifa", "daima", "chengxu",
        "gaoxing", "kuaile", "xiexie", "xuexi", "gongzuo",
        "a", "b", "c", "d", "e", "f", "g", "h", "i"
    ]

    print("=" * 60)
    print("      GOLF 输入法底座 v0 性能延迟 Benchmark")
    print("=" * 60)

    # 1. 拼音模式性能
    config_py = InputMethodConfig(
        mode="pinyin",
        dict_path=os.path.join("data", "lexicon", "dict.jsonl")
    )
    engine_py = InputMethodEngine(config_py)

    latencies_ms: List[float] = []

    print("开始进行引擎响应耗时评测...")
    
    # 对每个拼音进行按键式 handle_char 测试
    for query in pinyin_queries:
        engine_py.clear()
        for char in query:
            start_time = time.perf_counter()
            engine_py.handle_char(char)
            elapsed = (time.perf_counter() - start_time) * 1000.0
            latencies_ms.append(elapsed)

    # 2. 联想词耗时测试 (当 composing 为空，通过 commit 触发联想词更新)
    for _ in range(50):
        engine_py.clear()
        start_time = time.perf_counter()
        engine_py.commit_text("我")
        elapsed = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(elapsed)

    # 计算分位数
    latencies = np.array(latencies_ms)
    count = len(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    max_val = np.max(latencies)
    avg_val = np.mean(latencies)

    print("-" * 60)
    print(f" 评测样本总按键数 : {count} 次")
    print(f" 平均延迟         : {avg_val:.3f} ms")
    print(f" P50 (中位数) 延迟 : {p50:.3f} ms")
    print(f" P95 延迟         : {p95:.3f} ms")
    print(f" P99 延迟         : {p99:.3f} ms")
    print(f" 最大延迟         : {max_val:.3f} ms")
    print("=" * 60)
    print(" 性能结论:")
    if p99 < 5.0:
        print(" [优秀] 核心引擎 P99 响应时间远低于 5ms，感觉不到任何输入卡顿。")
    elif p99 < 15.0:
        print(" [良好] 核心引擎 P99 响应时间低于 15ms，交互体验流畅。")
    else:
        print(" [警告] 部分输入可能存在卡顿现象，建议优化检索算法与词表大小。")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
