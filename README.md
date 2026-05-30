# golf — AI 输入法项目

> **当前阶段定位：** 不依赖大语言模型 (LLM) 也能使用的传统输入法底座，为后续 AI 排词提供完整基础设施。

## 项目概述

golf 目标是构建一个 **大语言模型参与选词/排词的 AI 输入法**。当前版本实现了不依赖 LLM 的完整输入法底座，包括：

- ✅ 拼音/英文/日语（原型）三模式输入
- ✅ 完整的输入缓冲区管理与候选栏交互
- ✅ 基于词频的 baseline 排序（可解释打分）
- ✅ 用户选词学习与权重记忆（持久化到本地 JSON）
- ✅ 外部词库导入/合成词库生成/词库压测工具链
- ✅ GUI 图形客户端 + 终端模拟器
- ✅ 候选质量评估 + 性能延迟 Benchmark
- ✅ Windows 系统级输入法接入工程路线文档

### ⚠️ 明确声明

| 项目 | 状态 |
|------|------|
| 当前是否为 Windows 系统级输入法 | **否**，当前为 Tkinter 桌面应用原型 |
| 是否接入真实 LLM 排词 | **否**，`ModelReranker` 为 **STUB** |
| 日语模式 | **原型阶段**，仅支持基础罗马音转假名 |
| 候选窗是否跟随系统光标 | **否**，仅跟随 Tkinter Text 控件光标 |

## 快速启动

### 安装依赖

```bash
pip install -r requirements.txt
```

### GUI 图形客户端

```bash
python -m src.input_method.app
```

### 终端模拟器

```bash
python -m src.input_method.main
```

### 使用配置文件

```bash
python -m src.input_method.app --config config/default.json
python -m src.input_method.main --config config/default.json --mode english --log-level DEBUG
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--mode` / `-m` | 输入模式：pinyin / english / japanese |
| `--page-size` / `-p` | 每页候选数（默认 5） |
| `--config` | JSON 配置文件路径 |
| `--dict-path` | 外部词库路径 |
| `--use-model` | 启用 AI 排词（STUB） |
| `--learning-enabled` / `--no-learning` | 启用/禁用用户学习 |
| `--log-level` | 日志级别：DEBUG/INFO/WARNING/ERROR |
| `--show-console` | 保留控制台窗口（仅 GUI） |

## 词库管理

### 导入内置词库

```bash
python scripts/import_lexicon.py
```

### 导入外部词库

```bash
# JSONL 格式
python scripts/import_lexicon.py --input my_dict.jsonl --format jsonl --append

# TSV 格式
python scripts/import_lexicon.py --input my_dict.tsv --format tsv --append

# CSV 格式（自动检测）
python scripts/import_lexicon.py --input my_dict.csv
```

### 生成合成词库（压测用）

```bash
python scripts/generate_synthetic_lexicon.py --count 100000 --output .smoke_data/lexicon_100k.jsonl
```

### 词库压测

```bash
python scripts/benchmark_lexicon.py --dict-path data/lexicon/dict.jsonl
```

### 词库格式

每行一个 JSON 对象（JSONL），字段说明详见 [`data/lexicon/README.md`](data/lexicon/README.md)。

## 评估

```bash
python scripts/evaluate_candidates.py
```

输出 Top-1 / Top-3 / Top-5 命中率和召回覆盖率。**注意：当前评估基于小样本测试集，不代表日用质量。**

## 性能 Benchmark

```bash
# 引擎延迟
python scripts/benchmark_latency.py

# 词库延迟
python scripts/benchmark_lexicon.py
```

## 测试

```bash
python -m pytest tests/ -q
```

测试覆盖：引擎状态机、候选生成器、排序器、用户记忆、模式切换、端到端冒烟测试。

## 用户记忆

- 默认保存路径：`~/.golf_user_memory.json`
- 自动记录用户选词频率，优化排序
- 同键下其他词自动衰减，支持新词超越旧词
- 支持通过 `--user-memory-path` 指定自定义路径

### 清空用户记忆

- GUI：点击工具栏"清空用户记忆"按钮
- 终端：输入 `/clear_memory`
- 手动删除：`rm ~/.golf_user_memory.json`（Linux/Mac）或 `del %USERPROFILE%\.golf_user_memory.json`（Windows）

## 配置系统

默认配置文件：[`config/default.json`](config/default.json)

支持字段：mode、page_size、max_recall、dict_path、fuzzy_pinyin、candidate_layout、learning_enabled、log_level、theme、font_family、font_size 等。

## Windows 系统级输入法接入路线

**当前 golf 不是 Windows 系统级输入法。** 详细的 TSF/IMM32 接入工程路线见：

📄 [`docs/windows-ime-integration.md`](docs/windows-ime-integration.md)

核心路径：C++/Rust 实现 TSF DLL → Named Pipe/gRPC 与 Python 引擎通信 → regsvr32 注册为系统输入法。

## 项目结构

```
parameter-golf/
├── config/default.json              # 默认配置
├── data/lexicon/                    # 词库
│   ├── dict.jsonl                   # 主词库
│   └── README.md                    # 词库格式说明
├── docs/
│   └── windows-ime-integration.md   # Windows 系统级接入文档
├── scripts/
│   ├── benchmark_latency.py         # 引擎延迟 Benchmark
│   ├── benchmark_lexicon.py         # 词库延迟 Benchmark
│   ├── evaluate_candidates.py       # 候选质量评估
│   ├── generate_synthetic_lexicon.py# 合成词库生成
│   └── import_lexicon.py            # 词库导入
├── src/input_method/
│   ├── config.py                    # 配置系统
│   ├── engine.py                    # 核心引擎
│   ├── lexicon.py                   # 词库加载器
│   ├── user_memory.py               # 用户记忆
│   ├── app.py                       # GUI 入口
│   ├── main.py                      # 终端入口
│   ├── gui_editor.py                # GUI 编辑器
│   ├── gui_candidate_window.py      # 候选窗口
│   ├── cli_simulator.py             # 终端模拟器
│   ├── generator/                   # 候选召回
│   │   ├── base.py                  # 基类 + Candidate
│   │   ├── pinyin_generator.py      # 拼音召回
│   │   ├── english_generator.py     # 英文召回
│   │   └── japanese_generator.py    # 日语召回（原型）
│   └── reranker/                    # 候选排序
│       ├── base.py                  # 基类
│       ├── frequency_reranker.py    # 传统 baseline
│       └── model_reranker.py        # AI 排词 (STUB)
├── tests/                           # 测试
├── train_gpt.py                     # 待重构训练基线
├── PROJECT_GOALS_AND_READINESS.md   # 项目目标
├── 修改记录.md                       # 改动记录
└── requirements.txt                 # 依赖
```

## 后续开发优先级

1. 真实词库接入（开源词库、用户自定义词库）
2. 拼音切分算法优化（多音字、歧义消解）
3. 传统排序 baseline 持续调优
4. 用户常用词记忆权重策略优化
5. 性能 Benchmark 扩展（10万+词库压测）
6. 真实 LLM 排词接口接入
7. Windows TSF 系统级输入法 DLL
8. UI 美化与主题定制

## 许可证

见 [LICENSE](LICENSE)。第三方来源说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
