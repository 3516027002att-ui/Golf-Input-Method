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

## 训练数据位置

模型训练和公开训练数据重建默认使用仓库外目录：

```text
C:\training data
```

标准布局为 `C:\training data\datasets\fineweb10B_sp1024\fineweb_train_*.bin`、`C:\training data\datasets\fineweb10B_sp1024\fineweb_val_*.bin` 和 `C:\training data\tokenizers\fineweb_1024_bpe.model`。`train_gpt.py`、`data/cached_challenge_fineweb.py` 和 `data/download_hf_docs_and_tokenize.py` 默认都读取或写入该目录。

可用 `TRAINING_DATA_ROOT` 改变训练数据根目录，也可用 `DATA_PATH` / `TOKENIZER_PATH` 分别覆盖训练 shard 目录和 tokenizer 路径。仓库内 `data/` 只保留数据脚本、`tokenizer_specs.json`、输入法词库和说明；`data/lexicon/` 不属于模型训练语料缓存，仍保留在仓库内。

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
## Plan1 候选排序训练程序

Plan1 是当前输入法项目的第一版真实候选 reranker 训练入口，不复用历史比赛里的 depth-recurrence plan1。训练侧保持官方原版风格，只有一个文件：`training/plan1_reranker.py`。训练数据、checkpoint 和模型产物默认写入仓库外：

```text
C:\training data\golf-ime-data
```

训练依赖单独放在 `requirements-train.txt`，不会进入运行时最小依赖：

```bash
python -m pip install -r requirements-train.txt
```

`train` 子命令和 Plan1 checkpoint 加载必须依赖 `torch`、`transformers` 等训练依赖；只有 `build-data` 子命令可以在未安装 `transformers` 时单独跑数据构建 smoke。

构建数据：

```bash
# smoke：只合并现有样本，不遍历 zhwiki 全量语料
python training/plan1_reranker.py build-data --skip-corpus-generation --output-prefix plan1_smoke

# full：遍历 C:\training data\golf-ime-data\processed\zhwiki_corpus.jsonl 全量生成
python training/plan1_reranker.py build-data
```

`build-data` 默认按 `source_doc_id` 稳定哈希切成 `train/val/test = 90/5/5`，避免训练集扩容后验证集或测试集仍停留在过小规模。正在运行训练时不要重建默认 `plan1_ranking_*.jsonl`，应等当前训练完成后再用新切分重建数据，并重新启动新一轮训练。

如果需要扩大验证集和测试集的绝对规模，可以用新前缀生成多来源数据，不覆盖当前训练文件：

```bash
python training/plan1_reranker.py build-data \
  --output-prefix plan1_ranking_multi_1m_90_5_5 \
  --max-generated-per-corpus 250000 \
  --corpus-file zhwiki_corpus.jsonl \
  --corpus-file news2016zh_corpus.jsonl \
  --corpus-file baike2018qa_corpus.jsonl \
  --corpus-file webtext2019zh_corpus.jsonl
```

训练：

```bash
python training/plan1_reranker.py train --per-device-batch 3
```

默认 encoder 为 `hfl/chinese-macbert-base`，训练方式为 listwise cross-encoder reranker，输出目录为：

当前默认 `per_device_batch` 已从 4 降为 3，以降低 RTX 4060 Laptop 长时间训练时的整机稳定性风险；若仍触发重启或显卡驱动异常，再显式降到 `--per-device-batch 2`。

```text
C:\training data\golf-ime-data\models\plan1-reranker\best
```

从旧 Plan1 checkpoint 接力到更大数据集时，默认仍要求标签空间完全一致。若确认只是 full data 新增了领域/来源标签，可显式使用兼容扩展策略：

```bash
python training/plan1_reranker.py train \
  --init-from-checkpoint "C:\training data\golf-ime-data\models\plan1-reranker\best" \
  --checkpoint-label-space-policy expand-compatible \
  --train-path "C:\training data\golf-ime-data\processed\plan1_ranking_colab_ready_20260602_train.jsonl" \
  --val-path "C:\training data\golf-ime-data\processed\plan1_ranking_colab_ready_20260602_val.jsonl" \
  --test-path "C:\training data\golf-ime-data\processed\plan1_ranking_colab_ready_20260602_test.jsonl" \
  --output-dir "C:\training data\golf-ime-data\models\plan1-colab-warmstart-run1" \
  --colab-pro-mode \
  --cache-data-dir /content/golf-ime-cache \
  --auto-batch-probe \
  --target-effective-batch 24 \
  --checkpoint-every-steps 0 \
  --checkpoint-every-seconds 3600 \
  --max-runtime-seconds 39000 \
  --max-epochs 3 \
  --min-epochs 1 \
  --patience 2
```

`expand-compatible` 只在训练 warm-start 中生效：它按 label 名称复制旧 embedding 行，新增 label 从旧 `unknown` 行初始化，并写出 `label_space_migration.json`。推理侧 checkpoint 加载仍保持严格，不做隐式迁移。

训练输出目录会记录数据反馈信息：`data_manifest.json` / `training_config.json` 中包含 `data_version`、`cleaning_rules_version` 和上游数据 manifest 摘要；每轮验证和最终测试会在 `cleaning_feedback\` 下写出脱敏报告。报告按 `sample_source`、`license_bucket`、`domain`、`source_doc_prefix`、候选数量桶、输入长度桶和 target 初始位置桶统计 loss / Top-K / MRR，并只保存高损失样本的哈希 ID、来源桶和长度桶，不保存上下文原文、target 文本或候选文本。清洗规则仍应放在 `audit-data` / `clean-data` 数据管线中，训练侧只负责给下一轮清洗提供反馈信号。

本地 agent 可通过同步后的 Google Drive 输出目录只读监督训练：

```bash
python scripts/watch_plan1_colab.py --run-dir "G:\My Drive\golf-ime-runs\plan1_colab_ready_20260602_warmstart_run1" --once
```

断电或异常中断后的接力训练不要直接复用默认输出目录。先从已备份的 `best` checkpoint 进行权重 warm-start，写入新的隔离目录，并用 `--dry-run` 只验证数据、模型和 manifest，不启动优化：

```bash
python training/plan1_reranker.py train \
  --init-from-checkpoint "C:\training data\golf-ime-data\models\plan1-reranker\best" \
  --output-dir "C:\training data\golf-ime-data\models\plan1-reranker-continue-YYYYMMDD-epoch3" \
  --per-device-batch 1 \
  --grad-accum 8 \
  --max-epochs 1 \
  --min-epochs 1 \
  --patience 1 \
  --device cpu \
  --dry-run
```

去掉 `--dry-run` 和 `--device cpu` 后才是真正训练。该流程只恢复模型权重、tokenizer 和标签空间，optimizer、scheduler、GradScaler 会重新初始化；因此它是安全 fine-tune 接力，不是精确恢复中断 step。

Plan1 不会默认替换当前输入法行为，必须显式启用：

```bash
python -m src.input_method.main --use-model --model-path "C:\training data\golf-ime-data\models\plan1-reranker\best"
```

如果 checkpoint 缺失、加载失败、推理异常或超出 `PLAN1_RERANKER_TIMEOUT_MS`，`ModelReranker` 会自动回退到 `FrequencyReranker`。包含 OpenIME / Wikipedia 派生数据的 Plan1 产物只适合作为内部研究模型，公开发布前必须重新做许可证审计。

## Context reranker v2 审计入口

`training/context_reranker_v2.py` 是用于证明上下文重排能力的干净实验台，不是最终生产输入法模型。默认在线模式只使用 `context_before`、`composing` 和 `candidate`；`context_after` 只用于句中编辑、离线纠错或已有文本重排场景。

常用命令：

```bash
python training/context_reranker_v2.py audit-splits --train <train.jsonl> --val <val.jsonl> --test <test.jsonl> --report <audit.md>
python training/context_reranker_v2.py train --train <train.jsonl> --val <val.jsonl> --output-dir <checkpoint> --context-mode online
python training/context_reranker_v2.py train --train <train.jsonl> --val <val.jsonl> --output-dir <checkpoint-random-label> --label-mode random
python training/context_reranker_v2.py eval --data <val.jsonl> --checkpoint <checkpoint> --context-mode online
python training/context_reranker_v2.py eval --data <val.jsonl> --checkpoint <checkpoint> --context-mode none
python training/context_reranker_v2.py eval --data <val.jsonl> --checkpoint <checkpoint> --context-perturb shuffle
python training/context_reranker_v2.py eval --data <val.jsonl> --checkpoint <checkpoint> --candidate-order shuffle
```

本轮 context reranker v2 训练准备优先使用 clean_dataset_v3 的新语料切分，不把旧 `train.jsonl` 作为第一轮主训练集：

```text
G:\我的云端硬盘\golf-ime-data-rebuild\clean_dataset_v3\left_context_only\train_new_corpus.jsonl
G:\我的云端硬盘\golf-ime-data-rebuild\clean_dataset_v3\left_context_only\val_new_corpus_v2.jsonl
G:\我的云端硬盘\golf-ime-data-rebuild\clean_dataset_v3\left_context_only\test_new_corpus_v2.jsonl
```

Windows 本地审计入口：

```bat
scripts\run_context_v2_local_audit.bat
```

该脚本写出 `reports/context_v2_new_corpus_audit.md`。Colab 端训练准备见 `notebooks/colab_context_reranker_v2.ipynb`，其中包含挂载 Google Drive、clone/checkout 分支、安装 `requirements-train.txt`、复制新语料、运行 audit、smoke train、eval 和保存 checkpoint/eval 产物的完整命令。

离线 checkpoint 推理入口：

```bash
python scripts/predict_context_reranker_v2.py --checkpoint <checkpoint> --context-before "今天我想" --composing "nihao" --candidate "你好" --candidate "拟好"
```

该入口只用于本地离线验证候选排序，不会接入输入法 UI，也不会让 `ModelReranker` 的 STUB 冒充真实 v2 模型。

审计报告必须优先解释异常高分来源：`candidates[0]`、`static_rank`、`freq`、random、记忆 baseline、split overlap、候选生成器偏置、困难负例比例、no-context/shuffled-context/shuffled-candidates sanity，以及原候选第一正确时的保持率、错误时的纠错率和误伤率。
