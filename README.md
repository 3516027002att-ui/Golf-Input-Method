# golf

本仓库已经从 Parameter Golf 比赛项目转型为 golf。golf 的目标是利用现有公开语料和 tokenizer 资产，训练一个用于输入法候选词、候选短语和句子补全的自动排词模型，并逐步补齐本地输入法客户端、候选生成、排序推理、评估和模型导出流程。

当前阶段是项目转型后的整理期：历史比赛脚本、日志、checkpoint 和实验记录会被清理；可复用的数据资产、依赖说明、许可证和主训练基线会保留。仓库不会声称已经实现完整输入法客户端或正式排词模型入口，后续实现必须以真实代码、真实命令和真实验证结果为准。

## 项目入口

- GitHub 仓库：<https://github.com/3516027002att-ui/Golf-Input-Method>
- 当前展示名：`golf`
- 本地历史目录名可能仍是 `parameter-golf`，但后续协作和文档引用应以 GitHub 仓库 `Golf-Input-Method` 和项目名 `golf` 为准。

## 当前状态

- 已保留公开语料缓存：`data/datasets/fineweb10B_sp1024/`
- 已保留 tokenizer：`data/tokenizers/fineweb_1024_bpe.model`
- 已保留数据处理脚本：`data/cached_challenge_fineweb.py`、`data/download_hf_docs_and_tokenize.py`
- 已保留主训练基线：`train_gpt.py`
- 已新增本地输入法原型代码：`src/input_method/`
- 已新增最小单元测试：`tests/`
- 尚未实现正式输入法客户端、推理服务、排词模型训练入口和导出格式

`train_gpt.py` 目前只作为可复用训练基线和重构参考。后续应将它拆分或替换为面向排词任务的训练代码，而不是继续沿用比赛指标和提交约束。

## 快速启动与命令手册

### 启动命令

| 模式 | 命令 | 说明 |
|------|------|------|
| **系统 IME** 🆕 | `python -m src.input_method.ime_app` | **全局键盘钩子，可在任意应用（记事本/浏览器等）中使用** |
| 桌面双击 | `scripts\launch_golf.bat` | 启动系统 IME 模式 |
| GUI 编辑器 | `python -m src.input_method.app` | 自带编辑器的 Demo 原型 |
| 终端模拟器 | `python -m src.input_method.main` | 命令行交互测试 |
| 后台启动器 | `python -m src.input_method.tray_app` | 单实例保护后台入口 |

**系统 IME 模式快捷键**：
- `Ctrl+Shift`：切换中/英文模式
- 关闭控制台窗口：退出 IME

### 评估与测试命令
- **候选质量评估**：
  ```bash
  python scripts/evaluate_candidates.py
  ```
- **响应延迟 Benchmark**：
  ```bash
  python scripts/benchmark_latency.py
  ```
- **词库扩充/导入脚本**：
  ```bash
  python scripts/import_lexicon.py --input <文件> --output <文件>
  ```
- **合成词库生成 (10万级压测)**：
  ```bash
  python scripts/generate_synthetic_lexicon.py --count 100000
  ```
- **词库性能压测**：
  ```bash
  python scripts/benchmark_lexicon.py --dict-path <词库文件>
  ```

### 工业级词库压测结果 (10 万 smoke)
```
生成: 100,000 条 → 导入去重: 99,449 条
加载耗时: 0.231s  |  P95 查询: 22.06ms  |  内存: 46.5MB
✅ 10 万级词库加载达标
```

### 交互操作与快捷键
- **中/英文切换**：GUI 点击 “切换中/英文” 按钮，终端输入 `/mode`。仅支持中英双路。
- **中文模式**：输入拼音字母触发候选匹配；空格选首词；数字键 1-5 选对应候选。
- **英文模式**：字母直通上屏，不进入 IME 候选流程。
- **基础误触纠错**：支持 i/o/u 相邻键位误触纠正（如 `nohao`→纠错为`你好`），纠错候选标记 `correction` 来源，不挤占精确匹配首位。
- **修改缓冲区**：按 `Backspace` 删除缓冲区末尾拼音；`Esc` 取消输入；`Enter` 原文上屏。
- **候选词翻页**：按 `-` 键 (PageUp) 或 `=` 键 (PageDown)。
- **清空用户学习记忆**：GUI 点击工具栏 “清空用户记忆” 按钮；终端输入 `/clear_memory`。

### 词库格式说明
外部词库文件位于 `data/lexicon/dict.jsonl`，采用每行一条 JSON 格式记录，例如：
```json
{"word": "中国", "pinyin": "zhongguo", "short_pinyin": "zg", "freq": 6500, "source": "lexicon"}
```
可通过 `python scripts/import_lexicon.py` 进行重构、合并与自定义扩充。

### 用户记忆文件位置
默认持久化在用户主目录下：`~/.golf_user_memory.json`，不会被 Git 仓库跟踪，有效保护用户隐私。

### 当前 v0 可试用输入示例
在中文拼音模式下，请尝试输入以下拼音：
- `nihao` -> `你好`
- `nh` -> `你好`
- `wo` -> `我`
- `women` -> `我们`
- `jintian` -> `今天`
- `zhongguo` -> `中国`
- `zg` -> `中国`
- `shurufa` -> `输入法`
- `xiangyao` -> `想要`
- `woxiangyao` -> `我想要`

### 当前限制 (重要说明)
1. **非系统级输入法**：目前是本地输入法框架 + GUI 编辑器 + 后台启动器，未挂接 Windows TSF/IMM32 系统级输入法框架。不能在记事本/浏览器等外部应用中使用。
2. **未接入真实 LLM**：机器学习排序 `ModelReranker` 依然是桩类 (STUB)，当前默认使用基于词频与用户记忆的传统 baseline 排序。
3. **日语模式内部预留**：`JapaneseCandidateGenerator` 代码保留，通过 `engine.switch_language('ja')` 内部接口可访问。**当前用户界面不暴露日语切换入口**，日语模式仅是原型骨架，不具备日常日语输入能力。
4. **外部词库仍非工业级**：默认词典为扩充后的 400+ 个高频核心常用词。提供完整的 10 万级合成词库生成和压测工具链（方案 B）。
5. **基础误触纠错已实现**：覆盖 i/o/u 相邻键位误触，纠错候选标记 `correction` 来源，不挤占精确匹配首位。
6. **台式启动已实现**：`scripts/launch_golf.bat` 可直接双击启动（无需命令行），带单实例保护。

## 当前状态与已实现底座 v0

目前，项目已完成了 **输入法底座 v0** 的构建，实现了一个在没有大语言模型时也能勉强可用的基础输入法框架：

- **外部词库加载框架**：支持从 `data/lexicon/dict.jsonl` 中动态载入高频词库。内置 `RAW_WORDS` 降级为 fallback。
- **连续拼音切分与候选召回**：在 `PinyinCandidateGenerator` 中内置 400+ 标准汉语拼音音节匹配，支持最大匹配法将输入串进行连续切分（例如 `woxiangyao` -> `['wo', 'xiang', 'yao']`），并实现了组合切分片下的多段短语拼合召回。
- **用户常用词记忆与权重持久化**：用户选中词后自动累加其在对应输入键下的权重，并对同键下其他候选权重做微衰减，权重自动持久化到本地 `~/.golf_user_memory.json`。支持通过 `engine.clear_user_memory()` 一键清空。
- **日语输入模式（内部预留）**：`JapaneseCandidateGenerator` 代码保留，通过 `engine.switch_language('ja')` 可编程访问。当前 GUI/终端不暴露日语切换入口。
- **基础键位误触纠错**：`PinyinCandidateGenerator` 内置 i/o/u 相邻键位纠错，标记 `correction` 来源，默认不挤占精确匹配首位。
- **后台启动与单实例保护**：`src/input_method/tray_app.py` + `scripts/launch_golf.bat` 提供双击启动和单实例保护。
- **评估与 Benchmark 基准**：提供了自动评估指标脚本 `scripts/evaluate_candidates.py` 和延迟性能评测 `scripts/benchmark_latency.py`。

### 🚨 诚实说明 (重要约束)

- **AI 排词模型仍是桩类**：当前默认没有接入真实大语言模型排词，`ModelReranker` 依然是个桩类（STUB）而非完整 AI 模型。
- **GUI 记事本依然是 Demo**：目前客户端界面仍是一个内置编辑器记事本 Demo，并非挂载到系统级的输入法客户端。
- **日语模式仅是证明通路的原型骨架**：仅内置了极少数常用假名转换映射，无法进行日常的日语流畅输入。
- **新评估和性能测试仅为小样本验证**：两脚本仅代表基本开发冒烟测试，不代表最终实际日常使用的打字质量。
- **详细约束与偏离** 见 `PROJECT_GOALS_AND_READINESS.md`。

## 目标架构

完整 AI 输入法按以下模块推进：

1. 客户端输入层
   - 管理输入缓冲区、光标上下文、候选栏状态和用户选择事件。
   - 先以本地原型验证交互，再接入系统级输入法框架。

2. 候选生成层
   - 根据拼音、前缀、上下文和词典召回候选词、短语或补全文本。
   - 首版可以使用规则召回和轻量词表，后续再接入更复杂的生成模型。

3. 自动排词模型
   - 输入：上下文、当前 composing 文本、候选列表和可选用户偏好特征。
   - 输出：每个候选的排序分数。
   - 训练目标优先使用可解释、可回放的候选排序任务，避免把生成式语言模型指标误当作输入法体验指标。

4. 本地推理层
   - 提供低延迟候选排序接口。
   - 默认离线可用，不依赖网络调用。
   - 不记录明文用户输入，除非用户明确开启并完成脱敏策略。

5. 评估与导出层
   - 评估排序准确率、首选命中率、Top-K 命中率、延迟和内存占用。
   - 导出面向本地推理的模型和必要词表，保留可复现实验记录。

## 数据资产

当前可用数据位于 `data/`：

- `data/datasets/fineweb10B_sp1024/`：公开语料分词后的二进制 shard，可用于预训练或构造排序任务的启动语料。
- `data/tokenizers/`：SentencePiece tokenizer 文件。
- `data/tokenizer_specs.json`：tokenizer 配置。
- `data/README.md`：数据布局、来源和校验说明。

这些数据并不等价于真实输入法日志。后续如需使用用户输入样本，必须默认关闭采集，并要求显式授权、最小化字段、脱敏处理和本地可删除。

## 后续路线

短期目标：

- 定义输入法排词任务的数据样本格式。
- 从公开语料构造可复现的候选排序训练集。
- 将 `train_gpt.py` 中可复用的训练、checkpoint、评估逻辑迁移到新的排词训练入口。
- 建立最小评估集：首选命中率、Top-3 命中率、平均排序损失和推理延迟。

中期目标：

- 实现本地候选生成与排序推理原型。
- 建立模型导出格式和加载校验。
- 增加真实输入法交互层，但保持隐私默认安全。

长期目标：

- 支持个性化词频、领域词库和本地增量学习。
- 支持多输入模式和多语言扩展。
- 在不牺牲隐私和延迟的前提下提升候选质量。

## 开发原则

- 不新增技术债：先复用现有数据和训练基础，再逐步拆分。
- 不编造能力：文档、命令和指标必须对应真实实现。
- 不写入明文凭据和用户隐私样本。
- 不为了短期演示关闭校验、吞异常或硬编码路径。
- 涉及训练数据、模型导出、推理接口时必须同步更新文档和验证记录。
