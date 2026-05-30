# 数据资产说明

`data/` 目录保存 golf 当前可复用的数据和 tokenizer 资产。它们来自公开语料处理流程，可作为自动排词模型的启动语料，但不等同于真实输入法用户日志。

## 当前布局

- `data/datasets/fineweb10B_sp1024/`
  - 公开语料分词后的二进制 shard。
  - 可用于预训练、语言建模基线、候选排序样本构造和数据处理 smoke。
- `data/tokenizers/`
  - `fineweb_1024_bpe.model`
  - `fineweb_1024_bpe.vocab`
- `data/tokenizer_specs.json`
  - tokenizer 规格。
- `data/lexicon/`
  - `dict.jsonl`
  - 外部高频中文词库（JSONL 格式）。目前通过 `scripts/import_lexicon.py` 自动扩充，包含了约 500 个常用无版权争议的汉字和词组，用于输入法底座的召回和排序。
  - 格式为：每行一条 JSON 格式记录，必须包含 `word`、`pinyin`、`short_pinyin`、`freq`、`source` 字段，例如：
    `{"word": "你好", "pinyin": "nihao", "short_pinyin": "nh", "freq": 6700, "source": "lexicon"}`
- `data/cached_challenge_fineweb.py`
  - 现有数据缓存下载脚本，保留用于重建公开语料缓存。
- `data/download_hf_docs_and_tokenize.py`
  - 文档下载与重新分词脚本，保留用于后续数据重建或 tokenizer 实验。

## 已知缓存

当前主缓存目录为：

```text
data/datasets/fineweb10B_sp1024/
```

历史审计口径：

- 训练分片：`fineweb_train_*.bin`
- 验证分片：`fineweb_val_000000.bin`
- tokenizer：`data/tokenizers/fineweb_1024_bpe.model`

实施训练或清理前，应先确认这些文件仍存在。不要把 `data/datasets/` 和 `data/tokenizers/` 当作比赛遗留产物删除。

## 在 golf 中的用途

公开语料可用于：

- 构造上下文窗口和候选排序样本。
- 训练通用排词模型的初始表示。
- 评估候选排序模型在公开文本上的泛化能力。
- 验证数据加载、batch 构造和 tokenizer roundtrip。

公开语料不能直接代表：

- 用户真实输入习惯。
- 拼音到汉字候选召回质量。
- 个性化词库效果。
- 输入法候选栏交互体验。

## 后续数据方向

后续应新增独立的数据构造流程，至少包含：

- 排词样本格式：上下文、候选列表、目标候选、样本来源和可选权重。
- 候选生成来源：词典、规则召回、语言模型补全或其他召回器。
- 数据切分：训练、验证、测试按来源和时间隔离。
- 评估指标：首选命中率、Top-K 命中率、平均排序损失和推理延迟。

任何用户输入日志都必须默认关闭采集。只有在用户明确授权后，才能在本地最小化记录，并且必须提供脱敏和删除机制。
