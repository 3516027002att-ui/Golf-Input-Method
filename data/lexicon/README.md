# 词库目录 (lexicon)

## 文件说明

- `dict.jsonl` — 当前基础中文词库 (~435 条)，包含常用单字、双字和三字词。
  格式为 JSONL，每行一个 JSON 对象。来源为手工整理的无版权争议常用词。
- `README.md` — 本文件。

## 词库格式

每行一个 JSON 对象，标准字段：

```json
{"word": "中国", "pinyin": "zhongguo", "short_pinyin": "zg", "freq": 6500.0, "source": "lexicon"}
```

字段说明：
- `word` (必填): 词语文本
- `pinyin` (必填): 全拼（小写，不含声调）
- `short_pinyin` (必填): 首字母简拼
- `freq` (必填): 静态词频，数字越大越优先
- `source` (可选): 词条来源，如 `lexicon`、`user`、`synthetic`

## 工业级词库方案

当前采用方案 B：提供完整的词库生成、导入和压测工具。示例词库 (~435 条) 仅用于验证流程，
**不得**将其标识为工业级词库。

### 工具链

| 脚本 | 功能 |
|------|------|
| `scripts/generate_synthetic_lexicon.py` | 生成 10 万级合成拼音词库用于压测 |
| `scripts/import_lexicon.py` | 导入外部词库，合并去重，同步到 `dict.jsonl` |
| `scripts/benchmark_lexicon.py` | 词库加载和查询性能压测 |

### 操作流程

```bash
# 生成 10 万合成词库
python scripts/generate_synthetic_lexicon.py --count 100000 --output .smoke_data/lexicon_100k.jsonl

# 导入到标准格式
python scripts/import_lexicon.py --input .smoke_data/lexicon_100k.jsonl --output .smoke_data/imported.jsonl

# 压测
python scripts/benchmark_lexicon.py --dict-path .smoke_data/imported.jsonl
```

### 使用外部词库启动

```bash
python -m src.input_method.app --dict-path .smoke_data/imported.jsonl
```

## 数据许可

- `dict.jsonl` 中的词条为手工整理的无版权争议常用词。
- 合成词库由脚本随机生成，无版权问题。
- 若导入外部词库，用户需自行确保来源合法并遵守相应许可证。
