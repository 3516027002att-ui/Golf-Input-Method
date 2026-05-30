# data/lexicon/ — 词库目录

本目录存放 golf 输入法使用的中文词库文件。

## 词库格式

默认词库文件为 `dict.jsonl`，采用 **JSONL**（每行一条 JSON 记录）格式。

### 字段定义

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `word` | string | ✅ | 词条的中文文本，如 `"你好"` |
| `pinyin` | string | ✅ | 完整拼音（不含声调，无空格分隔），如 `"nihao"` |
| `short_pinyin` | string | ✅ | 简拼（每个音节首字母），如 `"nh"` |
| `freq` | number | ✅ | 词频权重，正数，数值越大表示越常用 |
| `source` | string | 可选 | 词条来源标识，如 `"lexicon"`、`"imported"`、`"synthetic"`，默认 `"lexicon"` |
| `domain` | string | 可选 | 所属领域标签，如 `"general"`、`"tech"`、`"medical"` |
| `enabled` | bool | 可选 | 是否启用该词条，默认 `true` |

### 示例

```json
{"word": "中国", "pinyin": "zhongguo", "short_pinyin": "zg", "freq": 6500, "source": "lexicon"}
{"word": "输入法", "pinyin": "shurufa", "short_pinyin": "srf", "freq": 3600, "source": "lexicon"}
{"word": "人工智能", "pinyin": "rengongzhineng", "short_pinyin": "rgzn", "freq": 2800, "source": "imported", "domain": "tech"}
```

## TSV 格式

也支持通过 `import_lexicon.py --format tsv` 导入 TSV 文件，字段顺序为：

```
word<TAB>pinyin<TAB>short_pinyin<TAB>freq[<TAB>source[<TAB>domain]]
```

## 许可证要求

- 本项目内置的约 500 个高频常用词不涉及任何第三方版权。
- 导入外部词库时，**使用者自行负责确认词库来源的许可证合规性**。
- 禁止将含有版权限制的商业词库直接提交到本仓库。
- 如果导入的词库有明确许可证，应在词条的 `source` 字段中标注来源，并在本目录下附加对应的许可证文件或说明。

## 导入方法

### 使用内置词库生成

```bash
python scripts/import_lexicon.py
```

### 从外部文件导入（覆盖模式）

```bash
python scripts/import_lexicon.py --input my_dict.txt --output data/lexicon/dict.jsonl
```

### 从外部文件追加合并

```bash
python scripts/import_lexicon.py --input my_dict.txt --output data/lexicon/dict.jsonl --append
```

### 从 JSONL 文件导入

```bash
python scripts/import_lexicon.py --input external.jsonl --format jsonl --append
```

### 从 TSV 文件导入

```bash
python scripts/import_lexicon.py --input external.tsv --format tsv --append
```

### 生成合成测试词库

```bash
python scripts/generate_synthetic_lexicon.py --count 5000 --output data/lexicon/synthetic.jsonl
```

### 词库性能基准测试

```bash
python scripts/benchmark_lexicon.py --dict-path data/lexicon/dict.jsonl
```

## 词频分布

词频字段 (`freq`) 建议遵循 Zipf 分布：高频词的词频值较高（如 `9000`+），低频专业词的词频值较低（如 `100`~`500`）。导入脚本在合并时会取较大词频值。

## 注意事项

1. 词库文件不应包含任何用户个人信息或隐私数据。
2. `dict.jsonl` 会被 Git 跟踪，请勿在其中放入过大的词库文件（建议 < 10MB）。
3. 大型词库建议通过 `.gitignore` 排除，仅保留生成/导入脚本。
