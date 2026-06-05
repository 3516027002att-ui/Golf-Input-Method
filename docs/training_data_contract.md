# 训练数据合同与上下文选词模型经验记录

本文档用于固定 Golf Input Method 后续训练文件的“问题合同”。它吸收了 Parameter Golf 官方 baseline 的写作方式、Parameter Golf SOTA 的演化经验，以及 Rime、Fcitx/libpinyin、OpenIME、PinyinGPT/Transformers4IME 等输入法/选词模型经验。

## 1. 核心判断

输入法选词模型的目标不是从零生成中文，也不是替代词库、拼音解析、用户词典和候选召回。更可靠的路线是：

传统候选系统负责召回候选；模型只负责在候选列表内根据上下文重排。

因此训练样本必须尽量贴近真实输入过程：用户输入了什么拼音、候选栏当时出现了哪些候选、原始排序是什么、用户最终选择了哪个、上下文是什么、是否翻页、是否删除重输、是否来自用户词典或领域词库。

## 2. 从 Parameter Golf 学到的训练文件写法

官方 baseline 的价值不在于它就是 SOTA，而在于它把任务合同固定得很清楚：

- 单文件入口可以保留，但必须分区清楚：超参、数据读取、审计、模型、训练、评估、导出。
- 默认脚本应该是 launching-off point，而不是把所有实验性技巧塞进一份难以审计的巨型文件。
- 指标口径必须稳定。Parameter Golf 用固定验证集和 tokenizer-agnostic BPB；本项目也必须固定 split、baseline、ablation 和 sanity check。
- 训练脚本要把“是否可信”放在“是否能跑”前面。能跑出漂亮指标不能证明任务成立。
- SOTA 常常是多代系统堆栈的结果。最后一次 PR 可能只是小调参，但它继承的是 tokenizer、架构、训练、压缩、TTT、评测口径长期迭代后的系统。

对本项目的翻译是：先写清楚输入法排序数据合同，再让 Codex 落地实现。Codex 适合施工，GPT Thinking 适合先把任务边界、泄漏红线和验证设计想清楚。

## 3. 从输入法社区学到的经验

Rime、Fcitx/libpinyin、万象拼音等项目说明，成熟输入法不是一个神经模型，而是混合系统：

- 词库、词频、用户词典和自学习仍然是基础。
- filter / reranker 层很适合接入神经模型。
- 用户需要可控性：忘记词、关闭自动调频、手动造词、领域词库优先级都很重要。
- 开源输入法用户尤其重视本地、隐私、可解释和可迁移。
- 真实难点集中在简拼、缩写、短上下文、领域词和多候选都合理的场景。

OpenIME 与 PinyinGPT/Transformers4IME 的经验说明，神经输入法训练必须区分拼音源、中文目标、候选/词典、训练集与测试集。缩写拼音会让候选空间爆炸，因此模型必须同时看拼音、上下文和候选，而不是只学习词频排序。

## 4. 数据合同

新的训练样本建议使用 JSONL，每行一个候选选择事件。

最小字段：

```json
{
  "sample_id": "stable unique id",
  "source_doc_key": "document-or-session level key for split isolation",
  "context_before": "上文",
  "context_after": "下文，可为空",
  "composing": "用户输入的拼音或简拼",
  "candidates": ["候选1", "候选2", "候选3"],
  "target_index": 1
}
```

推荐字段：

```json
{
  "original_ranks": [1, 2, 3],
  "candidate_meta": [
    {"freq": 123.0, "source": "system_dict", "match_type": "exact_pinyin"},
    {"freq": 12.0, "source": "user_dict", "match_type": "abbr"},
    {"freq": 3.0, "source": "domain_dict", "match_type": "fuzzy"}
  ],
  "event_type": "commit",
  "domain": "coding/math/general",
  "user_action": "selected_first_page",
  "license_bucket": "open|attribution|research_only|unknown"
}
```

兼容旧字段：

- 如果旧样本有 `target`，新脚本会在 `candidates` 中寻找它并恢复 `target_index`。
- 如果旧样本有 `candidate_features`，新脚本会把它归一化为 `candidate_meta`。
- 如果只有 `source_doc_id`，新脚本会作为 `source_doc_key` 的备选来源。

## 5. 泄漏红线

以下情况必须在审计报告中标记，严重时禁止进入正式训练集：

- train/val/test 中有相同 `source_doc_key`。
- train/val/test 中有相同或近似相同的上下文、拼音、候选、目标组合。
- `target_index` 分布异常，例如正确答案长期固定在第一个位置或某个 hash 插入规律。
- 候选负例过弱，例如候选数量长期小于 3，或者错误候选明显与拼音/上下文无关。
- 模型输入包含目标词专属字段，例如 `target`、`target_pinyin_full`、`target_pinyin_abbr`。
- 去掉上下文后仍然接近满分。
- 打乱上下文后仍然接近满分。
- 只用 `static_rank` 或 `freq` 的 baseline 已经接近神经模型。
- 验证集和测试集来自同一篇文章、同一会话、同一模板批量改写样本。

## 6. 必须报告的 baseline

每次训练前至少报告：

- original rank top1/top3/top5。
- frequency top1/top3/top5，如果有频率字段。
- random top1/top3/top5。
- candidate count 分布。
- target index 分布。
- source/domain/license 分布。
- split 间 sample signature 重复数。
- split 间 source_doc_key 重复数。

这些 baseline 的意义是判断模型到底学到了上下文，还是只是在复读候选系统原始排序。

## 7. 新训练文件的设计原则

`training/context_reranker_v2.py` 的目标是替代 `plan1_reranker.py` 的“规则合成优先”路线，变成“数据合同和审计优先”路线。

原则：

- 默认不把 `static_rank`、`freq` 等强旁路特征喂给神经模型。
- 这些特征只用于 baseline 和审计。
- 模型输入只包含上文、下文、composing 和 candidate。
- 训练前先跑 `audit-data`。
- split 按 `source_doc_key` 稳定 hash 进行，优先避免同文档/同会话泄漏。
- 训练指标同时报告 neural topK 与 baseline topK。
- 如果 neural 只比原始 rank baseline 高一点，则说明模型价值有限。
- 如果无上下文/打乱上下文仍很高，则说明数据集或任务定义有问题。

## 8. 后续路线

第一阶段：用真实候选日志或高质量 OpenIME/PinyinGPT 风格数据重建数据集。

第二阶段：训练纯文本 cross-encoder reranker，证明上下文确实提供增益。

第三阶段：加入轻量 side features，但必须通过 ablation 证明 side features 没有吞掉上下文价值。

第四阶段：做端侧小模型蒸馏或量化，把 reranker 接到候选 filter 层。

第五阶段：加入用户短期记忆、领域词库、最近上屏上下文，但必须保证隐私和可控性。

## 9. 给 Codex / Claude Code 的执行边界

当让 agent 修改训练系统时，提示词应强调：

- 先读本文档，再修改训练代码。
- 目标是提高数据可信度和评测可信度。
- 不要为了凑指标生成更多弱合成数据。
- 不要把审计警告静默吞掉。
- 不要把目标词、目标拼音、target 字段拼进模型输入。
- 新增复杂功能前先补 audit 和 baseline。
- 输出详细日志，尤其记录数据被过滤、修复、切分和 baseline 结果。


## 10. 0.995 异常高分审计口径

如果 reranker 在验证集上出现接近 0.995 的 top1，第一反应不应是继续堆模型，而应先证明这个数字来自哪里。旧 `plan1_reranker.py` 最大问题不是“用了规则和强特征”，而是没有把规则能力、候选原序能力、词频能力、记忆能力、上下文能力拆开计量。

必须固定报告以下数字：

- `candidates[0]` baseline top1/top3/top5。
- random baseline top1/top3/top5。
- `static_rank` baseline top1/top3/top5。
- `freq` baseline top1/top3/top5。
- `raw_input -> target` 记忆 baseline。
- `context_suffix_32 + raw_input -> target` 记忆 baseline。
- `candidates_set -> target` 记忆 baseline。
- no-context model top1/top3/top5。
- shuffled-context top1/top3/top5。
- shuffled-candidates top1/top3/top5。
- dummy-first model top1/top3/top5。

必须固定审计 split 泄漏：完整样本重复、`raw_input + target` 重复、`context_suffix_32 + raw_input + target` 重复、`candidates_set + target` 重复、`source_doc_key` 重复、原句/段落/模板级签名重复，以及近重复上下文签名跨 split 风险。

必须固定审计候选生成器：target 是否被固定插入 candidates、target 插入位置是否有偏置、target 是否是唯一能匹配拼音或简拼的候选、负例是否随机低质、target 是否总是唯一正常词、候选原始 rank 是否已经近似标签、困难负例比例是否足够。

必须审计评估链路本身：eval loader 真的读取 val/test，文件物理行数、有效样本数和报告样本数一致，label 与 candidates 没有错位，topK 排序方向正确，padding 没有参与排序，`target_index` 没有被错误重置成 0，batch 内样本没有错位，target 文本、target 拼音或 teacher forcing 信息没有进入 eval 输入。

硬 sanity check：随机打乱 label 后训练，val top1 应接近 random baseline；dummy model 永远选第一个候选，用来暴露候选顺序偏置；打乱候选顺序后评估，用来暴露 rank 依赖和 label 错位；打乱 context 后评估，用来判断上下文是否真的有贡献；清空 context 后评估，用来判断模型是否只学了 composing、candidate 或候选顺序。

红线：

- 如果 `candidates[0]` baseline top1 > 0.95，当前 val 不适合证明 reranker 能力。
- 如果 no-context top1 与 full/online-context top1 差距 < 1%，上下文贡献未被证明。
- 如果 train-val `raw_input + target` overlap > 30%，随机切分风险极高。
- 如果 `context_suffix_32 + raw_input` 记忆 baseline > 0.90，基本可判定泄漏或模板化。
- 如果 exact sample overlap 或 `source_doc_key` 跨 split > 0，split 必须重做。
- 如果随机标签训练仍显著高于 random baseline，评估或训练链路疑似错误。
- 如果 hard-negative 子集 top1 大幅低于普通 val，普通 val 太简单，不能支撑 0.995 结论。

## 11. v2-clean 与 production hybrid 的边界

`context_reranker_v2.py` 的第一身份是 clean context 实验台，不是最终生产模型。它的目标是隔离并证明：在不依赖 `static_rank`、`freq`、`source`、`domain`、用户词频等强旁路特征时，`context_before` 是否真的能提升候选重排。

两条路线必须分开：

- `v2-clean`：只看 `context_before`、`composing`、`candidate`。默认在线主任务不使用 `context_after`。`context_after` 只属于句中编辑、已有文本重排、离线纠错等场景。
- `production hybrid`：在 clean 结论成立后，允许加入候选原序、全局词频、领域词频、用户词频、最近上屏、用户词典、domain 等特征，但必须做 `rank only`、`context only`、`side features only`、`context + side features` 消融。

真实输入法最终一定是混合系统。禁止把“第一阶段去掉强特征”误读成“生产模型永远不能使用词频、原始 rank 或用户记忆”。这些特征是合法产品信号，只是不能用来证明模型学会了上下文。

在线评估必须分两层：

- 候选召回：target 是否进入 top10/top30/top100。召回失败不算 reranker 的错，但必须单独统计。
- 候选重排：只在召回成功的候选集合内统计 top1/top3、MRR、NDCG、误伤率和纠错率。

产品体验指标必须包含：原始 rank top1 正确时模型保持正确的比例；原始 rank top1 错误时模型纠正的比例；原始 rank top1 正确却被模型挪错的误伤比例。如果纠错收益低于误伤，reranker 不适合默认启用，即使整体 top1 看起来提升。
