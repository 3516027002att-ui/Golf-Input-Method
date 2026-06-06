# Context reranker v2 training workflow

This workflow keeps local compute as the audit/smoke/acceptance station and
uses Colab as the long GPU training runner.

## Data

Use these first-round splits only:

- `train_new_corpus.jsonl`
- `val_new_corpus_v2.jsonl`
- `test_new_corpus_v2.jsonl`

Do not use the legacy `train.jsonl` as the first main training set; its earlier
audit found rare CJK candidate-distribution pollution.

Default local data root:

```text
G:\我的云端硬盘\golf-ime-data-rebuild\clean_dataset_v3\left_context_only
```

## Local gate

Run the full split audit first:

```bat
scripts\run_context_v2_local_audit.bat
```

Expected report:

```text
reports\context_v2_new_corpus_audit.md
```

Then run the local smoke/sanity gate:

```bat
scripts\run_context_v2_local_smoke.bat
```

The smoke script copies only small heads of the new corpus into ignored
`.smoke_data/`, trains a tiny online checkpoint plus a random-label checkpoint,
runs online/no-context/shuffle/random evals, and writes a summary under:

```text
reports\context_v2_local_smoke\<timestamp>\
```

Smoke metrics are chain checks, not product-quality claims. Any red-line
findings in the generated eval JSON must be inspected before full training.

## Colab run

Use:

```text
notebooks/colab_context_reranker_v2.ipynb
```

The notebook mounts Drive, clones this branch, installs `requirements-train.txt`,
copies the new corpus to `/content/golf-ime-data/context_v2/`, runs audit, runs
smoke/sanity, and writes artifacts to:

```text
/content/drive/MyDrive/golf-ime-runs/context_reranker_v2_new_corpus/
```

Full training is guarded by:

```python
RUN_FULL_TRAIN = False
```

Flip it to `True` only after audit and smoke/sanity look sane. The full run uses
`training/context_reranker_v2.py`; do not switch back to `training/plan1_reranker.py`.

## Local acceptance after Colab

After Drive syncs the Colab checkpoint, run offline prediction locally:

```bat
python scripts\predict_context_reranker_v2.py ^
  --checkpoint "G:\...\context_reranker_v2_new_corpus\checkpoint" ^
  --context-before "我今天想去" ^
  --composing "xuexiao" ^
  --candidate "学校" ^
  --candidate "睡觉" ^
  --candidate "北京" ^
  --candidate "一个" ^
  --device cpu
```

This is an acceptance check before UI integration. Do not claim the real model
is enabled until a checkpoint is loaded and this offline path behaves well on
representative input-method scenarios.
