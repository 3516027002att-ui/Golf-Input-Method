"""
Context-aware candidate reranker v2 for Golf Input Method.

This file is intentionally audit-first.

Commands:
    python training/context_reranker_v2.py audit-data --input data.jsonl --report audit.md
    python training/context_reranker_v2.py split-data --input data.jsonl --output-dir data/processed
    python training/context_reranker_v2.py train --train data/processed/context_v2_train.jsonl --val data/processed/context_v2_val.jsonl --output-dir checkpoints/context_v2
    python training/context_reranker_v2.py eval --data data/processed/context_v2_test.jsonl --checkpoint checkpoints/context_v2

Data contract:
    Each JSONL line should describe one real candidate choice event:
        context_before: str
        context_after: str
        composing: str
        candidates: list[str] or list[{"text": str, ...}]
        target_index: int

    Compatible legacy fields:
        target: str
        candidate_features: list[dict]
        source_doc_id: str

Design choice:
    The neural model does not consume static_rank/frequency by default.
    Those strong side channels are kept for baseline and auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


torch = None
F = None
nn = None
Dataset = None
DataLoader = None
AutoModel = None
AutoTokenizer = None
get_cosine_schedule_with_warmup = None
_TRAINING_IMPORT_ERROR: Optional[BaseException] = None


def _try_import_training_deps() -> None:
    global torch, F, nn, Dataset, DataLoader
    global AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
    global _TRAINING_IMPORT_ERROR

    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
        from torch import nn as nn_module
        from torch.utils.data import DataLoader as data_loader_class
        from torch.utils.data import Dataset as dataset_class
        from transformers import AutoModel as auto_model_class
        from transformers import AutoTokenizer as auto_tokenizer_class
        from transformers import get_cosine_schedule_with_warmup as schedule_fn
    except ImportError as exc:
        _TRAINING_IMPORT_ERROR = exc
        return

    torch = torch_module
    F = functional_module
    nn = nn_module
    Dataset = dataset_class
    DataLoader = data_loader_class
    AutoModel = auto_model_class
    AutoTokenizer = auto_tokenizer_class
    get_cosine_schedule_with_warmup = schedule_fn
    _TRAINING_IMPORT_ERROR = None


def require_training_deps() -> None:
    if _TRAINING_IMPORT_ERROR is not None:
        _try_import_training_deps()
    if _TRAINING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Training/eval requires torch and transformers. Install them before running train/eval."
        ) from _TRAINING_IMPORT_ERROR


_try_import_training_deps()


class Defaults:
    encoder = os.environ.get("CONTEXT_RERANKER_ENCODER", "hfl/chinese-macbert-base")
    max_length = int(os.environ.get("CONTEXT_RERANKER_MAX_LENGTH", "192"))
    context_before_chars = int(os.environ.get("CONTEXT_RERANKER_BEFORE_CHARS", "128"))
    context_after_chars = int(os.environ.get("CONTEXT_RERANKER_AFTER_CHARS", "64"))
    train_percent = int(os.environ.get("CONTEXT_RERANKER_TRAIN_PERCENT", "90"))
    val_percent = int(os.environ.get("CONTEXT_RERANKER_VAL_PERCENT", "5"))
    batch_size = int(os.environ.get("CONTEXT_RERANKER_BATCH", "4"))
    eval_batch_size = int(os.environ.get("CONTEXT_RERANKER_EVAL_BATCH", "8"))
    epochs = int(os.environ.get("CONTEXT_RERANKER_EPOCHS", "3"))
    lr = float(os.environ.get("CONTEXT_RERANKER_LR", "2e-5"))
    weight_decay = float(os.environ.get("CONTEXT_RERANKER_WEIGHT_DECAY", "0.01"))
    warmup_ratio = float(os.environ.get("CONTEXT_RERANKER_WARMUP_RATIO", "0.06"))
    seed = int(os.environ.get("CONTEXT_RERANKER_SEED", "1337"))


@dataclass
class CandidateMeta:
    freq: Optional[float] = None
    source: str = "unknown"
    match_type: str = "unknown"
    original_rank: int = 0


@dataclass
class NormalizedSample:
    sample_id: str
    source_doc_key: str
    context_before: str
    context_after: str
    composing: str
    candidates: List[str]
    target_index: int
    candidate_meta: List[CandidateMeta]
    domain: str = "unknown"
    license_bucket: str = "unknown"
    raw: Optional[dict] = None

    @property
    def target(self) -> str:
        return self.candidates[self.target_index]


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def stable_bucket(value: str, modulo: int = 100) -> int:
    return int(stable_hash(value)[:8], 16) % modulo


def normalize_label(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    out = []
    prev_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        elif not prev_sep:
            out.append("_")
            prev_sep = True
    return "".join(out).strip("_") or default


def split_name(source_doc_key: str, train_percent: int, val_percent: int) -> str:
    bucket = stable_bucket(source_doc_key or "unknown")
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + val_percent:
        return "val"
    return "test"


def build_pair_text(
    *,
    context_before: str,
    context_after: str,
    composing: str,
    candidate: str,
    before_chars: int = Defaults.context_before_chars,
    after_chars: int = Defaults.context_after_chars,
    include_context: bool = True,
) -> str:
    before = (context_before or "")[-before_chars:] if include_context else ""
    after = (context_after or "")[:after_chars] if include_context else ""
    return (
        f"[context_before] {before}\n"
        f"[composing] {composing or ''}\n"
        f"[candidate] {candidate or ''}\n"
        f"[context_after] {after}"
    )


def _candidate_text_and_meta(item: Any, rank: int) -> Tuple[str, CandidateMeta]:
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("word") or item.get("candidate") or "").strip()
        freq = item.get("freq")
        try:
            freq_value = float(freq) if freq is not None else None
        except (TypeError, ValueError):
            freq_value = None
        return text, CandidateMeta(
            freq=freq_value,
            source=str(item.get("source", "unknown") or "unknown"),
            match_type=str(item.get("match_type", "unknown") or "unknown"),
            original_rank=int(item.get("static_rank", rank) or rank),
        )
    return str(item).strip(), CandidateMeta(original_rank=rank)


def _meta_from_legacy_feature(feature: dict, rank: int) -> CandidateMeta:
    freq = feature.get("freq")
    try:
        freq_value = float(freq) if freq is not None else None
    except (TypeError, ValueError):
        freq_value = None
    return CandidateMeta(
        freq=freq_value,
        source=str(feature.get("source", "unknown") or "unknown"),
        match_type=str(feature.get("match_type", "unknown") or "unknown"),
        original_rank=int(feature.get("static_rank", rank) or rank),
    )


def sample_signature(sample: NormalizedSample, include_context: bool = True) -> str:
    payload = {
        "context_before": sample.context_before if include_context else "",
        "context_after": sample.context_after if include_context else "",
        "composing": sample.composing,
        "candidates": sample.candidates,
        "target_index": sample.target_index,
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def parse_sample(obj: dict, line_no: int = 0) -> Tuple[Optional[NormalizedSample], List[str]]:
    warnings: List[str] = []
    context_before = str(obj.get("context_before", ""))
    context_after = str(obj.get("context_after", ""))
    composing = str(obj.get("composing", "")).strip()
    raw_candidates = obj.get("candidates") or []
    candidates: List[str] = []
    candidate_meta: List[CandidateMeta] = []
    seen = set()

    for rank, raw in enumerate(raw_candidates, start=1):
        text, meta = _candidate_text_and_meta(raw, rank)
        if not text:
            warnings.append("empty_candidate_removed")
            continue
        if text in seen:
            warnings.append("duplicate_candidate_removed")
            continue
        seen.add(text)
        candidates.append(text)
        candidate_meta.append(meta)

    legacy_features = obj.get("candidate_features") or []
    if legacy_features and len(legacy_features) == len(candidates):
        candidate_meta = [
            _meta_from_legacy_feature(feature, rank)
            for rank, feature in enumerate(legacy_features, start=1)
        ]

    target_index = obj.get("target_index")
    target = obj.get("target")
    try:
        target_index = int(target_index)
    except (TypeError, ValueError):
        target_index = -1

    if target_index < 0 or target_index >= len(candidates):
        if target in candidates:
            target_index = candidates.index(target)
            warnings.append("target_index_recovered_from_target")
        else:
            warnings.append("target_index_invalid")
            return None, warnings

    if not composing:
        warnings.append("composing_empty")
        return None, warnings
    if len(candidates) < 2:
        warnings.append("candidate_count_under_2")
        return None, warnings
    if not context_before.strip() and not context_after.strip():
        warnings.append("context_both_empty")

    source_doc_key = str(
        obj.get("source_doc_key")
        or obj.get("source_doc_id")
        or obj.get("session_id")
        or ""
    ).strip()
    if not source_doc_key:
        identity = json.dumps(
            {
                "line_no": line_no,
                "context_before": context_before,
                "context_after": context_after,
                "composing": composing,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        source_doc_key = f"unknown:{stable_hash(identity)[:16]}"
        warnings.append("source_doc_key_synthesized")

    sample_id = str(obj.get("sample_id") or "").strip()
    if not sample_id:
        identity = json.dumps(
            {
                "source_doc_key": source_doc_key,
                "context_before": context_before,
                "context_after": context_after,
                "composing": composing,
                "candidates": candidates,
                "target_index": target_index,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        sample_id = stable_hash(identity)[:20]

    while len(candidate_meta) < len(candidates):
        candidate_meta.append(CandidateMeta(original_rank=len(candidate_meta) + 1))

    normalized = NormalizedSample(
        sample_id=sample_id,
        source_doc_key=source_doc_key,
        context_before=context_before,
        context_after=context_after,
        composing=composing,
        candidates=candidates,
        target_index=target_index,
        candidate_meta=candidate_meta[: len(candidates)],
        domain=str(obj.get("domain", "unknown") or "unknown"),
        license_bucket=str(obj.get("license_bucket", "unknown") or "unknown"),
        raw=obj,
    )
    return normalized, warnings


def read_jsonl_samples(path: Path) -> Tuple[List[NormalizedSample], Counter]:
    samples: List[NormalizedSample] = []
    counters: Counter = Counter()
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            counters["read"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                counters["json_decode_error"] += 1
                continue
            sample, warnings = parse_sample(obj, line_no=line_no)
            for warning in warnings:
                counters[f"warning_{warning}"] += 1
            if sample is None:
                counters["dropped"] += 1
                continue
            samples.append(sample)
            counters["kept"] += 1
    return samples, counters


def sample_to_json(sample: NormalizedSample) -> dict:
    return {
        "sample_id": sample.sample_id,
        "source_doc_key": sample.source_doc_key,
        "context_before": sample.context_before,
        "context_after": sample.context_after,
        "composing": sample.composing,
        "candidates": sample.candidates,
        "target_index": sample.target_index,
        "candidate_meta": [
            {
                "freq": meta.freq,
                "source": meta.source,
                "match_type": meta.match_type,
                "original_rank": meta.original_rank,
            }
            for meta in sample.candidate_meta
        ],
        "domain": sample.domain,
        "license_bucket": sample.license_bucket,
    }


def write_jsonl(path: Path, samples: Iterable[NormalizedSample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(json.dumps(sample_to_json(sample), ensure_ascii=False) + "\n")
            count += 1
    return count


def topk_from_rank(target_indices: Sequence[int], candidate_counts: Sequence[int]) -> Dict[str, float]:
    total = max(len(target_indices), 1)
    out = {}
    for k in (1, 3, 5):
        hits = 0
        for idx, count in zip(target_indices, candidate_counts):
            if idx < min(k, count):
                hits += 1
        out[f"top{k}"] = hits / total
    return out


def topk_from_frequency(samples: Sequence[NormalizedSample]) -> Dict[str, float]:
    hits = {1: 0, 3: 0, 5: 0}
    total = max(len(samples), 1)
    usable = 0
    for sample in samples:
        if not any(meta.freq is not None for meta in sample.candidate_meta):
            continue
        usable += 1
        ranked = sorted(
            range(len(sample.candidates)),
            key=lambda i: (
                -float(sample.candidate_meta[i].freq or 0.0),
                sample.candidate_meta[i].original_rank or i + 1,
            ),
        )
        for k in hits:
            if sample.target_index in ranked[: min(k, len(ranked))]:
                hits[k] += 1
    if usable == 0:
        return {"usable": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0}
    return {
        "usable": usable / total,
        "top1": hits[1] / usable,
        "top3": hits[3] / usable,
        "top5": hits[5] / usable,
    }


def random_baseline(samples: Sequence[NormalizedSample]) -> Dict[str, float]:
    total = max(len(samples), 1)
    out = {}
    for k in (1, 3, 5):
        out[f"top{k}"] = sum(min(k, len(s.candidates)) / len(s.candidates) for s in samples) / total
    return out


def distribution(counter: Counter, limit: int = 20) -> List[Tuple[Any, int]]:
    return counter.most_common(limit)


def make_audit_markdown(
    *,
    title: str,
    samples_by_name: Dict[str, List[NormalizedSample]],
    counters_by_name: Dict[str, Counter],
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("本报告用于判断数据是否有资格进入上下文选词模型训练。")
    lines.append("")
    all_samples = [sample for samples in samples_by_name.values() for sample in samples]
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- kept samples: {len(all_samples)}")
    for name, samples in samples_by_name.items():
        lines.append(f"- {name}: {len(samples)}")
    lines.append("")
    lines.append("## 读取与修复/丢弃计数")
    lines.append("")
    for name, counters in counters_by_name.items():
        lines.append(f"### {name}")
        for key, value in sorted(counters.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")

    for name, samples in samples_by_name.items():
        if not samples:
            continue
        candidate_counts = [len(s.candidates) for s in samples]
        target_indices = [s.target_index for s in samples]
        source_counts = Counter(s.source_doc_key.split(":", 1)[0] for s in samples)
        domain_counts = Counter(normalize_label(s.domain) for s in samples)
        license_counts = Counter(normalize_label(s.license_bucket) for s in samples)
        target_index_counts = Counter(target_indices)
        candidate_count_counts = Counter(candidate_counts)
        lines.append(f"## Split: {name}")
        lines.append("")
        lines.append(f"- samples: {len(samples)}")
        lines.append(f"- avg candidates: {statistics.mean(candidate_counts):.2f}")
        lines.append(f"- median candidates: {statistics.median(candidate_counts):.2f}")
        lines.append(f"- min/max candidates: {min(candidate_counts)} / {max(candidate_counts)}")
        lines.append("")
        lines.append("### Baselines")
        lines.append("")
        rank_base = topk_from_rank(target_indices, candidate_counts)
        freq_base = topk_from_frequency(samples)
        rand_base = random_baseline(samples)
        lines.append(f"- original-rank: {json.dumps(rank_base, ensure_ascii=False)}")
        lines.append(f"- frequency: {json.dumps(freq_base, ensure_ascii=False)}")
        lines.append(f"- random: {json.dumps(rand_base, ensure_ascii=False)}")
        lines.append("")
        lines.append("### Distributions")
        lines.append("")
        lines.append(f"- target_index: {distribution(target_index_counts)}")
        lines.append(f"- candidate_count: {distribution(candidate_count_counts)}")
        lines.append(f"- source_prefix: {distribution(source_counts)}")
        lines.append(f"- domain: {distribution(domain_counts)}")
        lines.append(f"- license: {distribution(license_counts)}")
        lines.append("")
        if rank_base["top1"] > 0.85:
            lines.append("### Warning")
            lines.append("")
            lines.append(
                "- original-rank top1 is very high. The neural model may only learn "
                "the existing candidate order unless the data contains hard reranking cases."
            )
            lines.append("")

    if len(samples_by_name) > 1:
        lines.append("## Split leakage checks")
        lines.append("")
        doc_to_splits: Dict[str, set] = defaultdict(set)
        sig_to_splits: Dict[str, set] = defaultdict(set)
        noctx_to_splits: Dict[str, set] = defaultdict(set)
        for name, samples in samples_by_name.items():
            for sample in samples:
                doc_to_splits[sample.source_doc_key].add(name)
                sig_to_splits[sample_signature(sample, include_context=True)].add(name)
                noctx_to_splits[sample_signature(sample, include_context=False)].add(name)
        leaked_docs = {k: v for k, v in doc_to_splits.items() if len(v) > 1}
        leaked_sigs = {k: v for k, v in sig_to_splits.items() if len(v) > 1}
        leaked_noctx = {k: v for k, v in noctx_to_splits.items() if len(v) > 1}
        lines.append(f"- source_doc_key crossing splits: {len(leaked_docs)}")
        lines.append(f"- exact sample signatures crossing splits: {len(leaked_sigs)}")
        lines.append(f"- no-context signatures crossing splits: {len(leaked_noctx)}")
        if leaked_docs:
            lines.append("- sample leaked source_doc_key examples:")
            for key, splits in list(leaked_docs.items())[:10]:
                lines.append(f"  - {key}: {sorted(splits)}")
        lines.append("")

    lines.append("## 结论建议")
    lines.append("")
    lines.append("- 先确认 original-rank / frequency baseline 是否已经过高。")
    lines.append("- 再训练纯上下文 cross-encoder。")
    lines.append("- 如果纯上下文模型没有明显超过 baseline，应优先修数据，而不是堆模型。")
    lines.append("- 如果无上下文 ablation 接近正常模型，应视为任务定义或数据切分存在问题。")
    lines.append("")
    return "\n".join(lines)


def command_audit_data(args: argparse.Namespace) -> None:
    samples_by_name: Dict[str, List[NormalizedSample]] = {}
    counters_by_name: Dict[str, Counter] = {}
    if args.input:
        path = Path(args.input)
        samples, counters = read_jsonl_samples(path)
        samples_by_name[path.stem] = samples
        counters_by_name[path.stem] = counters
    else:
        for name, path_text in (("train", args.train), ("val", args.val), ("test", args.test)):
            if not path_text:
                continue
            samples, counters = read_jsonl_samples(Path(path_text))
            samples_by_name[name] = samples
            counters_by_name[name] = counters
    if not samples_by_name:
        raise ValueError("Provide --input or at least one of --train/--val/--test")
    report = make_audit_markdown(
        title="Context reranker v2 data audit",
        samples_by_name=samples_by_name,
        counters_by_name=counters_by_name,
    )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"wrote audit report: {report_path}")
    else:
        print(report)


def command_split_data(args: argparse.Namespace) -> None:
    train_percent = int(args.train_percent)
    val_percent = int(args.val_percent)
    if train_percent <= 0 or val_percent <= 0 or train_percent + val_percent >= 100:
        raise ValueError("Expected train_percent > 0, val_percent > 0, train + val < 100")
    samples, counters = read_jsonl_samples(Path(args.input))
    by_split: Dict[str, List[NormalizedSample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        by_split[split_name(sample.source_doc_key, train_percent, val_percent)].append(sample)
    output_dir = Path(args.output_dir)
    prefix = args.prefix
    paths = {name: output_dir / f"{prefix}_{name}.jsonl" for name in ("train", "val", "test")}
    for name, path in paths.items():
        count = write_jsonl(path, by_split[name])
        print(f"wrote {name}: {count} -> {path}")
    audit_report = make_audit_markdown(
        title=f"{prefix} split audit",
        samples_by_name=by_split,
        counters_by_name={"input": counters},
    )
    report_path = output_dir / f"{prefix}_audit.md"
    report_path.write_text(audit_report, encoding="utf-8")
    manifest = {
        "prefix": prefix,
        "input": str(args.input),
        "train_percent": train_percent,
        "val_percent": val_percent,
        "test_percent": 100 - train_percent - val_percent,
        "counts": {name: len(items) for name, items in by_split.items()},
        "paths": {name: str(path) for name, path in paths.items()},
        "audit_report": str(report_path),
    }
    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote audit: {report_path}")
    print(f"wrote manifest: {manifest_path}")


if Dataset is not None:
    _DatasetBase = Dataset
else:
    _DatasetBase = object


class RankingJsonlDataset(_DatasetBase):
    def __init__(self, path: str | os.PathLike[str]) -> None:
        require_training_deps()
        self.path = Path(path)
        self.samples, self.counters = read_jsonl_samples(self.path)
        if not self.samples:
            raise ValueError(f"No usable samples loaded from {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> NormalizedSample:
        return self.samples[index]


class ContextRerankerModel(nn.Module if nn is not None else object):
    def __init__(self, encoder: Any, dropout: float = 0.1) -> None:
        require_training_deps()
        super().__init__()
        self.encoder = encoder
        hidden_size = int(getattr(encoder.config, "hidden_size"))
        self.scorer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, hidden_size // 2), 1),
        )

    def forward(self, *, input_ids: Any, attention_mask: Any, token_type_ids: Optional[Any] = None) -> Any:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        pooled = outputs.last_hidden_state[:, 0]
        return self.scorer(pooled).squeeze(-1)


def make_collate_fn(tokenizer: Any, args: argparse.Namespace, *, include_context: bool = True):
    def collate(samples: List[NormalizedSample]) -> Dict[str, Any]:
        texts: List[str] = []
        group_sizes: List[int] = []
        target_indices: List[int] = []
        for sample in samples:
            group_sizes.append(len(sample.candidates))
            target_indices.append(sample.target_index)
            for candidate in sample.candidates:
                texts.append(
                    build_pair_text(
                        context_before=sample.context_before,
                        context_after=sample.context_after,
                        composing=sample.composing,
                        candidate=candidate,
                        before_chars=int(args.context_before_chars),
                        after_chars=int(args.context_after_chars),
                        include_context=include_context,
                    )
                )
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
            return_tensors="pt",
        )
        return {
            "encoded": encoded,
            "group_sizes": group_sizes,
            "target_indices": target_indices,
            "samples": samples,
        }
    return collate


def move_batch(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {**batch, "encoded": {key: value.to(device) for key, value in batch["encoded"].items()}}


def listwise_loss(scores: Any, group_sizes: Sequence[int], target_indices: Sequence[int]) -> Any:
    losses = []
    cursor = 0
    for size, target_index in zip(group_sizes, target_indices):
        group_scores = scores[cursor: cursor + size].unsqueeze(0)
        target = torch.tensor([int(target_index)], dtype=torch.long, device=scores.device)
        losses.append(F.cross_entropy(group_scores, target))
        cursor += size
    return torch.stack(losses).mean()


def score_topk(scores: Any, group_sizes: Sequence[int], target_indices: Sequence[int]) -> Dict[str, int]:
    hits = {1: 0, 3: 0, 5: 0}
    cursor = 0
    for size, target_index in zip(group_sizes, target_indices):
        group = scores[cursor: cursor + size]
        order = torch.argsort(group, descending=True).detach().cpu().tolist()
        for k in hits:
            if int(target_index) in order[: min(k, size)]:
                hits[k] += 1
        cursor += size
    return {f"top{k}": value for k, value in hits.items()}


def evaluate_model(model: Any, loader: Any, device: Any) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    hits = Counter()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            encoded = batch["encoded"]
            scores = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids"),
            )
            loss = listwise_loss(scores, batch["group_sizes"], batch["target_indices"])
            batch_size = len(batch["target_indices"])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            hits.update(score_topk(scores, batch["group_sizes"], batch["target_indices"]))
    denom = max(total_samples, 1)
    model.train()
    return {
        "loss": total_loss / denom,
        "top1": hits["top1"] / denom,
        "top3": hits["top3"] / denom,
        "top5": hits["top5"] / denom,
        "samples": float(total_samples),
    }


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    head_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("encoder.")
    }
    torch.save(head_state, output_dir / "context_reranker_head.pt")
    (output_dir / "context_reranker_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(checkpoint_dir: Path, device: Any) -> Tuple[Any, Any, dict]:
    config_path = checkpoint_dir / "context_reranker_config.json"
    head_path = checkpoint_dir / "context_reranker_head.pt"
    if not config_path.is_file() or not head_path.is_file():
        raise FileNotFoundError(
            f"Expected context_reranker_config.json and context_reranker_head.pt in {checkpoint_dir}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    encoder = AutoModel.from_pretrained(checkpoint_dir)
    model = ContextRerankerModel(encoder, dropout=float(config.get("dropout", 0.0)))
    missing, unexpected = model.load_state_dict(torch.load(head_path, map_location="cpu"), strict=False)
    missing_head = [name for name in missing if not name.startswith("encoder.")]
    unexpected_head = [name for name in unexpected if not name.startswith("encoder.")]
    if missing_head or unexpected_head:
        raise RuntimeError(f"head mismatch missing={missing_head} unexpected={unexpected_head}")
    model.to(device)
    model.eval()
    return model, tokenizer, config


def set_seed(seed: int) -> None:
    random.seed(seed)
    require_training_deps()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def command_train(args: argparse.Namespace) -> None:
    require_training_deps()
    set_seed(int(args.seed))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder)
    model = ContextRerankerModel(encoder, dropout=float(args.dropout)).to(device)
    train_ds = RankingJsonlDataset(args.train)
    val_ds = RankingJsonlDataset(args.val)
    collate = make_collate_fn(tokenizer, args, include_context=not args.no_context)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    train_rank_base = topk_from_rank(
        [sample.target_index for sample in train_ds.samples],
        [len(sample.candidates) for sample in train_ds.samples],
    )
    val_rank_base = topk_from_rank(
        [sample.target_index for sample in val_ds.samples],
        [len(sample.candidates) for sample in val_ds.samples],
    )
    print(f"train original-rank baseline: {json.dumps(train_rank_base)}")
    print(f"val original-rank baseline: {json.dumps(val_rank_base)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(1, len(train_loader) * int(args.epochs))
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    best_top1 = -1.0
    output_dir = Path(args.output_dir)
    config = {
        "encoder": args.encoder,
        "max_length": int(args.max_length),
        "context_before_chars": int(args.context_before_chars),
        "context_after_chars": int(args.context_after_chars),
        "dropout": float(args.dropout),
        "no_context": bool(args.no_context),
        "data_contract": "context_reranker_v2",
        "side_features_used_by_model": False,
    }
    global_step = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        start_time = time.time()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            encoded = batch["encoded"]
            scores = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids"),
            )
            loss = listwise_loss(scores, batch["group_sizes"], batch["target_indices"])
            loss.backward()
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            batch_size = len(batch["target_indices"])
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            global_step += 1
            if global_step % int(args.log_every) == 0:
                print(
                    f"epoch={epoch} step={global_step} train_loss={loss_sum / max(sample_count, 1):.6f}",
                    flush=True,
                )
        val_metrics = evaluate_model(model, val_loader, device)
        elapsed = time.time() - start_time
        print(
            f"epoch={epoch} elapsed={elapsed:.1f}s train_loss={loss_sum / max(sample_count, 1):.6f} "
            f"val={json.dumps(val_metrics, ensure_ascii=False)}",
            flush=True,
        )
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            save_checkpoint(model, tokenizer, output_dir, {**config, "best_val": val_metrics})
            print(f"saved best checkpoint to {output_dir}")
    print(f"best_val_top1={best_top1:.6f}")


def command_eval(args: argparse.Namespace) -> None:
    require_training_deps()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_dir = Path(args.checkpoint)
    model, tokenizer, config = load_checkpoint(checkpoint_dir, device)
    eval_args = argparse.Namespace(
        max_length=int(config.get("max_length", Defaults.max_length)),
        context_before_chars=int(config.get("context_before_chars", Defaults.context_before_chars)),
        context_after_chars=int(config.get("context_after_chars", Defaults.context_after_chars)),
    )
    dataset = RankingJsonlDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, eval_args, include_context=not args.no_context),
        num_workers=0,
    )
    metrics = evaluate_model(model, loader, device)
    rank_base = topk_from_rank(
        [sample.target_index for sample in dataset.samples],
        [len(sample.candidates) for sample in dataset.samples],
    )
    freq_base = topk_from_frequency(dataset.samples)
    print(json.dumps({
        "metrics": metrics,
        "original_rank_baseline": rank_base,
        "frequency_baseline": freq_base,
        "no_context_eval": bool(args.no_context),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit-first context reranker v2")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-data", help="Audit one JSONL file or train/val/test splits")
    audit.add_argument("--input")
    audit.add_argument("--train")
    audit.add_argument("--val")
    audit.add_argument("--test")
    audit.add_argument("--report")
    audit.set_defaults(func=command_audit_data)
    split = sub.add_parser("split-data", help="Normalize and split JSONL by source_doc_key")
    split.add_argument("--input", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--prefix", default="context_v2")
    split.add_argument("--train-percent", type=int, default=Defaults.train_percent)
    split.add_argument("--val-percent", type=int, default=Defaults.val_percent)
    split.set_defaults(func=command_split_data)
    train = sub.add_parser("train", help="Train a pure context cross-encoder reranker")
    train.add_argument("--train", required=True)
    train.add_argument("--val", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--encoder", default=Defaults.encoder)
    train.add_argument("--max-length", type=int, default=Defaults.max_length)
    train.add_argument("--context-before-chars", type=int, default=Defaults.context_before_chars)
    train.add_argument("--context-after-chars", type=int, default=Defaults.context_after_chars)
    train.add_argument("--batch-size", type=int, default=Defaults.batch_size)
    train.add_argument("--eval-batch-size", type=int, default=Defaults.eval_batch_size)
    train.add_argument("--epochs", type=int, default=Defaults.epochs)
    train.add_argument("--lr", type=float, default=Defaults.lr)
    train.add_argument("--weight-decay", type=float, default=Defaults.weight_decay)
    train.add_argument("--warmup-ratio", type=float, default=Defaults.warmup_ratio)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--log-every", type=int, default=50)
    train.add_argument("--seed", type=int, default=Defaults.seed)
    train.add_argument("--device")
    train.add_argument("--no-context", action="store_true", help="Ablation: remove context from model input")
    train.set_defaults(func=command_train)
    ev = sub.add_parser("eval", help="Evaluate checkpoint and print baseline comparisons")
    ev.add_argument("--data", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--batch-size", type=int, default=Defaults.eval_batch_size)
    ev.add_argument("--device")
    ev.add_argument("--no-context", action="store_true")
    ev.set_defaults(func=command_eval)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
