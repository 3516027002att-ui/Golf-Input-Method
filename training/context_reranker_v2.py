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


def _try_import_training_deps(*, include_transformers: bool = True) -> None:
    global torch, F, nn, Dataset, DataLoader
    global AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
    global _TRAINING_IMPORT_ERROR

    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
        from torch import nn as nn_module
        from torch.utils.data import DataLoader as data_loader_class
        from torch.utils.data import Dataset as dataset_class
    except ImportError as exc:
        _TRAINING_IMPORT_ERROR = exc
        return

    torch = torch_module
    F = functional_module
    nn = nn_module
    Dataset = dataset_class
    DataLoader = data_loader_class
    if not include_transformers:
        _TRAINING_IMPORT_ERROR = None
        return

    try:
        from transformers import AutoModel as auto_model_class
        from transformers import AutoTokenizer as auto_tokenizer_class
        from transformers import get_cosine_schedule_with_warmup as schedule_fn
    except ImportError as exc:
        _TRAINING_IMPORT_ERROR = exc
        return

    AutoModel = auto_model_class
    AutoTokenizer = auto_tokenizer_class
    get_cosine_schedule_with_warmup = schedule_fn
    _TRAINING_IMPORT_ERROR = None


def require_training_deps() -> None:
    if _TRAINING_IMPORT_ERROR is not None or AutoModel is None or AutoTokenizer is None:
        _try_import_training_deps(include_transformers=True)
    if _TRAINING_IMPORT_ERROR is not None or AutoModel is None or AutoTokenizer is None:
        raise RuntimeError(
            "Training/eval requires torch and transformers. Install them before running train/eval."
        ) from _TRAINING_IMPORT_ERROR


_try_import_training_deps(include_transformers=False)


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


CONTEXT_MODES = {"online", "edit", "none"}
CONTEXT_PERTURBS = {"none", "shuffle", "empty"}
CANDIDATE_ORDERS = {"original", "shuffle"}
LABEL_MODES = {"normal", "random"}

HARD_NEGATIVE_MATCH_HINTS = {
    "same_pinyin",
    "exact_pinyin",
    "full_pinyin",
    "pinyin",
    "same_abbr",
    "abbr",
    "short_pinyin",
    "fuzzy",
    "domain",
    "context",
    "confusable",
}


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


def validate_choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def resolve_context_mode(args: argparse.Namespace, *, include_context: bool = True) -> str:
    if not include_context or bool(getattr(args, "no_context", False)):
        return "none"
    mode = str(getattr(args, "context_mode", "online") or "online")
    return validate_choice(mode, CONTEXT_MODES, "context_mode")


def deterministic_shuffle(items: Sequence[Any], salt: str) -> List[Any]:
    out = list(items)
    rng = random.Random(int(stable_hash(salt)[:16], 16))
    rng.shuffle(out)
    return out


def perturb_context(text: str, *, perturb: str, salt: str) -> str:
    text = text or ""
    perturb = validate_choice(perturb, CONTEXT_PERTURBS, "context_perturb")
    if perturb == "none":
        return text
    if perturb == "empty":
        return ""
    return "".join(deterministic_shuffle(list(text), salt))


def build_pair_text(
    *,
    context_before: str,
    context_after: str,
    composing: str,
    candidate: str,
    before_chars: int = Defaults.context_before_chars,
    after_chars: int = Defaults.context_after_chars,
    include_context: bool = True,
    context_mode: str = "online",
    context_perturb: str = "none",
    sample_id: str = "",
) -> str:
    if not include_context:
        context_mode = "none"
    context_mode = validate_choice(context_mode, CONTEXT_MODES, "context_mode")
    context_perturb = validate_choice(context_perturb, CONTEXT_PERTURBS, "context_perturb")
    if context_mode == "none":
        before = ""
        after = ""
    else:
        before = (context_before or "")[-before_chars:]
        after = (context_after or "")[:after_chars] if context_mode == "edit" else ""
        before = perturb_context(before, perturb=context_perturb, salt=f"{sample_id}:before")
        after = perturb_context(after, perturb=context_perturb, salt=f"{sample_id}:after")
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
        original_rank=int(feature.get("static_rank", feature.get("original_rank", rank)) or rank),
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

    legacy_features = obj.get("candidate_meta") or obj.get("candidate_features") or []
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
    with path.open("r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            counters["physical_lines"] += 1
            if not line.strip():
                counters["blank_lines"] += 1
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


def topk_from_static_rank(samples: Sequence[NormalizedSample]) -> Dict[str, float]:
    hits = {1: 0, 3: 0, 5: 0}
    total = max(len(samples), 1)
    usable = 0
    for sample in samples:
        usable += 1
        ranked = sorted(
            range(len(sample.candidates)),
            key=lambda i: (
                sample.candidate_meta[i].original_rank or i + 1,
                i,
            ),
        )
        for k in hits:
            if sample.target_index in ranked[: min(k, len(ranked))]:
                hits[k] += 1
    return {
        "usable": usable / total if samples else 0.0,
        "top1": hits[1] / max(usable, 1),
        "top3": hits[3] / max(usable, 1),
        "top5": hits[5] / max(usable, 1),
    }


def random_baseline(samples: Sequence[NormalizedSample]) -> Dict[str, float]:
    total = max(len(samples), 1)
    out = {}
    for k in (1, 3, 5):
        out[f"top{k}"] = sum(min(k, len(s.candidates)) / len(s.candidates) for s in samples) / total
    return out


def recall_by_candidate_position(samples: Sequence[NormalizedSample]) -> Dict[str, float]:
    total = max(len(samples), 1)
    out = {}
    for k in (10, 30, 100):
        hits = sum(1 for sample in samples if sample.target_index < min(k, len(sample.candidates)))
        out[f"recall@{k}"] = hits / total
    return out


def target_text(sample: NormalizedSample) -> str:
    return sample.candidates[sample.target_index]


def context_suffix(sample: NormalizedSample, chars: int = 32) -> str:
    return (sample.context_before or "")[-chars:]


def candidates_set_key(sample: NormalizedSample) -> str:
    return "\u241f".join(sorted(sample.candidates))


def normalized_text(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if not ch.isspace())


def signature_raw_input_target(sample: NormalizedSample) -> str:
    return stable_hash(json.dumps([sample.composing, target_text(sample)], ensure_ascii=False))


def signature_context_suffix_raw_input_target(sample: NormalizedSample) -> str:
    return stable_hash(
        json.dumps([context_suffix(sample, 32), sample.composing, target_text(sample)], ensure_ascii=False)
    )


def signature_candidates_set_target(sample: NormalizedSample) -> str:
    return stable_hash(json.dumps([candidates_set_key(sample), target_text(sample)], ensure_ascii=False))


def signature_context_window_target(sample: NormalizedSample) -> str:
    payload = [
        (sample.context_before or "")[-128:],
        target_text(sample),
        (sample.context_after or "")[:64],
    ]
    return stable_hash(json.dumps(payload, ensure_ascii=False))


def signature_near_context_window(sample: NormalizedSample) -> str:
    payload = normalized_text(
        f"{(sample.context_before or '')[-160:]}|{target_text(sample)}|{(sample.context_after or '')[:96]}"
    )
    return stable_hash(payload)


def build_majority_target_table(
    train_samples: Sequence[NormalizedSample],
    key_fn: Any,
) -> Dict[str, str]:
    buckets: Dict[str, Counter] = defaultdict(Counter)
    for sample in train_samples:
        key = key_fn(sample)
        if key:
            buckets[key][target_text(sample)] += 1
    return {
        key: counts.most_common(1)[0][0]
        for key, counts in buckets.items()
        if counts
    }


def memory_baseline(
    train_samples: Sequence[NormalizedSample],
    eval_samples: Sequence[NormalizedSample],
    key_fn: Any,
) -> Dict[str, float]:
    table = build_majority_target_table(train_samples, key_fn)
    total = max(len(eval_samples), 1)
    covered = 0
    hits = 0
    for sample in eval_samples:
        key = key_fn(sample)
        if key not in table:
            continue
        covered += 1
        if table[key] == target_text(sample):
            hits += 1
    return {
        "train_keys": float(len(table)),
        "eval_samples": float(len(eval_samples)),
        "covered": float(covered),
        "coverage": covered / total,
        "hit_on_covered": hits / max(covered, 1),
        "top1_all": hits / total,
    }


def cross_split_overlap_counts(
    samples_by_name: Dict[str, List[NormalizedSample]],
    signature_fn: Any,
) -> Dict[str, Any]:
    sig_to_splits: Dict[str, set[str]] = defaultdict(set)
    sig_to_examples: Dict[str, List[str]] = defaultdict(list)
    for name, samples in samples_by_name.items():
        for sample in samples:
            sig = signature_fn(sample)
            sig_to_splits[sig].add(name)
            if len(sig_to_examples[sig]) < 3:
                sig_to_examples[sig].append(sample.sample_id)
    leaked = {sig: splits for sig, splits in sig_to_splits.items() if len(splits) > 1}
    return {
        "count": len(leaked),
        "examples": [
            {"signature": sig, "splits": sorted(splits), "sample_ids": sig_to_examples.get(sig, [])}
            for sig, splits in list(leaked.items())[:10]
        ],
    }


def train_eval_overlap_rate(
    train_samples: Sequence[NormalizedSample],
    eval_samples: Sequence[NormalizedSample],
    signature_fn: Any,
) -> float:
    train_keys = {signature_fn(sample) for sample in train_samples}
    if not eval_samples:
        return 0.0
    return sum(1 for sample in eval_samples if signature_fn(sample) in train_keys) / len(eval_samples)


def classify_hard_negative(meta: CandidateMeta, target_meta: CandidateMeta) -> bool:
    match_type = normalize_label(meta.match_type)
    source = normalize_label(meta.source)
    if any(hint in match_type for hint in HARD_NEGATIVE_MATCH_HINTS):
        return True
    if source in {"user_dict", "domain_dict", "recent_user", "global_high_freq"}:
        return True
    if meta.freq is not None and target_meta.freq is not None and meta.freq >= target_meta.freq:
        return True
    return False


def candidate_generation_audit(samples: Sequence[NormalizedSample]) -> Dict[str, Any]:
    target_positions = Counter()
    target_match_types = Counter()
    negative_match_types = Counter()
    negative_sources = Counter()
    total_negative_candidates = 0
    hard_negative_candidates = 0
    samples_without_hard_negative = 0
    samples_with_high_freq_conflict = 0
    samples_target_unique_known_match = 0
    candidate_count_under_3 = 0

    for sample in samples:
        target_positions[sample.target_index] += 1
        if len(sample.candidates) < 3:
            candidate_count_under_3 += 1
        target_meta = sample.candidate_meta[sample.target_index]
        target_match = normalize_label(target_meta.match_type)
        target_match_types[target_match] += 1
        known_match_indices = [
            i for i, meta in enumerate(sample.candidate_meta)
            if normalize_label(meta.match_type) not in {"unknown", "none", ""}
        ]
        if target_match not in {"unknown", "none", ""} and known_match_indices == [sample.target_index]:
            samples_target_unique_known_match += 1

        hard_in_sample = 0
        high_freq_conflict = False
        for i, meta in enumerate(sample.candidate_meta):
            if i == sample.target_index:
                continue
            total_negative_candidates += 1
            negative_match_types[normalize_label(meta.match_type)] += 1
            negative_sources[normalize_label(meta.source)] += 1
            if classify_hard_negative(meta, target_meta):
                hard_negative_candidates += 1
                hard_in_sample += 1
            if meta.freq is not None and target_meta.freq is not None and meta.freq >= target_meta.freq:
                high_freq_conflict = True
        if hard_in_sample == 0:
            samples_without_hard_negative += 1
        if high_freq_conflict:
            samples_with_high_freq_conflict += 1

    total = max(len(samples), 1)
    return {
        "target_position_distribution": distribution(target_positions),
        "target_first_rate": target_positions[0] / total,
        "candidate_count_under_3_rate": candidate_count_under_3 / total,
        "target_match_type_distribution": distribution(target_match_types),
        "negative_match_type_distribution": distribution(negative_match_types),
        "negative_source_distribution": distribution(negative_sources),
        "hard_negative_candidate_ratio": hard_negative_candidates / max(total_negative_candidates, 1),
        "samples_without_hard_negative_rate": samples_without_hard_negative / total,
        "target_unique_known_match_rate": samples_target_unique_known_match / total,
        "high_freq_conflict_sample_rate": samples_with_high_freq_conflict / total,
    }


def split_memory_baselines(samples_by_name: Dict[str, List[NormalizedSample]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    train_samples = samples_by_name.get("train", [])
    if not train_samples:
        return {}
    key_fns = {
        "raw_input_to_target": lambda sample: sample.composing,
        "context_suffix32_raw_input_to_target": lambda sample: f"{context_suffix(sample, 32)}\u241f{sample.composing}",
        "candidates_set_to_target": candidates_set_key,
    }
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split_name_value, samples in samples_by_name.items():
        if split_name_value == "train":
            continue
        out[split_name_value] = {
            key_name: memory_baseline(train_samples, samples, key_fn)
            for key_name, key_fn in key_fns.items()
        }
    return out


def audit_red_lines(
    samples_by_name: Dict[str, List[NormalizedSample]],
    overlap_summary: Dict[str, Dict[str, Any]],
    memory_summary: Dict[str, Dict[str, Dict[str, float]]],
) -> List[str]:
    findings: List[str] = []
    for name, samples in samples_by_name.items():
        if not samples:
            continue
        candidate0 = topk_from_rank(
            [sample.target_index for sample in samples],
            [len(sample.candidates) for sample in samples],
        )["top1"]
        if candidate0 > 0.95:
            findings.append(f"{name}: candidates[0] baseline top1 {candidate0:.4f} > 0.95")

    if overlap_summary.get("source_doc_key", {}).get("count", 0) > 0:
        findings.append("source_doc_key crosses splits; split must be rebuilt")
    if overlap_summary.get("exact_sample", {}).get("count", 0) > 0:
        findings.append("exact sample signatures cross splits; split must be rebuilt")

    train_samples = samples_by_name.get("train", [])
    if train_samples:
        for split_name_value, samples in samples_by_name.items():
            if split_name_value == "train" or not samples:
                continue
            raw_target_overlap = train_eval_overlap_rate(train_samples, samples, signature_raw_input_target)
            if raw_target_overlap > 0.30:
                findings.append(
                    f"train-{split_name_value}: raw_input+target overlap {raw_target_overlap:.4f} > 0.30"
                )
            suffix_memory = memory_summary.get(split_name_value, {}).get(
                "context_suffix32_raw_input_to_target", {}
            )
            if float(suffix_memory.get("top1_all", 0.0)) > 0.90:
                findings.append(
                    f"train-{split_name_value}: context_suffix32+raw_input memory "
                    f"top1_all {suffix_memory.get('top1_all', 0.0):.4f} > 0.90"
                )
    return findings


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
    overlap_summary: Dict[str, Dict[str, Any]] = {}
    memory_summary = split_memory_baselines(samples_by_name)
    if len(samples_by_name) > 1:
        overlap_summary = {
            "source_doc_key": cross_split_overlap_counts(samples_by_name, lambda sample: sample.source_doc_key),
            "exact_sample": cross_split_overlap_counts(
                samples_by_name, lambda sample: sample_signature(sample, include_context=True)
            ),
            "no_context_sample": cross_split_overlap_counts(
                samples_by_name, lambda sample: sample_signature(sample, include_context=False)
            ),
            "raw_input_target": cross_split_overlap_counts(samples_by_name, signature_raw_input_target),
            "context_suffix32_raw_input_target": cross_split_overlap_counts(
                samples_by_name, signature_context_suffix_raw_input_target
            ),
            "candidates_set_target": cross_split_overlap_counts(samples_by_name, signature_candidates_set_target),
            "context_window_target": cross_split_overlap_counts(samples_by_name, signature_context_window_target),
            "near_context_window": cross_split_overlap_counts(samples_by_name, signature_near_context_window),
        }
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
        static_rank_base = topk_from_static_rank(samples)
        recall_base = recall_by_candidate_position(samples)
        lines.append(f"- candidates[0]: {json.dumps(rank_base, ensure_ascii=False)}")
        lines.append(f"- static-rank: {json.dumps(static_rank_base, ensure_ascii=False)}")
        lines.append(f"- frequency: {json.dumps(freq_base, ensure_ascii=False)}")
        lines.append(f"- random: {json.dumps(rand_base, ensure_ascii=False)}")
        lines.append(f"- candidate recall buckets: {json.dumps(recall_base, ensure_ascii=False)}")
        lines.append("")
        lines.append("### Candidate generator audit")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(candidate_generation_audit(samples), ensure_ascii=False, indent=2))
        lines.append("```")
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
        for key, summary in overlap_summary.items():
            lines.append(f"- {key} crossing splits: {summary['count']}")
            if summary["examples"]:
                lines.append(f"  examples: {json.dumps(summary['examples'], ensure_ascii=False)}")
        lines.append("")

    lines.append("## 结论建议")
    lines.append("")
    lines.append("- 先确认 original-rank / frequency baseline 是否已经过高。")
    lines.append("- 再训练纯上下文 cross-encoder。")
    lines.append("- 如果纯上下文模型没有明显超过 baseline，应优先修数据，而不是堆模型。")
    lines.append("- 如果无上下文 ablation 接近正常模型，应视为任务定义或数据切分存在问题。")
    lines.append("")
    if memory_summary:
        lines.append("## Train-to-eval memorization baselines")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(memory_summary, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    red_lines = audit_red_lines(samples_by_name, overlap_summary, memory_summary)
    lines.append("## Red-line findings")
    lines.append("")
    if red_lines:
        for finding in red_lines:
            lines.append(f"- {finding}")
    else:
        lines.append("- No audit red lines triggered by static data checks.")
    lines.append("")
    return "\n".join(lines)


def command_audit_data(args: argparse.Namespace) -> None:
    samples_by_name: Dict[str, List[NormalizedSample]] = {}
    counters_by_name: Dict[str, Counter] = {}
    if getattr(args, "input", None):
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


def candidate_order_indices(sample: NormalizedSample, candidate_order: str) -> List[int]:
    candidate_order = validate_choice(candidate_order, CANDIDATE_ORDERS, "candidate_order")
    order = list(range(len(sample.candidates)))
    if candidate_order == "shuffle":
        order = deterministic_shuffle(order, f"{sample.sample_id}:candidate_order")
    return order


def target_index_for_order(sample: NormalizedSample, order: Sequence[int], label_mode: str) -> int:
    label_mode = validate_choice(label_mode, LABEL_MODES, "label_mode")
    if label_mode == "random":
        return stable_bucket(f"{sample.sample_id}:random_label", len(order))
    return list(order).index(sample.target_index)


def make_collate_fn(tokenizer: Any, args: argparse.Namespace, *, include_context: bool = True):
    def collate(samples: List[NormalizedSample]) -> Dict[str, Any]:
        texts: List[str] = []
        group_sizes: List[int] = []
        target_indices: List[int] = []
        candidate_orders: List[List[int]] = []
        context_mode = resolve_context_mode(args, include_context=include_context)
        context_perturb = validate_choice(
            str(getattr(args, "context_perturb", "none") or "none"),
            CONTEXT_PERTURBS,
            "context_perturb",
        )
        candidate_order = validate_choice(
            str(getattr(args, "candidate_order", "original") or "original"),
            CANDIDATE_ORDERS,
            "candidate_order",
        )
        label_mode = validate_choice(
            str(getattr(args, "label_mode", "normal") or "normal"),
            LABEL_MODES,
            "label_mode",
        )
        for sample in samples:
            order = candidate_order_indices(sample, candidate_order)
            candidate_orders.append(order)
            group_sizes.append(len(order))
            target_indices.append(target_index_for_order(sample, order, label_mode))
            for candidate_index in order:
                candidate = sample.candidates[candidate_index]
                texts.append(
                    build_pair_text(
                        context_before=sample.context_before,
                        context_after=sample.context_after,
                        composing=sample.composing,
                        candidate=candidate,
                        before_chars=int(args.context_before_chars),
                        after_chars=int(args.context_after_chars),
                        include_context=include_context,
                        context_mode=context_mode,
                        context_perturb=context_perturb,
                        sample_id=sample.sample_id,
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
            "candidate_orders": candidate_orders,
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


def score_group_metrics(
    scores: Any,
    group_sizes: Sequence[int],
    target_indices: Sequence[int],
    candidate_orders: Sequence[Sequence[int]],
    samples: Sequence[NormalizedSample],
) -> Tuple[Dict[str, int], Counter]:
    hits = {1: 0, 3: 0, 5: 0}
    product = Counter()
    cursor = 0
    for sample, size, target_index, order in zip(samples, group_sizes, target_indices, candidate_orders):
        group = scores[cursor: cursor + size]
        ranked_positions = torch.argsort(group, descending=True).detach().cpu().tolist()
        for k in hits:
            if int(target_index) in ranked_positions[: min(k, size)]:
                hits[k] += 1

        predicted_position = int(ranked_positions[0]) if ranked_positions else 0
        predicted_original_index = int(order[predicted_position]) if order else 0
        original_top1_correct = sample.target_index == 0
        model_actual_correct = predicted_original_index == sample.target_index
        if original_top1_correct:
            product["original_top1_correct_total"] += 1
            if model_actual_correct:
                product["keep_when_original_top1_correct"] += 1
            else:
                product["harm_when_original_top1_correct"] += 1
        else:
            product["original_top1_wrong_total"] += 1
            if model_actual_correct:
                product["fix_when_original_top1_wrong"] += 1
        cursor += size
    return {f"top{k}": value for k, value in hits.items()}, product


def finalize_product_metrics(product: Counter) -> Dict[str, Any]:
    correct_total = int(product["original_top1_correct_total"])
    wrong_total = int(product["original_top1_wrong_total"])
    keep = int(product["keep_when_original_top1_correct"])
    harm = int(product["harm_when_original_top1_correct"])
    fix = int(product["fix_when_original_top1_wrong"])
    keep_rate = keep / max(correct_total, 1)
    harm_rate = harm / max(correct_total, 1)
    fix_rate = fix / max(wrong_total, 1)
    recommendation = "reject_default_enable" if harm_rate > fix_rate else "no_product_harm_red_line"
    return {
        "original_top1_correct_total": correct_total,
        "original_top1_wrong_total": wrong_total,
        "keep_when_original_top1_correct": keep_rate,
        "fix_when_original_top1_wrong": fix_rate,
        "harm_when_original_top1_correct": harm_rate,
        "keep_count": keep,
        "fix_count": fix,
        "harm_count": harm,
        "default_enable_recommendation": recommendation,
    }


def evaluate_model(model: Any, loader: Any, device: Any) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    hits = Counter()
    product = Counter()
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
            batch_hits, batch_product = score_group_metrics(
                scores,
                batch["group_sizes"],
                batch["target_indices"],
                batch["candidate_orders"],
                batch["samples"],
            )
            hits.update(batch_hits)
            product.update(batch_product)
    denom = max(total_samples, 1)
    model.train()
    metrics = {
        "loss": total_loss / denom,
        "top1": hits["top1"] / denom,
        "top3": hits["top3"] / denom,
        "top5": hits["top5"] / denom,
        "samples": float(total_samples),
    }
    metrics.update(finalize_product_metrics(product))
    return metrics


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


def make_runtime_eval_args(config: dict, args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = {
        "max_length": int(config.get("max_length", Defaults.max_length)),
        "context_before_chars": int(config.get("context_before_chars", Defaults.context_before_chars)),
        "context_after_chars": int(config.get("context_after_chars", Defaults.context_after_chars)),
        "context_mode": str(getattr(args, "context_mode", "online") or "online"),
        "context_perturb": str(getattr(args, "context_perturb", "none") or "none"),
        "candidate_order": str(getattr(args, "candidate_order", "original") or "original"),
        "label_mode": "normal",
        "no_context": bool(getattr(args, "no_context", False)),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def evaluate_variant(
    *,
    model: Any,
    tokenizer: Any,
    dataset: RankingJsonlDataset,
    config: dict,
    args: argparse.Namespace,
    device: Any,
    context_mode: str,
    context_perturb: str = "none",
    candidate_order: str = "original",
) -> Dict[str, Any]:
    eval_args = make_runtime_eval_args(
        config,
        args,
        context_mode=context_mode,
        context_perturb=context_perturb,
        candidate_order=candidate_order,
        no_context=(context_mode == "none"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, eval_args, include_context=context_mode != "none"),
        num_workers=0,
    )
    return evaluate_model(model, loader, device)


def eval_red_lines(
    *,
    sanity_metrics: Dict[str, Dict[str, Any]],
    candidate0_baseline: Dict[str, float],
) -> List[str]:
    findings: List[str] = []
    if candidate0_baseline.get("top1", 0.0) > 0.95:
        findings.append(
            f"candidates[0] baseline top1 {candidate0_baseline.get('top1', 0.0):.4f} > 0.95"
        )
    online = sanity_metrics.get("online_context", {})
    no_context = sanity_metrics.get("no_context", {})
    if online and no_context:
        delta = float(online.get("top1", 0.0)) - float(no_context.get("top1", 0.0))
        if abs(delta) < 0.01:
            findings.append(f"no-context top1 is within 1% of online-context top1 (delta={delta:.4f})")
    primary = sanity_metrics.get("primary", online)
    if primary:
        harm = float(primary.get("harm_when_original_top1_correct", 0.0))
        fix = float(primary.get("fix_when_original_top1_wrong", 0.0))
        if harm > fix:
            findings.append(f"product harm rate {harm:.4f} is higher than fix rate {fix:.4f}")
    return findings


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
    if bool(args.no_context):
        args.context_mode = "none"
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
    print(
        "train sanity mode: "
        f"context_mode={resolve_context_mode(args, include_context=not args.no_context)} "
        f"context_perturb={args.context_perturb} candidate_order={args.candidate_order} "
        f"label_mode={args.label_mode}"
    )
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
        "context_mode": resolve_context_mode(args, include_context=not args.no_context),
        "context_perturb": str(args.context_perturb),
        "candidate_order": str(args.candidate_order),
        "label_mode": str(args.label_mode),
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
    if bool(args.no_context):
        args.context_mode = "none"
    dataset = RankingJsonlDataset(args.data)
    primary_metrics = evaluate_variant(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        args=args,
        device=device,
        context_mode=str(args.context_mode),
        context_perturb=str(args.context_perturb),
        candidate_order=str(args.candidate_order),
    )
    sanity_metrics = {
        "primary": primary_metrics,
        "online_context": evaluate_variant(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            args=args,
            device=device,
            context_mode="online",
        ),
        "edit_context": evaluate_variant(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            args=args,
            device=device,
            context_mode="edit",
        ),
        "no_context": evaluate_variant(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            args=args,
            device=device,
            context_mode="none",
        ),
        "shuffled_context": evaluate_variant(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            args=args,
            device=device,
            context_mode="online",
            context_perturb="shuffle",
        ),
        "shuffled_candidates": evaluate_variant(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            args=args,
            device=device,
            context_mode="online",
            candidate_order="shuffle",
        ),
    }
    rank_base = topk_from_rank(
        [sample.target_index for sample in dataset.samples],
        [len(sample.candidates) for sample in dataset.samples],
    )
    static_rank_base = topk_from_static_rank(dataset.samples)
    freq_base = topk_from_frequency(dataset.samples)
    rand_base = random_baseline(dataset.samples)
    recall_base = recall_by_candidate_position(dataset.samples)
    print(json.dumps({
        "selected_eval": {
            "context_mode": args.context_mode,
            "context_perturb": args.context_perturb,
            "candidate_order": args.candidate_order,
        },
        "metrics": primary_metrics,
        "sanity_metrics": sanity_metrics,
        "dummy_first_baseline": rank_base,
        "static_rank_baseline": static_rank_base,
        "frequency_baseline": freq_base,
        "random_baseline": rand_base,
        "candidate_recall": recall_base,
        "candidate_generation_audit": candidate_generation_audit(dataset.samples),
        "red_line_findings": eval_red_lines(
            sanity_metrics=sanity_metrics,
            candidate0_baseline=rank_base,
        ),
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
    audit_splits = sub.add_parser("audit-splits", help="Audit train/val/test JSONL splits")
    audit_splits.add_argument("--train", required=True)
    audit_splits.add_argument("--val", required=True)
    audit_splits.add_argument("--test")
    audit_splits.add_argument("--report")
    audit_splits.set_defaults(func=command_audit_data)
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
    train.add_argument("--context-mode", choices=sorted(CONTEXT_MODES), default="online")
    train.add_argument("--context-perturb", choices=sorted(CONTEXT_PERTURBS), default="none")
    train.add_argument("--candidate-order", choices=sorted(CANDIDATE_ORDERS), default="original")
    train.add_argument("--label-mode", choices=sorted(LABEL_MODES), default="normal")
    train.add_argument("--no-context", action="store_true", help="Backward-compatible alias for --context-mode none")
    train.set_defaults(func=command_train)
    ev = sub.add_parser("eval", help="Evaluate checkpoint and print baseline comparisons")
    ev.add_argument("--data", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--batch-size", type=int, default=Defaults.eval_batch_size)
    ev.add_argument("--device")
    ev.add_argument("--context-mode", choices=sorted(CONTEXT_MODES), default="online")
    ev.add_argument("--context-perturb", choices=sorted(CONTEXT_PERTURBS), default="none")
    ev.add_argument("--candidate-order", choices=sorted(CANDIDATE_ORDERS), default="original")
    ev.add_argument("--no-context", action="store_true")
    ev.set_defaults(func=command_eval)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
