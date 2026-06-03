"""
Plan1 candidate reranker for golf IME.

This file is the single training entry for the first real candidate reranking
model. It intentionally follows the official baseline style: one file, explicit
sections, minimal hidden machinery, and large data/checkpoints kept outside the
repository under C:/training data/golf-ime-data by default.

Commands:
    python training/plan1_reranker.py build-data
    python training/plan1_reranker.py train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# -----------------------------
# OPTIONAL TRAINING DEPENDENCIES
# -----------------------------
#
# build-data is pure Python and can run before transformers is installed.
# train and checkpoint inference require torch + transformers and fail loudly.

torch = None
F = None
nn = None
DataLoader = None
Dataset = None
AutoModel = None
AutoTokenizer = None
get_cosine_schedule_with_warmup = None
_TRAINING_IMPORT_ERROR: Optional[BaseException] = None


def _try_import_training_deps() -> None:
    global torch, F, nn, DataLoader, Dataset
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
        from transformers import get_cosine_schedule_with_warmup as cosine_schedule
    except ImportError as exc:
        _TRAINING_IMPORT_ERROR = exc
        return

    torch = torch_module
    F = functional_module
    nn = nn_module
    DataLoader = data_loader_class
    Dataset = dataset_class
    AutoModel = auto_model_class
    AutoTokenizer = auto_tokenizer_class
    get_cosine_schedule_with_warmup = cosine_schedule
    _TRAINING_IMPORT_ERROR = None


def require_training_deps() -> None:
    if _TRAINING_IMPORT_ERROR is not None:
        _try_import_training_deps()
    if _TRAINING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Plan1 train/inference requires torch and transformers. "
            "Install them with: python -m pip install -r requirements-train.txt"
        ) from _TRAINING_IMPORT_ERROR


_try_import_training_deps()


# -----------------------------
# DEFAULTS
# -----------------------------


class Hyperparameters:
    data_root = Path(os.environ.get("GOLF_IME_DATA_ROOT", "C:/training data/golf-ime-data"))
    encoder = os.environ.get("PLAN1_ENCODER", "hfl/chinese-macbert-base")

    max_length = int(os.environ.get("PLAN1_MAX_LENGTH", 192))
    context_before_chars = int(os.environ.get("PLAN1_CONTEXT_BEFORE_CHARS", 96))
    context_after_chars = int(os.environ.get("PLAN1_CONTEXT_AFTER_CHARS", 48))

    max_candidates = int(os.environ.get("PLAN1_MAX_CANDIDATES", 10))
    per_device_batch = int(os.environ.get("PLAN1_BATCH", 3))
    eval_batch = int(os.environ.get("PLAN1_EVAL_BATCH", 8))
    grad_accum = int(os.environ.get("PLAN1_GRAD_ACCUM", 8))
    max_epochs = int(os.environ.get("PLAN1_MAX_EPOCHS", 20))
    min_epochs = int(os.environ.get("PLAN1_MIN_EPOCHS", 3))
    patience = int(os.environ.get("PLAN1_PATIENCE", 3))

    lr = float(os.environ.get("PLAN1_LR", 2e-5))
    weight_decay = float(os.environ.get("PLAN1_WEIGHT_DECAY", 0.01))
    warmup_ratio = float(os.environ.get("PLAN1_WARMUP_RATIO", 0.06))
    max_grad_norm = float(os.environ.get("PLAN1_MAX_GRAD_NORM", 1.0))
    dropout = float(os.environ.get("PLAN1_DROPOUT", 0.1))
    num_workers = int(os.environ.get("PLAN1_NUM_WORKERS", 0))
    train_split_percent = int(os.environ.get("PLAN1_TRAIN_SPLIT_PERCENT", 90))
    val_split_percent = int(os.environ.get("PLAN1_VAL_SPLIT_PERCENT", 5))
    test_split_percent = int(os.environ.get("PLAN1_TEST_SPLIT_PERCENT", 5))


DEFAULT_WORDLISTS = (
    "thuocl_pinyin_words.jsonl",
    "rime_pinyin_simp_words.jsonl",
    "open_gram_words.jsonl",
    "hsk30_words.jsonl",
)
DEFAULT_EXISTING_SAMPLES = (
    "ranking_samples.jsonl",
    "openime_ranking_samples.jsonl",
)
DEFAULT_CORPUS_FILES = (
    "zhwiki_corpus.jsonl",
)
DEFAULT_COLAB_BATCH_CANDIDATES = "64,48,32,24,16,12,8,6,4,3,2,1"
NUMERIC_FEATURE_NAMES = (
    "log_freq",
    "candidate_len",
    "composing_len",
    "static_rank",
)
MATCH_TYPE_TO_ID = {
    "unknown": 0,
    "exact_pinyin": 1,
    "exact_short": 2,
    "prefix": 3,
    "segmented": 4,
    "association": 5,
}
CHINESE_RE = re.compile(r"[\u3400-\u9fff]+")


# -----------------------------
# SHARED FEATURES
# -----------------------------


@dataclass
class Plan1CandidateFeatures:
    freq: float = 0.0
    static_rank: int = 0
    match_type: str = "unknown"
    source: str = "unknown"
    domain: str = "unknown"

    def numeric(self, composing: str, candidate_text: str) -> List[float]:
        return [
            math.log1p(max(float(self.freq), 0.0)),
            float(len(candidate_text)),
            float(len(composing)),
            float(self.static_rank),
        ]


def normalize_label(value: Optional[str], default: str = "unknown") -> str:
    if not value:
        return default
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or default


def infer_match_type(source: str, composing: str, candidate_text: str) -> str:
    source_l = (source or "").lower()
    if "exact_short" in source_l or "abbr" in source_l:
        return "exact_short"
    if "exact" in source_l:
        return "exact_pinyin"
    if "prefix" in source_l:
        return "prefix"
    if "segmented" in source_l:
        return "segmented"
    if "association" in source_l:
        return "association"
    if composing and len(composing) <= max(1, len(candidate_text)):
        return "exact_short"
    return "unknown"


def build_pair_text(
    context_before: str,
    context_after: str,
    composing: str,
    candidate: str,
    *,
    before_chars: int = Hyperparameters.context_before_chars,
    after_chars: int = Hyperparameters.context_after_chars,
) -> str:
    before = (context_before or "")[-before_chars:]
    after = (context_after or "")[:after_chars]
    return (
        f"[context_before] {before}\n"
        f"[composing] {composing or ''}\n"
        f"[candidate] {candidate or ''}\n"
        f"[context_after] {after}"
    )


def safe_lookup(mapping: Dict[str, int], value: str) -> int:
    return int(mapping.get(normalize_label(value), mapping.get("unknown", 0)))


# -----------------------------
# DATA BUILDING
# -----------------------------


def normalize_pinyin(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return "".join(ch.lower() for ch in stripped if "a" <= ch.lower() <= "z")


def short_pinyin(value: str) -> str:
    cleaned = unicodedata.normalize("NFD", value or "")
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Mn")
    parts = [part for part in re.split(r"[^A-Za-z]+", cleaned) if part]
    if parts:
        return "".join(part[0].lower() for part in parts)
    return normalize_pinyin(value)[:1]


def stable_bucket(value: str, modulo: int = 100) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def validate_split_percents(train_percent: int, val_percent: int, test_percent: int) -> None:
    percents = {
        "train": train_percent,
        "val": val_percent,
        "test": test_percent,
    }
    for name, value in percents.items():
        if value < 0:
            raise ValueError(f"{name} split percent must be non-negative, got {value}")
    total = train_percent + val_percent + test_percent
    if total != 100:
        raise ValueError(
            "Plan1 split percentages must sum to 100, "
            f"got train={train_percent}, val={val_percent}, test={test_percent}"
        )
    if train_percent == 0 or val_percent == 0 or test_percent == 0:
        raise ValueError("Plan1 train/val/test split percentages must all be greater than 0")


def split_name(source_doc_id: str, train_percent: int = 90, val_percent: int = 5) -> str:
    bucket = stable_bucket(source_doc_id or "unknown")
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + val_percent:
        return "val"
    return "test"


def load_word_index(
    wordlist_paths: Iterable[Path],
) -> Tuple[Dict[str, dict], Dict[str, List[dict]], Dict[str, List[dict]]]:
    words: Dict[str, dict] = {}
    full_map: Dict[str, List[dict]] = defaultdict(list)
    short_map: Dict[str, List[dict]] = defaultdict(list)

    for path in wordlist_paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                obj = json.loads(line)
                word = str(obj.get("word", "")).strip()
                full = normalize_pinyin(str(obj.get("pinyin", "")))
                abbr = normalize_pinyin(str(obj.get("short_pinyin", ""))) or short_pinyin(
                    str(obj.get("pinyin", ""))
                )
                if not word or not full:
                    continue
                freq = float(obj.get("freq", 0.0) or 0.0)
                entry = {
                    "word": word,
                    "pinyin": full,
                    "short_pinyin": abbr,
                    "freq": freq,
                    "source": obj.get("source", path.stem),
                    "domain": obj.get("domain", "general") or "general",
                }
                current = words.get(word)
                if current is None or freq > float(current.get("freq", 0.0)):
                    words[word] = entry

    for entry in words.values():
        full_map[entry["pinyin"]].append(entry)
        if entry["short_pinyin"]:
            short_map[entry["short_pinyin"]].append(entry)
    for bucket in list(full_map.values()) + list(short_map.values()):
        bucket.sort(
            key=lambda item: (-float(item.get("freq", 0.0)), len(item["word"]), item["word"])
        )
    return words, full_map, short_map


def candidate_entries(
    *,
    target: str,
    composing: str,
    target_entry: dict,
    full_map: Dict[str, List[dict]],
    short_map: Dict[str, List[dict]],
    max_candidates: int,
) -> List[dict]:
    full = target_entry["pinyin"]
    abbr = target_entry.get("short_pinyin", "")
    if composing == full:
        pool = full_map.get(full, [])
        match_type = "exact_pinyin"
    elif composing == abbr:
        pool = short_map.get(abbr, [])
        match_type = "exact_short"
    else:
        pool = full_map.get(full, [])
        match_type = "unknown"

    selected = [entry for entry in pool if entry["word"] != target][: max_candidates - 1]
    target_item = dict(target_entry)
    target_item["match_type"] = match_type
    insert_at = min(len(selected), stable_bucket(target + composing, max_candidates))
    selected.insert(insert_at, target_item)
    selected = selected[:max_candidates]

    output = []
    for rank, entry in enumerate(selected, start=1):
        output.append(
            {
                "word": entry["word"],
                "freq": float(entry.get("freq", 0.0) or 0.0),
                "source": entry.get("source", "wordlist"),
                "domain": entry.get("domain", "general") or "general",
                "static_rank": rank,
                "match_type": entry.get("match_type", match_type),
            }
        )
    return output


def enrich_existing_sample(sample: dict, words: Dict[str, dict], sample_source: str) -> Optional[dict]:
    candidates = list(sample.get("candidates") or [])
    target = sample.get("target")
    if not target or target not in candidates:
        return None

    composing = str(sample.get("composing", ""))
    features = []
    for rank, candidate in enumerate(candidates, start=1):
        entry = words.get(candidate, {})
        full = entry.get("pinyin", sample.get("target_pinyin_full", "") if candidate == target else "")
        abbr = entry.get("short_pinyin", sample.get("target_pinyin_abbr", "") if candidate == target else "")
        if composing and composing == full:
            match_type = "exact_pinyin"
        elif composing and composing == abbr:
            match_type = "exact_short"
        else:
            match_type = "unknown"
        features.append(
            {
                "freq": float(entry.get("freq", 0.0) or 0.0),
                "source": entry.get("source", sample_source),
                "domain": entry.get("domain", sample.get("domain", "general") or "general"),
                "static_rank": rank,
                "match_type": match_type,
            }
        )

    result = dict(sample)
    result["candidate_features"] = features
    result["target_index"] = candidates.index(target)
    result["sample_source"] = sample_source
    if not result.get("source_doc_id"):
        stable_key = "\n".join(
            [
                sample_source,
                str(sample.get("context_before", "")),
                str(sample.get("composing", "")),
                str(sample.get("target", "")),
            ]
        )
        result["source_doc_id"] = (
            f"{sample_source}:{hashlib.sha1(stable_key.encode('utf-8')).hexdigest()}"
        )
    if sample_source.startswith("openime"):
        result["license_bucket"] = "research_only"
    else:
        result["license_bucket"] = result.get("license_bucket", "attribution")
    return result


def iter_known_words(text: str, words: Dict[str, dict]) -> Iterator[Tuple[int, int, str]]:
    for match in CHINESE_RE.finditer(text):
        seq = match.group(0)
        offset = match.start()
        max_len = min(6, len(seq))
        for start in range(len(seq)):
            found = None
            for size in range(max_len, 0, -1):
                if start + size > len(seq):
                    continue
                token = seq[start : start + size]
                if token in words:
                    found = token
                    break
            if found:
                begin = offset + start
                yield begin, begin + len(found), found


def build_sample_from_corpus(
    doc: dict,
    begin: int,
    end: int,
    target: str,
    target_entry: dict,
    full_map: Dict[str, List[dict]],
    short_map: Dict[str, List[dict]],
    max_candidates: int,
) -> Optional[dict]:
    full = target_entry["pinyin"]
    abbr = target_entry.get("short_pinyin", "")
    if not full:
        return None

    composing = full if stable_bucket(str(doc.get("id", "")) + target + str(begin)) < 70 else abbr
    if not composing:
        composing = full
    candidates = candidate_entries(
        target=target,
        composing=composing,
        target_entry=target_entry,
        full_map=full_map,
        short_map=short_map,
        max_candidates=max_candidates,
    )
    candidate_words = [entry["word"] for entry in candidates]
    if target not in candidate_words:
        return None

    text = doc.get("text", "")
    source_label = normalize_label(str(doc.get("source", "zhwiki") or "zhwiki"))
    return {
        "context_before": text[max(0, begin - 160) : begin],
        "context_after": text[end : end + 96],
        "composing": composing,
        "target": target,
        "target_pinyin_full": full,
        "target_pinyin_abbr": abbr,
        "candidates": candidate_words,
        "candidate_features": [
            {key: value for key, value in entry.items() if key != "word"}
            for entry in candidates
        ],
        "target_index": candidate_words.index(target),
        "source_doc_id": f"{source_label}:{doc.get('id', 'unknown')}",
        "domain": doc.get("domain", "encyclopedia"),
        "sample_source": f"{source_label}_generated",
        "license_bucket": doc.get("license_bucket", "attribution"),
    }


def write_sample(
    writers: Dict[str, Any],
    sample: dict,
    counters: Counter,
    train_percent: int,
    val_percent: int,
) -> None:
    source_id = str(sample.get("source_doc_id", "unknown"))
    split = split_name(source_id, train_percent=train_percent, val_percent=val_percent)
    candidates = list(sample.get("candidates") or [])
    target = sample.get("target")
    target_index = sample.get("target_index")
    if target not in candidates:
        counters["target_missing"] += 1
        return
    if not isinstance(target_index, int) or target_index < 0 or target_index >= len(candidates):
        target_index = candidates.index(target)
        sample["target_index"] = target_index

    line = json.dumps(sample, ensure_ascii=False)
    writers["all"].write(line + "\n")
    writers[split].write(line + "\n")
    counters["total"] += 1
    counters[f"split_{split}"] += 1
    counters[f"source_{sample.get('sample_source', 'unknown')}"] += 1
    counters[f"license_{sample.get('license_bucket', 'unknown')}"] += 1
    counters[f"candidates_{len(candidates)}"] += 1
    for k in (1, 3, 5, 10):
        if target_index < min(k, len(candidates)):
            counters[f"oracle_top{k}"] += 1


def data_output_paths(processed_dir: Path, output_prefix: str) -> Dict[str, Path]:
    return {
        "all": processed_dir / f"{output_prefix}_samples.jsonl",
        "train": processed_dir / f"{output_prefix}_train.jsonl",
        "val": processed_dir / f"{output_prefix}_val.jsonl",
        "test": processed_dir / f"{output_prefix}_test.jsonl",
    }


def write_data_manifest(
    *,
    manifest_path: Path,
    output_paths: Dict[str, Path],
    wordlist_paths: List[Path],
    words: Dict[str, dict],
    full_map: Dict[str, List[dict]],
    short_map: Dict[str, List[dict]],
    args: argparse.Namespace,
    counters: Counter,
) -> None:
    total = max(int(counters["total"]), 1)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "output_paths": {key: str(path) for key, path in output_paths.items()},
        "wordlists": [str(path) for path in wordlist_paths],
        "word_count": len(words),
        "full_pinyin_keys": len(full_map),
        "short_pinyin_keys": len(short_map),
        "sample_counts": {
            "total": int(counters["total"]),
            "train": int(counters["split_train"]),
            "val": int(counters["split_val"]),
            "test": int(counters["split_test"]),
            "target_missing_or_filtered": int(counters["target_missing"]),
        },
        "candidate_count_distribution": {
            key.replace("candidates_", ""): value
            for key, value in sorted(counters.items())
            if key.startswith("candidates_")
        },
        "oracle_coverage": {
            f"top{k}": counters[f"oracle_top{k}"] / total
            for k in (1, 3, 5, 10)
        },
        "max_candidates": args.max_candidates,
        "skip_corpus_generation": args.skip_corpus_generation,
        "corpus_files": args.corpus_file or list(DEFAULT_CORPUS_FILES),
        "max_generated": args.max_generated,
        "max_generated_per_corpus": args.max_generated_per_corpus,
        "max_docs": args.max_docs,
        "split_percentages": {
            "train": args.train_split_percent,
            "val": args.val_split_percent,
            "test": args.test_split_percent,
        },
        "counters": dict(counters),
        "license_note": (
            "openime_existing samples are research_only. "
            "Wikipedia-derived samples inherit CC BY-SA 4.0 attribution/share-alike obligations."
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_plan1_data(args: argparse.Namespace) -> None:
    validate_split_percents(
        args.train_split_percent,
        args.val_split_percent,
        args.test_split_percent,
    )
    data_root = Path(args.data_root)
    processed = data_root / "processed"
    manifests = data_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    wordlist_paths = [processed / name for name in DEFAULT_WORDLISTS]
    words, full_map, short_map = load_word_index(wordlist_paths)
    if not words:
        raise RuntimeError("No usable word entries loaded for Plan1 data build")

    output_paths = data_output_paths(processed, args.output_prefix)
    counters: Counter = Counter()
    writers = {
        name: path.open("w", encoding="utf-8", newline="\n")
        for name, path in output_paths.items()
    }
    try:
        for existing_name in DEFAULT_EXISTING_SAMPLES:
            path = processed / existing_name
            if not path.is_file():
                continue
            sample_source = "openime_existing" if "openime" in existing_name else "zhwiki_existing"
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    counters[f"read_{existing_name}"] += 1
                    sample = enrich_existing_sample(json.loads(line), words, sample_source)
                    if sample is None:
                        counters[f"filtered_{existing_name}"] += 1
                        counters["target_missing"] += 1
                        continue
                    write_sample(
                        writers,
                        sample,
                        counters,
                        args.train_split_percent,
                        args.val_split_percent,
                    )

        generated = 0
        corpus_files = args.corpus_file or list(DEFAULT_CORPUS_FILES)
        if not args.skip_corpus_generation:
            for corpus_name in corpus_files:
                corpus_path = Path(corpus_name)
                if not corpus_path.is_absolute():
                    corpus_path = processed / corpus_path
                if not corpus_path.is_file():
                    raise FileNotFoundError(corpus_path)
                generated_for_corpus = 0
                with corpus_path.open("r", encoding="utf-8") as file:
                    for line_no, line in enumerate(file, start=1):
                        if args.max_docs and line_no > args.max_docs:
                            break
                        if args.max_generated and generated >= args.max_generated:
                            break
                        if (
                            args.max_generated_per_corpus
                            and generated_for_corpus >= args.max_generated_per_corpus
                        ):
                            break
                        if not line.strip():
                            continue
                        doc = json.loads(line)
                        text = str(doc.get("text", ""))
                        for begin, end, target in iter_known_words(text, words):
                            if args.max_generated and generated >= args.max_generated:
                                break
                            if (
                                args.max_generated_per_corpus
                                and generated_for_corpus >= args.max_generated_per_corpus
                            ):
                                break
                            entry = words.get(target)
                            if not entry:
                                continue
                            sample = build_sample_from_corpus(
                                doc,
                                begin,
                                end,
                                target,
                                entry,
                                full_map,
                                short_map,
                                args.max_candidates,
                            )
                            if sample is None:
                                counters["generated_filtered"] += 1
                                continue
                            write_sample(
                                writers,
                                sample,
                                counters,
                                args.train_split_percent,
                                args.val_split_percent,
                            )
                            generated += 1
                            generated_for_corpus += 1
                        if line_no % 10000 == 0:
                            print(
                                "processed_corpus="
                                f"{corpus_path.name} processed_docs={line_no} "
                                f"generated_for_corpus={generated_for_corpus} "
                                f"generated_total={generated} total={counters['total']}",
                                flush=True,
                            )
    finally:
        for writer in writers.values():
            writer.close()

    write_data_manifest(
        manifest_path=manifests / f"{args.output_prefix}_data_manifest.json",
        output_paths=output_paths,
        wordlist_paths=wordlist_paths,
        words=words,
        full_map=full_map,
        short_map=short_map,
        args=args,
        counters=counters,
    )


# -----------------------------
# DATA AUDITING / CLEANING / PACKAGING
# -----------------------------


VALID_LICENSE_BUCKETS = frozenset({"open", "attribution", "research_only", "unknown"})
SCHEMA_VERSION = "plan1_ranking_v1"
CLEANING_FEEDBACK_SCHEMA_VERSION = "plan1_cleaning_feedback_v1"
CLEANING_FEEDBACK_SLICE_FIELDS = (
    "sample_source",
    "license_bucket",
    "domain",
    "source_doc_prefix",
    "candidate_count_bucket",
    "composing_len_bucket",
    "target_index_bucket",
)
CHECKPOINT_LABEL_SPACE_POLICIES = ("strict", "expand-compatible")
LABEL_EMBEDDING_SPECS = {
    "match_type_to_id": "match_embedding.weight",
    "source_to_id": "source_embedding.weight",
    "domain_to_id": "domain_embedding.weight",
}
_SOURCE_PREFIX_MAP = {
    "zhwiki_existing": "zhwiki",
    "zhwiki_generated": "zhwiki",
    "news2016zh_existing": "news2016zh",
    "news2016zh_generated": "news2016zh",
    "baike2018qa_existing": "baike2018qa",
    "baike2018qa_generated": "baike2018qa",
    "webtext2019zh_existing": "webtext2019zh",
    "webtext2019zh_generated": "webtext2019zh",
    "openime_existing": "openime",
    "openime_generated": "openime",
}


def normalize_source_doc_key(source_doc_id: Any, sample_source: str) -> Optional[str]:
    """Normalize source_doc_id to a canonical source_doc_key.

    Rules:
    - If source_doc_id already has a colon, use it as-is.
    - If source_doc_id is a plain number/id, derive source prefix from sample_source.
    - source_doc_id that is empty/None/unknown → cannot recover → return None.
    """
    sid = str(source_doc_id or "").strip()
    if not sid or sid.lower() in ("unknown", "none", ""):
        return None
    if ":" in sid:
        return sid
    prefix = _SOURCE_PREFIX_MAP.get(sample_source, "unknown")
    return f"{prefix}:{sid}"


def _recover_pinyin_from_index(target: str, words: Dict[str, dict]) -> Tuple[str, str]:
    """Return (full_pinyin, short_pinyin) for target from word index, or ('', '')."""
    entry = words.get(target)
    if not entry:
        return "", ""
    full = normalize_pinyin(str(entry.get("pinyin", "")))
    if not full:
        return "", ""
    abbr = normalize_pinyin(str(entry.get("short_pinyin", ""))) or short_pinyin(full)
    return full, abbr


def _rebuild_candidate_features(
    candidates: List[str],
    words: Dict[str, dict],
    composing: str,
    sample_domain: str,
    sample_source: str,
) -> List[dict]:
    """Build candidate_features list matching candidates length with correct static_rank."""
    features = []
    for rank, candidate in enumerate(candidates, start=1):
        entry = words.get(candidate, {})
        full = normalize_pinyin(str(entry.get("pinyin", "")))
        abbr = normalize_pinyin(str(entry.get("short_pinyin", "")))
        if composing and composing == full:
            match_type = "exact_pinyin"
        elif composing and composing == abbr:
            match_type = "exact_short"
        else:
            match_type = infer_match_type(
                entry.get("source", "wordlist"), composing, candidate
            )
        features.append(
            {
                "freq": float(entry.get("freq", 0.0) or 0.0),
                "static_rank": rank,
                "match_type": match_type,
                "source": str(entry.get("source", sample_source) or "unknown"),
                "domain": str(entry.get("domain", sample_domain) or "general"),
            }
        )
    return features


def _supplement_candidates(
    target: str,
    composing: str,
    existing_candidates: List[str],
    words: Dict[str, dict],
    full_map: Dict[str, List[dict]],
    short_map: Dict[str, List[dict]],
    max_candidates: int,
) -> Optional[List[str]]:
    """Try to add more candidates from the word index.

    Returns a list of at least 2 candidates (including target), or None if impossible.
    Target must appear exactly once in the returned list.
    """
    # Find target entry
    target_entry = words.get(target)
    if not target_entry:
        return None
    full = target_entry.get("pinyin", "")
    abbr = target_entry.get("short_pinyin", "")
    if not full and not abbr:
        return None

    # Determine pool from the composing / pinyin match
    if composing and composing == full:
        pool = list(full_map.get(full, []))
    elif composing and composing == abbr:
        pool = list(short_map.get(abbr, []))
    else:
        pool = list(full_map.get(full, []))
    if not pool:
        pool = list(short_map.get(abbr, []))

    # Remove target from pool, then select candidates
    pool = [e for e in pool if e["word"] != target]
    # Deduplicate existing candidates while preserving order
    seen = set()
    deduped = []
    for c in existing_candidates:
        if c != target and c not in seen:
            deduped.append(c)
            seen.add(c)

    # Fill: first from existing (up to max_candidates-1), then from pool
    selected = []
    for c in deduped:
        if len(selected) >= max_candidates - 1:
            break
        if c not in seen or c == target:
            continue
        selected.append(c)

    for entry in pool:
        if len(selected) >= max_candidates - 1:
            break
        if entry["word"] not in seen and entry["word"] != target:
            selected.append(entry["word"])
            seen.add(entry["word"])

    if len(selected) < 1:
        return None  # Need at least 2 candidates total (target + 1 other)

    # Insert target at a stable position
    insert_at = stable_bucket(target + composing, min(max_candidates, len(selected) + 1))
    insert_at = min(insert_at, len(selected))
    selected.insert(insert_at, target)
    return selected[:max_candidates]


def clean_one_sample(
    sample: dict,
    words: Dict[str, dict],
    full_map: Dict[str, List[dict]],
    short_map: Dict[str, List[dict]],
    max_candidates: int,
) -> Tuple[Optional[dict], List[str], List[str]]:
    """Clean a single sample. Returns (cleaned_sample_or_None, repairs, drops).

    repair/drop tags are short strings for audit aggregation.
    """
    repairs: List[str] = []
    drops: List[str] = []

    # --- Parse ---
    try:
        obj = dict(sample)
    except Exception:
        return None, [], ["json_parse_error"]

    # --- Required strings ---
    target = str(obj.get("target", "")).strip()
    composing = str(obj.get("composing", "")).strip()
    context_before = str(obj.get("context_before", ""))
    context_after = str(obj.get("context_after", ""))
    source_doc_id = obj.get("source_doc_id")
    sample_source = str(obj.get("sample_source", "unknown"))

    if not target:
        drops.append("target_empty")
        return None, repairs, drops
    if not composing:
        drops.append("composing_empty")
        return None, repairs, drops
    if not context_before.strip() and not context_after.strip():
        drops.append("context_both_empty")
        return None, repairs, drops

    # --- source_doc_key ---
    source_doc_key = normalize_source_doc_key(source_doc_id, sample_source)
    if not source_doc_key:
        drops.append("source_doc_id_unrecoverable")
        return None, repairs, drops

    # --- target_pinyin_full ---
    target_pinyin_full = str(obj.get("target_pinyin_full", "")).strip()
    if not target_pinyin_full:
        full, abbr_tmp = _recover_pinyin_from_index(target, words)
        if full:
            target_pinyin_full = full
            repairs.append("target_pinyin_full_recovered")
        else:
            drops.append("target_pinyin_full_unrecoverable")
            return None, repairs, drops

    # --- target_pinyin_abbr ---
    target_pinyin_abbr = str(obj.get("target_pinyin_abbr", "")).strip()
    if not target_pinyin_abbr:
        _, abbr_tmp = _recover_pinyin_from_index(target, words)
        if abbr_tmp:
            target_pinyin_abbr = abbr_tmp
        else:
            target_pinyin_abbr = short_pinyin(target_pinyin_full)
        repairs.append("target_pinyin_abbr_recovered")

    # --- candidates ---
    candidates = list(obj.get("candidates") or [])
    if not candidates:
        drops.append("candidates_empty")
        return None, repairs, drops

    # Deduplicate candidates while preserving order (first occurrence wins)
    seen_cands: Dict[str, int] = {}
    dedup_candidates: List[str] = []
    dedup_map: List[int] = []  # old_index -> new_index (-1 if duplicate)
    for i, c in enumerate(candidates):
        c_str = str(c).strip()
        if c_str not in seen_cands:
            seen_cands[c_str] = len(dedup_candidates)
            dedup_candidates.append(c_str)
            dedup_map.append(len(dedup_candidates) - 1)
        else:
            dedup_map.append(-1)
            repairs.append("candidate_duplicate_removed")

    if len(dedup_candidates) < 2:
        # Try to supplement from word index
        supplemented = _supplement_candidates(
            target, composing, dedup_candidates, words, full_map, short_map, max_candidates
        )
        if supplemented is None or len(supplemented) < 2:
            drops.append("candidates_under_2_after_dedup_unsupplementable")
            return None, repairs, drops
        dedup_candidates = supplemented
        repairs.append("candidates_supplemented_from_wordlist")

    # --- target in candidates ---
    if target not in dedup_candidates:
        drops.append("target_not_in_candidates")
        return None, repairs, drops

    # --- candidate_features ---
    candidate_features = list(obj.get("candidate_features") or [])
    # If features were already deduped, rebuild them for simplicity
    if len(candidate_features) != len(dedup_candidates):
        candidate_features = _rebuild_candidate_features(
            dedup_candidates, words, composing,
            str(obj.get("domain", "general")), sample_source,
        )
        repairs.append("candidate_features_rebuilt")
        # Remap existing features if dedup happened
        if len(dedup_map) != len(candidates) and repairs.count("candidate_duplicate_removed") > 0:
            new_features = []
            for i in range(len(dedup_candidates)):
                # Find original feature for this deduped candidate
                found = False
                for old_i, new_i in enumerate(dedup_map):
                    if new_i == i and old_i < len(candidate_features):
                        feat = dict(candidate_features[old_i])
                        feat["static_rank"] = i + 1
                        new_features.append(feat)
                        found = True
                        break
                if not found:
                    # Use defaults
                    new_features.append({
                        "freq": 0.0,
                        "static_rank": i + 1,
                        "match_type": "unknown",
                        "source": sample_source,
                        "domain": str(obj.get("domain", "general")),
                    })
            candidate_features = new_features
    else:
        # Ensure each feature has required fields
        domain_default = str(obj.get("domain", "general"))
        for i, feat in enumerate(candidate_features):
            if "freq" not in feat:
                feat["freq"] = 0.0
                repairs.append("feature_freq_defaulted")
            if "static_rank" not in feat or feat.get("static_rank") != i + 1:
                feat["static_rank"] = i + 1
                repairs.append("feature_static_rank_fixed")
            if "match_type" not in feat:
                feat["match_type"] = "unknown"
                repairs.append("feature_match_type_defaulted")
            if "source" not in feat:
                feat["source"] = sample_source
                repairs.append("feature_source_defaulted")
            if "domain" not in feat:
                feat["domain"] = domain_default
                repairs.append("feature_domain_defaulted")

    # --- target_index ---
    target_index = obj.get("target_index")
    try:
        target_index = int(target_index)
    except (TypeError, ValueError):
        target_index = -1
    if target_index < 0 or target_index >= len(dedup_candidates):
        target_index = dedup_candidates.index(target)
        repairs.append("target_index_fixed")
    elif dedup_candidates[target_index] != target:
        target_index = dedup_candidates.index(target)
        repairs.append("target_index_fixed")

    # --- domain ---
    domain = str(obj.get("domain", "general") or "general").strip()
    if not domain:
        domain = "general"
        repairs.append("domain_defaulted")

    # --- license_bucket ---
    license_bucket = str(obj.get("license_bucket", "unknown")).strip()
    if license_bucket not in VALID_LICENSE_BUCKETS:
        license_bucket = "unknown"
        repairs.append("license_bucket_defaulted")

    # --- Build cleaned sample ---
    cleaned = {
        "schema_version": SCHEMA_VERSION,
        "context_before": context_before,
        "context_after": context_after,
        "composing": composing,
        "target": target,
        "target_pinyin_full": target_pinyin_full,
        "target_pinyin_abbr": target_pinyin_abbr,
        "candidates": dedup_candidates,
        "candidate_features": candidate_features,
        "target_index": target_index,
        "source_doc_id": str(source_doc_id or ""),
        "source_doc_key": source_doc_key,
        "domain": domain,
        "sample_source": sample_source,
        "license_bucket": license_bucket,
    }
    return cleaned, repairs, drops


def _iter_input_splits(
    input_prefix: str,
    processed_dir: Path,
) -> Iterator[Tuple[str, dict]]:
    """Yield (original_split_name, raw_sample_dict) from input files."""
    for sn in ("train", "val", "test"):
        path = processed_dir / f"{input_prefix}_{sn}.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield sn, json.loads(line)
                except json.JSONDecodeError:
                    continue


def _build_document_split_map(
    all_samples: List[dict],
    train_pct: int,
    val_pct: int,
) -> Dict[str, str]:
    """Build mapping from source_doc_key -> split name using stable hash."""
    doc_map: Dict[str, str] = {}
    for sample in all_samples:
        key = sample.get("source_doc_key", "")
        if key and key not in doc_map:
            doc_map[key] = split_name(key, train_percent=train_pct, val_percent=val_pct)
    return doc_map


def _write_split_files_v2(
    samples: List[dict],
    output_paths: Dict[str, Path],
    doc_split_map: Dict[str, str],
) -> Counter:
    """Write samples to split files using context managers. Returns counters per split."""
    counters = Counter()
    train_path = output_paths["train"]
    val_path = output_paths["val"]
    test_path = output_paths["test"]
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w", encoding="utf-8", newline="\n") as tw, \
         val_path.open("w", encoding="utf-8", newline="\n") as vw, \
         test_path.open("w", encoding="utf-8", newline="\n") as ew:
        for sample in samples:
            key = sample.get("source_doc_key", "")
            target_split = doc_split_map.get(key, "train")
            line = json.dumps(sample, ensure_ascii=False)
            if target_split == "train":
                tw.write(line + "\n")
            elif target_split == "val":
                vw.write(line + "\n")
            else:
                ew.write(line + "\n")
            counters[f"split_{target_split}"] += 1
            counters["total"] += 1
    return counters


def _generate_supplement_from_corpus_v2(
    corpus_path: Path,
    words: Dict[str, dict],
    full_map: Dict[str, List[dict]],
    short_map: Dict[str, List[dict]],
    target_splits: set,
    needed_per_split: Dict[str, int],
    max_candidates: int,
    train_pct: int,
    val_pct: int,
    max_per_corpus: int = 0,
) -> Tuple[List[dict], Dict[str, int]]:
    """Generate supplementary samples from a corpus file.

    Only keeps samples whose source_doc_key hashes to one of target_splits,
    up to the per-split needed counts. Returns (new_samples, per_split_counts).
    """
    new_samples: List[dict] = []
    per_split_done = {s: 0 for s in target_splits}

    def _all_done() -> bool:
        return all(
            per_split_done.get(s, 0) >= needed_per_split.get(s, 0)
            for s in target_splits
        )

    with corpus_path.open("r", encoding="utf-8") as fh:
        for doc_line in fh:
            if not doc_line.strip():
                continue
            if _all_done():
                break
            if max_per_corpus and len(new_samples) >= max_per_corpus:
                break
            try:
                doc = json.loads(doc_line.strip())
            except json.JSONDecodeError:
                continue
            text = str(doc.get("text", ""))
            for begin, end, target in iter_known_words(text, words):
                if _all_done():
                    break
                if max_per_corpus and len(new_samples) >= max_per_corpus:
                    break
                entry = words.get(target)
                if not entry:
                    continue
                raw = build_sample_from_corpus(
                    doc, begin, end, target, entry,
                    full_map, short_map, max_candidates,
                )
                if raw is None:
                    continue
                cleaned, _repairs, _drops = clean_one_sample(
                    raw, words, full_map, short_map, max_candidates,
                )
                if cleaned is None:
                    continue
                key = cleaned.get("source_doc_key", "")
                sp = split_name(key, train_percent=train_pct, val_percent=val_pct)
                if sp not in target_splits:
                    continue
                needed = needed_per_split.get(sp, 0)
                if needed > 0 and per_split_done.get(sp, 0) >= needed:
                    continue
                new_samples.append(cleaned)
                per_split_done[sp] = per_split_done.get(sp, 0) + 1

    return new_samples, dict(per_split_done)


def _append_supplemented(
    samples: List[dict],
    output_paths: Dict[str, Path],
    train_pct: int,
    val_pct: int,
    split_tracker: Dict[str, int],
) -> None:
    """Append supplemented samples to their split files based on source_doc_key hash.

    Deprecated: use _write_split_files_v2 with combined samples instead.
    """
    # This function is kept for backward compat but should not be used in new flows.
    # Use _write_split_files_v2 with combined all_clean + all_supplemented instead.


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def audit_plan1_data(args: argparse.Namespace) -> None:
    """Audit input-prefix train/val/test files and generate audit markdown."""
    processed_dir = Path(args.data_root) / "processed"
    manifests_dir = Path(args.data_root) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Load word index for recovery checks
    wordlist_paths = [processed_dir / name for name in DEFAULT_WORDLISTS]
    words, full_map, short_map = load_word_index(wordlist_paths)

    input_prefix = args.input_prefix
    stats: Dict[str, Counter] = {}
    all_samples: List[dict] = []

    # Collect and categorize
    global_drops = Counter()
    global_repairs = Counter()
    candidate_count_by_split: Dict[str, Counter] = {}
    sample_source_by_split: Dict[str, Counter] = {}
    license_by_split: Dict[str, Counter] = {}
    oracle_by_split: Dict[str, Counter] = {}

    for sn in ("train", "val", "test"):
        path = processed_dir / f"{input_prefix}_{sn}.jsonl"
        if not path.is_file():
            print(f"Warning: {path} not found, skipping.")
            continue
        ccount = Counter()
        ssource = Counter()
        lics = Counter()
        oracs = Counter()
        total = 0
        drop_here = Counter()
        repair_here = Counter()

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    global_drops["json_parse_error"] += 1
                    drop_here["json_parse_error"] += 1
                    continue

                # Quick validation (non-destructive for audit)
                target = obj.get("target")
                composing = obj.get("composing", "")
                candidates = obj.get("candidates") or []
                target_pinyin_full = obj.get("target_pinyin_full", "")

                total += 1
                ccount[len(candidates)] += 1
                ssource[obj.get("sample_source", "unknown")] += 1
                lics[obj.get("license_bucket", "unknown")] += 1

                target_idx = obj.get("target_index")
                if isinstance(target_idx, int) and 0 <= target_idx < len(candidates):
                    for k in (1, 3, 5, 10):
                        if target_idx < min(k, len(candidates)):
                            oracs[f"oracle_top{k}"] += 1

                # Count issues
                if not target or not str(target).strip():
                    drop_here["target_empty"] += 1
                if not str(composing).strip():
                    drop_here["composing_empty"] += 1
                if not str(obj.get("context_before", "")).strip() and not str(obj.get("context_after", "")).strip():
                    drop_here["context_both_empty"] += 1
                if len(candidates) < 2:
                    drop_here["candidates_under_2"] += 1
                if target and target not in candidates:
                    drop_here["target_not_in_candidates"] += 1
                if not str(target_pinyin_full).strip():
                    # Check if recoverable
                    full, _ = _recover_pinyin_from_index(str(target), words)
                    if full:
                        repair_here["target_pinyin_full_recoverable"] += 1
                    else:
                        drop_here["target_pinyin_full_unrecoverable"] += 1

                sid = obj.get("source_doc_id")
                ss = obj.get("sample_source", "")
                sk = normalize_source_doc_key(sid, str(ss))
                if not sk:
                    drop_here["source_doc_id_unrecoverable"] += 1

        candidate_count_by_split[sn] = ccount
        sample_source_by_split[sn] = ssource
        license_by_split[sn] = lics
        oracle_by_split[sn] = oracs
        stats[sn] = Counter({"total": total})
        global_drops.update(drop_here)
        global_repairs.update(repair_here)

    # Document leakage check
    doc_to_splits: Dict[str, set] = defaultdict(set)
    for sn_aud in ("train", "val", "test"):
        path = processed_dir / f"{input_prefix}_{sn_aud}.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = obj.get("source_doc_id")
                ss = str(obj.get("sample_source", ""))
                sk = normalize_source_doc_key(sid, ss)
                if sk:
                    doc_to_splits[sk].add(sn_aud)

    leakage_count = sum(1 for splits in doc_to_splits.values() if len(splits) > 1)

    # Generate audit markdown
    audit_lines = [
        f"# Plan1 Ranking Data Audit: {input_prefix}",
        f"",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Sample Counts",
    ]
    for sn_aud in ("train", "val", "test"):
        c = stats.get(sn_aud, Counter({"total": 0}))
        audit_lines.append(f"- **{sn_aud}**: {c['total']} samples")
    grand_total = sum(stats[s]["total"] for s in stats)
    audit_lines.append(f"- **total**: {grand_total}")
    audit_lines.append("")

    audit_lines.append("## Issue Summary (raw audit, non-destructive)")
    audit_lines.append("")
    audit_lines.append("| Issue | Count |")
    audit_lines.append("|-------|-------|")
    for issue, count in sorted(global_drops.items()):
        audit_lines.append(f"| {issue} | {count} |")
    for issue, count in sorted(global_repairs.items()):
        audit_lines.append(f"| {issue} (recoverable) | {count} |")
    audit_lines.append("")

    audit_lines.append("## Candidate Count Distribution by Split")
    for sn_aud in ("train", "val", "test"):
        cdist = candidate_count_by_split.get(sn_aud, Counter())
        total = stats.get(sn_aud, Counter({"total": 1}))["total"]
        denom = max(total, 1)
        audit_lines.append(f"### {sn_aud}")
        audit_lines.append("| Candidates | Count | % |")
        audit_lines.append("|------------|-------|---|")
        for k in sorted(cdist.keys()):
            audit_lines.append(f"| {k} | {cdist[k]} | {100*cdist[k]/denom:.1f}% |")
        audit_lines.append("")

    audit_lines.append("## Sample Source Distribution by Split")
    for sn_aud in ("train", "val", "test"):
        audit_lines.append(f"### {sn_aud}")
        ss_dist = sample_source_by_split.get(sn_aud, Counter())
        for k, v in sorted(ss_dist.items()):
            audit_lines.append(f"- {k}: {v}")
        audit_lines.append("")

    audit_lines.append("## License Distribution by Split")
    for sn_aud in ("train", "val", "test"):
        audit_lines.append(f"### {sn_aud}")
        lic_dist = license_by_split.get(sn_aud, Counter())
        for k, v in sorted(lic_dist.items()):
            audit_lines.append(f"- {k}: {v}")
        audit_lines.append("")

    audit_lines.append("## Oracle Top-K Coverage by Split")
    audit_lines.append("| Split | Top-1 | Top-3 | Top-5 | Top-10 |")
    audit_lines.append("|-------|-------|-------|-------|--------|")
    for sn_aud in ("train", "val", "test"):
        orac = oracle_by_split.get(sn_aud, Counter())
        total = stats.get(sn_aud, Counter({"total": 1}))["total"]
        denom = max(total, 1)
        audit_lines.append(
            f"| {sn_aud} | {100*orac['oracle_top1']/denom:.1f}% "
            f"| {100*orac['oracle_top3']/denom:.1f}% "
            f"| {100*orac['oracle_top5']/denom:.1f}% "
            f"| {100*orac['oracle_top10']/denom:.1f}% |"
        )
    audit_lines.append("")

    audit_lines.append("## Document Leakage Check")
    audit_lines.append(f"- Documents appearing in >1 split: **{leakage_count}**")
    if leakage_count > 0:
        audit_lines.append("- **WARNING**: Document leakage detected across splits!")
        leak_docs = [(k, v) for k, v in doc_to_splits.items() if len(v) > 1]
        for doc_key, splits in leak_docs[:20]:
            audit_lines.append(f"  - `{doc_key}` in {sorted(splits)}")
        if len(leak_docs) > 20:
            audit_lines.append(f"  - ... and {len(leak_docs)-20} more")
    else:
        audit_lines.append("- **PASS**: No document leakage found.")
    audit_lines.append("")

    audit_lines.append("## Recovery Feasibility")
    total_pinyin_recoverable = global_repairs.get("target_pinyin_full_recoverable", 0)
    audit_lines.append(f"- target_pinyin_full recoverable from word index: {total_pinyin_recoverable}")
    total_source_unrecoverable = global_drops.get("source_doc_id_unrecoverable", 0)
    audit_lines.append(f"- source_doc_id unrecoverable: {total_source_unrecoverable}")
    audit_lines.append("")

    audit_md = "\n".join(audit_lines)
    audit_path = manifests_dir / f"{input_prefix}_audit.md"
    with audit_path.open("w", encoding="utf-8") as fh:
        fh.write(audit_md)
    print(audit_md)
    print(f"\nAudit written to: {audit_path}")


def clean_plan1_data(args: argparse.Namespace) -> None:
    """Clean input data, re-split by source_doc_key, supplement if needed."""
    processed_dir = Path(args.data_root) / "processed"
    manifests_dir = Path(args.data_root) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Load word index
    wordlist_paths = [processed_dir / name for name in DEFAULT_WORDLISTS]
    words, full_map, short_map = load_word_index(wordlist_paths)
    if not words:
        raise RuntimeError("No usable word entries loaded for Plan1 data cleaning")

    input_prefix = args.input_prefix
    output_prefix = args.output_prefix
    max_candidates = args.max_candidates

    # ── Phase 1: Collect and clean all samples ──
    print(f"Phase 1: Cleaning samples from {input_prefix}_*.jsonl ...", flush=True)
    all_clean: List[dict] = []
    global_drops = Counter()
    global_repairs = Counter()
    total_input = 0

    for sn_orig, raw in _iter_input_splits(input_prefix, processed_dir):
        total_input += 1
        cleaned, repairs, drops = clean_one_sample(
            raw, words, full_map, short_map, max_candidates,
        )
        for d in drops:
            global_drops[d] += 1
        for r in repairs:
            global_repairs[r] += 1
        if cleaned is not None:
            all_clean.append(cleaned)
        if total_input % 100000 == 0:
            print(
                f"  processed={total_input} clean_kept={len(all_clean)} "
                f"dropped={sum(global_drops.values())}",
                flush=True,
            )

    print(
        f"Phase 1 done: input={total_input} kept={len(all_clean)} "
        f"dropped={sum(global_drops.values())} repaired={sum(global_repairs.values())}",
        flush=True,
    )

    # ── Phase 2: Build doc->split map (no writing yet) ──
    print("Phase 2: Computing doc->split map ...", flush=True)
    train_pct = args.train_split_percent
    val_pct = args.val_split_percent

    doc_split_map = _build_document_split_map(all_clean, train_pct, val_pct)
    output_paths = data_output_paths(processed_dir, output_prefix)

    # ── Phase 3: Check targets and supplement if needed ──
    target_map = {
        "train": (args.min_train_samples, args.target_train_samples),
        "val": (args.min_val_samples, args.target_val_samples),
        "test": (args.min_test_samples, args.target_test_samples),
    }

    # Count current per-split from doc_split_map applied to all_clean
    current_counts = Counter()
    for sample in all_clean:
        key = sample.get("source_doc_key", "")
        sp = doc_split_map.get(key, "train")
        current_counts[f"split_{sp}"] += 1

    print(
        f"Phase 2 done: train={current_counts['split_train']} "
        f"val={current_counts['split_val']} test={current_counts['split_test']}",
        flush=True,
    )

    undersized: Dict[str, Tuple[int, int]] = {}
    for sname, (hard_min, soft_target) in target_map.items():
        cur = current_counts.get(f"split_{sname}", 0)
        if cur < hard_min:
            undersized[sname] = (soft_target - cur, hard_min - cur)
            print(
                f"  {sname}: {cur} < hard_min={hard_min}, "
                f"need at least {hard_min - cur}, targeting {soft_target - cur}",
                flush=True,
            )

    supplemented_total = 0
    supplemented_by_split: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    supplemented_by_source: Dict[str, int] = {}

    if undersized:
        print("Phase 3: Supplementing from corpus files ...", flush=True)
        needed_per_split = {
            s: max(target_map[s][1] - current_counts.get(f"split_{s}", 0), 0)
            for s in undersized
        }

        corpus_files = args.corpus_file or list(DEFAULT_CORPUS_FILES)
        for corpus_name in corpus_files:
            corpus_path = Path(corpus_name)
            if not corpus_path.is_absolute():
                corpus_path = processed_dir / corpus_path
            if not corpus_path.is_file():
                print(f"  Warning: corpus not found: {corpus_path}, skipping.", flush=True)
                continue

            still_needed = needed_per_split.copy()
            for s in undersized:
                already = supplemented_by_split.get(s, 0)
                still_needed[s] = max(needed_per_split[s] - already, 0)
            total_still_needed = sum(still_needed.values())
            if total_still_needed <= 0:
                break

            print(
                f"  Scanning {corpus_path.name} (need {total_still_needed} more) ...",
                flush=True,
            )
            new_samples, s_counts = _generate_supplement_from_corpus_v2(
                corpus_path, words, full_map, short_map,
                target_splits=set(undersized.keys()),
                needed_per_split=still_needed,
                max_candidates=max_candidates,
                train_pct=train_pct,
                val_pct=val_pct,
            )

            src_label = normalize_label(
                str(Path(corpus_name).stem).replace("_corpus", "")
            )
            supplemented_by_source[src_label] = supplemented_by_source.get(src_label, 0) + len(new_samples)
            for s in undersized:
                supplemented_by_split[s] = supplemented_by_split.get(s, 0) + s_counts.get(s, 0)
            supplemented_total += len(new_samples)

            # Add to all_clean and update doc_split_map (new docs only)
            all_clean.extend(new_samples)
            for sample in new_samples:
                key = sample.get("source_doc_key", "")
                if key and key not in doc_split_map:
                    doc_split_map[key] = split_name(key, train_percent=train_pct, val_percent=val_pct)

            # Check if satisfied
            still_needed_after = {
                s: max(target_map[s][1] - current_counts.get(f"split_{s}", 0) - supplemented_by_split.get(s, 0), 0)
                for s in undersized
            }
            if all(v <= 0 for v in still_needed_after.values()):
                print("  All target splits satisfied.", flush=True)
                break

        print(
            f"Phase 3 done: supplemented={supplemented_total} "
            f"by_split={dict(supplemented_by_split)} by_source={supplemented_by_source}",
            flush=True,
        )
    else:
        print("Phase 3: All splits above hard minimum, skipping corpus supplementation.", flush=True)

    # ── Phase 4: Write everything at once ──
    print(f"Phase 4: Writing {len(all_clean)} samples to split files ...", flush=True)
    final_counters = _write_split_files_v2(all_clean, output_paths, doc_split_map)

    final_counts = {
        "train": final_counters["split_train"],
        "val": final_counters["split_val"],
        "test": final_counters["split_test"],
    }
    print(
        f"Phase 4 done: train={final_counts['train']} "
        f"val={final_counts['val']} test={final_counts['test']}",
        flush=True,
    )

    # ── Phase 5: Compute per-split distributions ──
    split_distributions = _compute_split_distributions(output_paths)

    # ── Phase 6: Document leakage check on output ──
    output_doc_to_splits: Dict[str, set] = defaultdict(set)
    for sn_chk in ("train", "val", "test"):
        path = output_paths[sn_chk]
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                sk = obj.get("source_doc_key", "")
                if sk:
                    output_doc_to_splits[sk].add(sn_chk)
    output_leakage = sum(1 for s in output_doc_to_splits.values() if len(s) > 1)

    # ── Phase 7: Check if targets met ──
    hard_met = all(
        final_counts.get(s, 0) >= target_map[s][0] for s in ("train", "val", "test")
    )
    soft_met = all(
        final_counts.get(s, 0) >= target_map[s][1] for s in ("train", "val", "test")
    )
    colab_ready = hard_met and output_leakage == 0

    # ── Phase 8: Write manifest ──
    total_final = sum(final_counts.values())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "output_prefix": output_prefix,
        "output_paths": {
            sn_out: str(output_paths[sn_out])
            for sn_out in ("train", "val", "test")
        },
        "sample_counts": {
            "train": final_counts.get("train", 0),
            "val": final_counts.get("val", 0),
            "test": final_counts.get("test", 0),
            "total": total_final,
        },
        "hard_minimums": {
            "train": args.min_train_samples,
            "val": args.min_val_samples,
            "test": args.min_test_samples,
        },
        "soft_targets": {
            "train": args.target_train_samples,
            "val": args.target_val_samples,
            "test": args.target_test_samples,
        },
        "hard_targets_met": hard_met,
        "soft_targets_met": soft_met,
        "colab_ready": colab_ready,
        "source_doc_key_split_policy": (
            f"stable SHA-1 hash bucket({train_pct}/{val_pct}/"
            f"{100-train_pct-val_pct}) on normalized source_doc_key"
        ),
        "dropped_counts_by_reason": dict(global_drops),
        "repaired_counts_by_reason": dict(global_repairs),
        "supplemented_counts_by_source_and_split": {
            "by_source": supplemented_by_source,
            "by_split": dict(supplemented_by_split),
            "total": supplemented_total,
        },
        "candidate_count_distribution_by_split": split_distributions.get("candidate_counts", {}),
        "sample_source_distribution_by_split": split_distributions.get("sample_sources", {}),
        "license_distribution_by_split": split_distributions.get("licenses", {}),
        "oracle_top1_top3_top5_top10_by_split": split_distributions.get("oracle", {}),
        "duplicate_sample_count_removed": global_drops.get("candidate_duplicate_removed", 0),
        "document_leakage_count": output_leakage,
        "license_note": (
            "Samples with license_bucket 'unknown' or 'research_only' "
            "are for internal research training only. "
            "Do NOT publish or redistribute without a separate license audit."
        ),
    }
    if not colab_ready:
        gaps = []
        for s in ("train", "val", "test"):
            c = final_counts.get(s, 0)
            h = target_map[s][0]
            if c < h:
                gaps.append(f"{s}: {c}/{h} (gap={h-c})")
        manifest["colab_ready_gaps"] = gaps

    manifest_path = manifests_dir / f"{output_prefix}_data_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"\nManifest written to: {manifest_path}")

    # ── Phase 9: Write audit markdown ──
    _write_clean_audit_markdown(
        manifests_dir=manifests_dir,
        output_prefix=output_prefix,
        final_counts=final_counts,
        target_map=target_map,
        hard_met=hard_met,
        soft_met=soft_met,
        colab_ready=colab_ready,
        drops=global_drops,
        repairs=global_repairs,
        supplemented_by_split=supplemented_by_split,
        supplemented_by_source=supplemented_by_source,
        split_distributions=split_distributions,
        output_leakage=output_leakage,
        train_pct=train_pct,
        val_pct=val_pct,
    )

    # ── Phase 10: Compute sha256 ──
    sha_path = manifests_dir / f"{output_prefix}_sha256.json"
    sha_hashes = {}
    for sn_sha in ("train", "val", "test"):
        path = output_paths[sn_sha]
        if path.is_file():
            sha_hashes[sn_sha] = compute_sha256(path)
    with sha_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"files": sha_hashes, "colab_ready": colab_ready},
            fh, ensure_ascii=False, indent=2,
        )
    print(f"SHA256 written to: {sha_path}")

    if not colab_ready:
        print("WARNING: colab_ready is FALSE - see manifest for gaps.")


def _append_supplemented(
    samples: List[dict],
    output_paths: Dict[str, Path],
    train_pct: int,
    val_pct: int,
    split_tracker: Dict[str, int],
) -> None:
    """Append supplemented samples to their split files based on source_doc_key hash."""
    writers: Dict[str, Any] = {}
    try:
        for sn, path in output_paths.items():
            writers[sn] = path.open("a", encoding="utf-8", newline="\n")
        for sample in samples:
            key = sample.get("source_doc_key", "")
            target_split = split_name(key, train_percent=train_pct, val_percent=val_pct)
            writers[target_split].write(json.dumps(sample, ensure_ascii=False) + "\n")
            split_tracker[target_split] = split_tracker.get(target_split, 0) + 1
    finally:
        for w in writers.values():
            w.close()


def _compute_split_distributions(
    output_paths: Dict[str, Path],
) -> Dict[str, Any]:
    """Compute per-split distributions (candidate counts, sources, licenses, oracle)."""
    ccounts: Dict[str, Counter] = {}
    ssources: Dict[str, Counter] = {}
    lics: Dict[str, Counter] = {}
    oracs: Dict[str, Counter] = {}

    for split_name in ("train", "val", "test"):
        path = output_paths[split_name]
        if not path.is_file():
            ccounts[split_name] = Counter()
            ssources[split_name] = Counter()
            lics[split_name] = Counter()
            oracs[split_name] = Counter()
            continue

        cc = Counter()
        ss = Counter()
        lc = Counter()
        oc = Counter()
        total = 0

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                total += 1
                cands = obj.get("candidates") or []
                cc[len(cands)] += 1
                ss[obj.get("sample_source", "unknown")] += 1
                lc[obj.get("license_bucket", "unknown")] += 1
                ti = obj.get("target_index", False)
                if isinstance(ti, int) and 0 <= ti < len(cands):
                    for k in (1, 3, 5, 10):
                        if ti < min(k, len(cands)):
                            oc[f"oracle_top{k}"] += 1

        ccounts[split_name] = cc
        ssources[split_name] = ss
        lics[split_name] = lc
        oracs[split_name] = oc

    return {
        "candidate_counts": {
            sn: {str(k): v for k, v in c.items()}
            for sn, c in ccounts.items()
        },
        "sample_sources": {
            sn: dict(s) for sn, s in ssources.items()
        },
        "licenses": {
            sn: dict(l) for sn, l in lics.items()
        },
        "oracle": {
            sn: dict(o) for sn, o in oracs.items()
        },
    }


def _write_clean_audit_markdown(
    *,
    manifests_dir: Path,
    output_prefix: str,
    final_counts: Dict[str, int],
    target_map: Dict[str, Tuple[int, int]],
    hard_met: bool,
    soft_met: bool,
    colab_ready: bool,
    drops: Counter,
    repairs: Counter,
    supplemented_by_split: Dict[str, int],
    supplemented_by_source: Dict[str, int],
    split_distributions: Dict[str, Any],
    output_leakage: int,
    train_pct: int,
    val_pct: int,
) -> None:
    """Write the post-clean audit markdown."""
    lines = [
        f"# Plan1 Ranking Data Audit: {output_prefix} (post-clean)",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Colab Ready**: {'YES' if colab_ready else 'NO'}",
        "",
        f"## Final Sample Counts",
        f"| Split | Count | Hard Min | Met | Soft Target | Met |",
        f"|-------|-------|----------|-----|-------------|-----|",
    ]
    for sn in ("train", "val", "test"):
        hmin, sft = target_map[sn]
        c = final_counts.get(sn, 0)
        lines.append(
            f"| {sn} | {c} | {hmin} | {'YES' if c >= hmin else 'NO'} "
            f"| {sft} | {'YES' if c >= sft else 'NO'} |"
        )
    lines.append(f"| **total** | {sum(final_counts.values())} | | | | |")
    lines.append("")

    lines.append(f"## Target Status")
    lines.append(f"- Hard minimums met: **{'YES' if hard_met else 'NO'}**")
    lines.append(f"- Soft targets met: **{'YES' if soft_met else 'NO'}**")
    lines.append(f"- Colab ready: **{'YES' if colab_ready else 'NO'}**")
    if not colab_ready:
        for sn in ("train", "val", "test"):
            hmin, _ = target_map[sn]
            c = final_counts.get(sn, 0)
            if c < hmin:
                lines.append(f"  - {sn} gap: {hmin - c}")
    lines.append("")

    lines.append(f"## Drop Statistics")
    lines.append(f"| Reason | Count |")
    lines.append(f"|--------|-------|")
    for reason, count in sorted(drops.items()):
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    lines.append(f"## Repair Statistics")
    lines.append(f"| Reason | Count |")
    lines.append(f"|--------|-------|")
    for reason, count in sorted(repairs.items()):
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    lines.append(f"## Supplement Statistics")
    lines.append(f"| Source | Count |")
    lines.append(f"|--------|-------|")
    for src, count in sorted(supplemented_by_source.items()):
        lines.append(f"| {src} | {count} |")
    lines.append(f"| **total** | {sum(supplemented_by_source.values())} |")
    lines.append("")
    lines.append("### By Split")
    for sn, count in sorted(supplemented_by_split.items()):
        lines.append(f"- {sn}: {count}")
    lines.append("")

    lines.append(f"## Candidate Count Distribution by Split")
    cdist = split_distributions.get("candidate_counts", {})
    for sn in ("train", "val", "test"):
        cnts = cdist.get(sn, {})
        lines.append(f"### {sn}")
        for k, v in sorted(cnts.items(), key=lambda x: int(x[0])):
            lines.append(f"- {k} candidates: {v}")
        lines.append("")

    lines.append(f"## Sample Source Distribution by Split")
    ssdist = split_distributions.get("sample_sources", {})
    for sn in ("train", "val", "test"):
        srcs = ssdist.get(sn, {})
        lines.append(f"### {sn}")
        for k, v in sorted(srcs.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append(f"## License Distribution by Split")
    ldist = split_distributions.get("licenses", {})
    for sn in ("train", "val", "test"):
        lics = ldist.get(sn, {})
        lines.append(f"### {sn}")
        for k, v in sorted(lics.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append(f"## Oracle Top-K by Split")
    odist = split_distributions.get("oracle", {})
    for sn in ("train", "val", "test"):
        ocs = odist.get(sn, {})
        lines.append(f"### {sn}")
        for k in ("oracle_top1", "oracle_top3", "oracle_top5", "oracle_top10"):
            lines.append(f"- {k}: {ocs.get(k, 'N/A')}")
        lines.append("")

    lines.append(f"## Document Leakage")
    lines.append(f"- Documents appearing in >1 output split: **{output_leakage}**")
    lines.append(f"  {'PASS' if output_leakage == 0 else 'FAIL'}")
    lines.append("")

    lines.append(f"## Split Policy")
    lines.append(f"- Source doc key hash: train={train_pct}% val={val_pct}% test={100-train_pct-val_pct}%")
    lines.append(f"- Same source_doc_key guaranteed not to appear in multiple splits.")
    lines.append("")

    lines.append(f"## License Note")
    lines.append(
        f"> Samples with license_bucket 'unknown' or 'research_only' "
        f"are for internal research training only. "
        f"Do NOT publish or redistribute without a separate license audit."
    )
    lines.append("")

    audit_path = manifests_dir / f"{output_prefix}_audit.md"
    md_content = "\n".join(lines)
    with audit_path.open("w", encoding="utf-8") as fh:
        fh.write(md_content)
    print(f"Audit written to: {audit_path}")


def package_plan1_data(args: argparse.Namespace) -> None:
    """Package cleaned data into export directory, with optional zip."""
    data_root = Path(args.data_root)
    processed_dir = data_root / "processed"
    manifests_dir = data_root / "manifests"
    export_dir = data_root / "exports" / args.input_prefix
    export_dir.mkdir(parents=True, exist_ok=True)

    input_prefix = args.input_prefix

    # Copy split files
    copied = []
    for split_name in ("train", "val", "test"):
        src = processed_dir / f"{input_prefix}_{split_name}.jsonl"
        if not src.is_file():
            print(f"Warning: source file not found: {src}")
            continue
        dst = export_dir / f"{input_prefix}_{split_name}.jsonl"
        with src.open("rb") as fsrc, dst.open("wb") as fdst:
            while True:
                chunk = fsrc.read(65536)
                if not chunk:
                    break
                fdst.write(chunk)
        copied.append(dst)
        print(f"Copied: {src.name} -> {dst}")

    # Copy manifest
    manifest_src = manifests_dir / f"{input_prefix}_data_manifest.json"
    if manifest_src.is_file():
        manifest_dst = export_dir / manifest_src.name
        with manifest_src.open("rb") as fsrc, manifest_dst.open("wb") as fdst:
            while True:
                chunk = fsrc.read(65536)
                if not chunk:
                    break
                fdst.write(chunk)
        copied.append(manifest_dst)
        print(f"Copied: {manifest_src.name} -> {manifest_dst}")

    # Copy audit
    audit_src = manifests_dir / f"{input_prefix}_audit.md"
    if audit_src.is_file():
        audit_dst = export_dir / audit_src.name
        with audit_src.open("rb") as fsrc, audit_dst.open("wb") as fdst:
            while True:
                chunk = fsrc.read(65536)
                if not chunk:
                    break
                fdst.write(chunk)
        copied.append(audit_dst)
        print(f"Copied: {audit_src.name} -> {audit_dst}")

    # Copy sha256
    sha_src = manifests_dir / f"{input_prefix}_sha256.json"
    if sha_src.is_file():
        sha_dst = export_dir / sha_src.name
        with sha_src.open("rb") as fsrc, sha_dst.open("wb") as fdst:
            while True:
                chunk = fsrc.read(65536)
                if not chunk:
                    break
                fdst.write(chunk)
        copied.append(sha_dst)
        print(f"Copied: {sha_src.name} -> {sha_dst}")

    # Compute sha256 of exported files
    export_hashes = {}
    for path in copied:
        export_hashes[path.name] = compute_sha256(path)

    # Write export sha256
    export_sha_path = export_dir / f"{input_prefix}_export_sha256.json"
    with export_sha_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"export_dir": str(export_dir), "files": export_hashes},
            fh, ensure_ascii=False, indent=2,
        )
    print(f"Export SHA256 written to: {export_sha_path}")

    # Try to create zip
    zip_path = export_dir.parent / f"{input_prefix}.zip"
    try:
        import zipfile

        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(copied):
                zf.write(str(path), path.relative_to(export_dir.parent))
            # Also add the sha256 file
            zf.write(
                str(export_sha_path),
                export_sha_path.relative_to(export_dir.parent),
            )
        zip_size = zip_path.stat().st_size
        print(f"Zip created: {zip_path} ({zip_size:,} bytes)")
    except Exception as exc:
        print(f"Warning: zip creation failed: {exc}")
        print(f"Export directory remains at: {export_dir}")


# -----------------------------
# MODEL
# -----------------------------


if nn is not None:

    class Plan1RerankerModel(nn.Module):
        def __init__(
            self,
            encoder: Any,
            *,
            match_type_count: int,
            source_count: int,
            domain_count: int,
            numeric_dim: int = len(NUMERIC_FEATURE_NAMES),
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            hidden_size = int(getattr(encoder.config, "hidden_size"))
            head_hidden_size = max(hidden_size // 2, 1)
            self.match_embedding = nn.Embedding(max(match_type_count, 1), 8)
            self.source_embedding = nn.Embedding(max(source_count, 1), 8)
            self.domain_embedding = nn.Embedding(max(domain_count, 1), 8)
            self.numeric_projection = nn.Sequential(
                nn.LayerNorm(numeric_dim),
                nn.Linear(numeric_dim, 16),
                nn.GELU(),
            )
            self.scorer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size + 8 + 8 + 8 + 16, head_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_size, 1),
            )

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Optional[Any] = None,
            numeric_features: Any,
            match_type_ids: Any,
            source_ids: Any,
            domain_ids: Any,
        ) -> Any:
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            outputs = self.encoder(**kwargs)
            pooled = outputs.last_hidden_state[:, 0]
            combined = torch.cat(
                [
                    pooled,
                    self.match_embedding(match_type_ids),
                    self.source_embedding(source_ids),
                    self.domain_embedding(domain_ids),
                    self.numeric_projection(numeric_features.float()),
                ],
                dim=-1,
            )
            return self.scorer(combined).squeeze(-1)

else:

    class Plan1RerankerModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_training_deps()


def save_plan1_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: str | os.PathLike[str],
    config: Dict[str, Any],
) -> None:
    require_training_deps()
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path)
    tokenizer.save_pretrained(path)
    head_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("encoder.")
    }
    torch.save(head_state, path / "plan1_head.pt")
    with (path / "plan1_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


class Plan1InferenceSession:
    def __init__(self, *, model: Any, tokenizer: Any, config: Dict[str, Any], device: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.max_length = int(config.get("max_length", Hyperparameters.max_length))
        self.before_chars = int(config.get("context_before_chars", Hyperparameters.context_before_chars))
        self.after_chars = int(config.get("context_after_chars", Hyperparameters.context_after_chars))
        self.match_type_to_id = config.get("match_type_to_id", MATCH_TYPE_TO_ID)
        self.source_to_id = config.get("source_to_id", {"unknown": 0})
        self.domain_to_id = config.get("domain_to_id", {"unknown": 0})

    @classmethod
    def load(
        cls,
        model_dir: str | os.PathLike[str],
        *,
        device_name: Optional[str] = None,
    ) -> "Plan1InferenceSession":
        require_training_deps()
        path = Path(model_dir)
        config_path = path / "plan1_config.json"
        head_path = path / "plan1_head.pt"
        if not config_path.is_file() or not head_path.is_file():
            raise FileNotFoundError(
                f"Plan1 checkpoint requires plan1_config.json and plan1_head.pt in {path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        tokenizer = AutoTokenizer.from_pretrained(path)
        encoder = AutoModel.from_pretrained(path)
        model = Plan1RerankerModel(
            encoder,
            match_type_count=len(config.get("match_type_to_id", MATCH_TYPE_TO_ID)),
            source_count=len(config.get("source_to_id", {"unknown": 0})),
            domain_count=len(config.get("domain_to_id", {"unknown": 0})),
            numeric_dim=len(config.get("numeric_feature_names", NUMERIC_FEATURE_NAMES)),
            dropout=float(config.get("dropout", 0.0)),
        )
        head_state = torch.load(head_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(head_state, strict=False)
        missing_non_encoder = [name for name in missing if not name.startswith("encoder.")]
        unexpected_non_encoder = [name for name in unexpected if not name.startswith("encoder.")]
        if missing_non_encoder or unexpected_non_encoder:
            raise RuntimeError(
                f"Plan1 head state mismatch missing={missing_non_encoder} "
                f"unexpected={unexpected_non_encoder}"
            )

        if device_name is None:
            device_name = os.environ.get(
                "PLAN1_RERANKER_DEVICE",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        device = torch.device(device_name)
        model.to(device)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, config=config, device=device)

    def score(
        self,
        *,
        context_before: str,
        context_after: str = "",
        composing: str,
        candidates: Sequence[str],
        features: Sequence[Plan1CandidateFeatures],
    ) -> List[float]:
        require_training_deps()
        if len(candidates) != len(features):
            raise ValueError("candidates and features must have identical length")

        texts = [
            build_pair_text(
                context_before,
                context_after,
                composing,
                candidate,
                before_chars=self.before_chars,
                after_chars=self.after_chars,
            )
            for candidate in candidates
        ]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tensors = {key: value.to(self.device) for key, value in encoded.items()}
        numeric = torch.tensor(
            [feature.numeric(composing, candidate) for feature, candidate in zip(features, candidates)],
            dtype=torch.float32,
            device=self.device,
        )
        match_ids = torch.tensor(
            [safe_lookup(self.match_type_to_id, feature.match_type) for feature in features],
            dtype=torch.long,
            device=self.device,
        )
        source_ids = torch.tensor(
            [safe_lookup(self.source_to_id, feature.source) for feature in features],
            dtype=torch.long,
            device=self.device,
        )
        domain_ids = torch.tensor(
            [safe_lookup(self.domain_to_id, feature.domain) for feature in features],
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            scores = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                token_type_ids=tensors.get("token_type_ids"),
                numeric_features=numeric,
                match_type_ids=match_ids,
                source_ids=source_ids,
                domain_ids=domain_ids,
            )
        return [float(value) for value in scores.detach().cpu().tolist()]


# -----------------------------
# DATASET AND BATCHING
# -----------------------------


_DatasetBase = Dataset if Dataset is not None else object


class JsonlRankingDataset(_DatasetBase):
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.offsets: List[int] = []
        with self.path.open("rb") as file:
            while True:
                pos = file.tell()
                line = file.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(pos)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        with self.path.open("rb") as file:
            file.seek(self.offsets[index])
            return json.loads(file.readline().decode("utf-8"))


def collect_label_maps(paths: Iterable[Path]) -> Dict[str, Dict[str, int]]:
    sources = {"unknown"}
    domains = {"unknown"}
    match_types = set(MATCH_TYPE_TO_ID)
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                sample = json.loads(line)
                domains.add(normalize_label(sample.get("domain", "unknown")))
                for feature in sample.get("candidate_features") or []:
                    sources.add(normalize_label(feature.get("source", "unknown")))
                    domains.add(normalize_label(feature.get("domain", sample.get("domain", "unknown"))))
                    match_types.add(normalize_label(feature.get("match_type", "unknown")))
    return {
        "source_to_id": {value: idx for idx, value in enumerate(sorted(sources))},
        "domain_to_id": {value: idx for idx, value in enumerate(sorted(domains))},
        "match_type_to_id": {value: idx for idx, value in enumerate(sorted(match_types))},
    }


def id_for(mapping: Dict[str, int], value: Any) -> int:
    return int(mapping.get(normalize_label(str(value)), mapping.get("unknown", 0)))


def make_collate_fn(tokenizer: Any, config: Dict[str, Any]) -> Any:
    require_training_deps()
    max_length = int(config["max_length"])
    before_chars = int(config["context_before_chars"])
    after_chars = int(config["context_after_chars"])
    source_to_id = config["source_to_id"]
    domain_to_id = config["domain_to_id"]
    match_type_to_id = config["match_type_to_id"]

    def collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts: List[str] = []
        numeric: List[List[float]] = []
        source_ids: List[int] = []
        domain_ids: List[int] = []
        match_ids: List[int] = []
        group_sizes: List[int] = []
        target_indices: List[int] = []
        sample_meta: List[Dict[str, Any]] = []

        for sample in samples:
            candidates = list(sample.get("candidates") or [])
            target = sample.get("target")
            if not candidates or target not in candidates:
                continue
            features = sample.get("candidate_features") or [{} for _ in candidates]
            if len(features) != len(candidates):
                features = [{} for _ in candidates]

            composing = str(sample.get("composing", ""))
            domain = sample.get("domain", "unknown")
            try:
                target_index = int(sample.get("target_index", candidates.index(target)))
            except (TypeError, ValueError):
                target_index = candidates.index(target)
            if target_index < 0 or target_index >= len(candidates) or candidates[target_index] != target:
                target_index = candidates.index(target)
            group_sizes.append(len(candidates))
            target_indices.append(target_index)
            sample_meta.append(build_feedback_sample_meta(sample, candidates, target_index))
            for rank, (candidate, feature) in enumerate(zip(candidates, features), start=1):
                plan_feature = Plan1CandidateFeatures(
                    freq=float(feature.get("freq", 0.0) or 0.0),
                    static_rank=int(feature.get("static_rank", rank) or rank),
                    match_type=feature.get("match_type", "unknown"),
                    source=feature.get("source", sample.get("sample_source", "unknown")),
                    domain=feature.get("domain", domain),
                )
                texts.append(
                    build_pair_text(
                        str(sample.get("context_before", "")),
                        str(sample.get("context_after", "")),
                        composing,
                        str(candidate),
                        before_chars=before_chars,
                        after_chars=after_chars,
                    )
                )
                numeric.append(plan_feature.numeric(composing, str(candidate)))
                source_ids.append(id_for(source_to_id, plan_feature.source))
                domain_ids.append(id_for(domain_to_id, plan_feature.domain))
                match_ids.append(id_for(match_type_to_id, plan_feature.match_type))

        if not texts:
            raise ValueError("Empty batch after filtering invalid samples")
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "encoded": encoded,
            "numeric_features": torch.tensor(numeric, dtype=torch.float32),
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "domain_ids": torch.tensor(domain_ids, dtype=torch.long),
            "match_type_ids": torch.tensor(match_ids, dtype=torch.long),
            "group_sizes": group_sizes,
            "target_indices": target_indices,
            "sample_meta": sample_meta,
        }

    return collate


def move_batch(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
    return {
        **batch,
        "encoded": encoded,
        "numeric_features": batch["numeric_features"].to(device),
        "source_ids": batch["source_ids"].to(device),
        "domain_ids": batch["domain_ids"].to(device),
        "match_type_ids": batch["match_type_ids"].to(device),
    }


def _sha1_short(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _count_bucket(value: int, bounds: Sequence[int]) -> str:
    for bound in bounds:
        if value <= bound:
            return f"le_{bound}"
    return f"gt_{bounds[-1]}" if bounds else "unknown"


def _target_index_bucket(index: int) -> str:
    if index <= 0:
        return "rank_1"
    if index <= 2:
        return "rank_2_3"
    if index <= 4:
        return "rank_4_5"
    return "rank_6_plus"


def _source_doc_prefix(source_doc_key: str) -> str:
    if ":" not in source_doc_key:
        return "unknown"
    return normalize_label(source_doc_key.split(":", 1)[0])


def build_feedback_sample_meta(
    sample: Dict[str, Any],
    candidates: Sequence[str],
    target_index: int,
) -> Dict[str, Any]:
    sample_source = str(sample.get("sample_source", "unknown") or "unknown")
    source_doc_key = str(
        sample.get("source_doc_key")
        or normalize_source_doc_key(sample.get("source_doc_id"), sample_source)
        or ""
    )
    composing = str(sample.get("composing", "") or "")
    target = str(sample.get("target", "") or "")
    identity = json.dumps(
        {
            "source_doc_key": source_doc_key,
            "sample_source": sample_source,
            "composing": composing,
            "target": target,
            "target_index": target_index,
            "candidate_count": len(candidates),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "sample_id": _sha1_short(identity),
        "source_doc_key_hash": _sha1_short(source_doc_key) if source_doc_key else "",
        "source_doc_prefix": _source_doc_prefix(source_doc_key),
        "sample_source": normalize_label(sample_source),
        "license_bucket": normalize_label(sample.get("license_bucket", "unknown")),
        "domain": normalize_label(sample.get("domain", "unknown")),
        "candidate_count": int(len(candidates)),
        "candidate_count_bucket": _count_bucket(len(candidates), (2, 4, 6, 8, 10)),
        "composing_len": len(composing),
        "composing_len_bucket": _count_bucket(len(composing), (2, 4, 8, 16, 32)),
        "target_len": len(target),
        "target_len_bucket": _count_bucket(len(target), (1, 2, 4, 8)),
        "target_index": int(target_index),
        "target_index_bucket": _target_index_bucket(int(target_index)),
    }


def split_prefix_for_path(path: Path, split: str) -> str:
    stem = path.stem
    suffix = f"_{split}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def infer_data_version(split_paths: Tuple[Path, Path, Path]) -> str:
    prefixes = {
        split_prefix_for_path(path, split)
        for split, path in zip(("train", "val", "test"), split_paths)
    }
    if len(prefixes) == 1:
        return next(iter(prefixes))
    joined = "|".join(str(path) for path in split_paths)
    return f"mixed_{_sha1_short(joined, 12)}"


def summarize_manifest_for_training(manifest: Dict[str, Any]) -> Dict[str, Any]:
    summary_keys = (
        "schema_version",
        "output_prefix",
        "sample_counts",
        "split_percentages",
        "hard_targets_met",
        "soft_targets_met",
        "colab_ready",
        "dropped_counts_by_reason",
        "repaired_counts_by_reason",
        "supplemented_counts_by_source_and_split",
        "candidate_count_distribution_by_split",
        "sample_source_distribution_by_split",
        "license_distribution_by_split",
        "oracle_top1_top3_top5_top10_by_split",
        "oracle_coverage",
        "document_leakage_count",
    )
    return {key: manifest[key] for key in summary_keys if key in manifest}


def build_training_data_profile(
    args: argparse.Namespace,
    split_paths: Tuple[Path, Path, Path],
) -> Dict[str, Any]:
    data_version = infer_data_version(split_paths)
    manifests_dir = Path(args.data_root) / "manifests"
    manifest_path = manifests_dir / f"{data_version}_data_manifest.json"
    audit_path = manifests_dir / f"{data_version}_audit.md"
    sha_path = manifests_dir / f"{data_version}_sha256.json"

    profile: Dict[str, Any] = {
        "schema_version": "plan1_training_data_profile_v1",
        "data_version": data_version,
        "split_paths": {
            "train": str(split_paths[0]),
            "val": str(split_paths[1]),
            "test": str(split_paths[2]),
        },
        "data_manifest_path": str(manifest_path) if manifest_path.is_file() else "",
        "data_audit_path": str(audit_path) if audit_path.is_file() else "",
        "data_sha256_path": str(sha_path) if sha_path.is_file() else "",
        "cleaning_rules_version": "unknown",
        "manifest_summary": {},
    }
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        schema = str(manifest.get("schema_version", "unknown"))
        has_cleaning_rules = bool(
            manifest.get("dropped_counts_by_reason")
            or manifest.get("repaired_counts_by_reason")
            or manifest.get("supplemented_counts_by_source_and_split")
        )
        profile["cleaning_rules_version"] = (
            f"{schema}:clean-data" if has_cleaning_rules else f"{schema}:build-data"
        )
        profile["manifest_summary"] = summarize_manifest_for_training(manifest)
    return profile


# -----------------------------
# METRICS
# -----------------------------


def model_scores(model: Any, batch: Dict[str, Any]) -> Any:
    encoded = batch["encoded"]
    return model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        token_type_ids=encoded.get("token_type_ids"),
        numeric_features=batch["numeric_features"],
        match_type_ids=batch["match_type_ids"],
        source_ids=batch["source_ids"],
        domain_ids=batch["domain_ids"],
    )


def listwise_loss(scores: Any, group_sizes: List[int], target_indices: List[int]) -> Any:
    require_training_deps()
    losses = []
    start = 0
    for size, target_idx in zip(group_sizes, target_indices):
        logits = scores[start : start + size].unsqueeze(0)
        target = torch.tensor([target_idx], dtype=torch.long, device=scores.device)
        losses.append(F.cross_entropy(logits, target))
        start += size
    return torch.stack(losses).mean()


def ranking_result_rows(
    scores: Any,
    group_sizes: List[int],
    target_indices: List[int],
    sample_meta: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    require_training_deps()
    start = 0
    rows: List[Dict[str, Any]] = []
    cpu_scores = scores.detach().float().cpu()
    for sample_idx, (size, target_idx) in enumerate(zip(group_sizes, target_indices)):
        group = cpu_scores[start : start + size].float()
        order = torch.argsort(group, descending=True).tolist()
        rank = order.index(target_idx) + 1 if target_idx in order else len(order) + 1
        probabilities = torch.softmax(group, dim=0)
        target_prob = (
            float(probabilities[target_idx].item())
            if 0 <= target_idx < len(probabilities)
            else 0.0
        )
        top_index = int(order[0]) if order else -1
        target_score = float(group[target_idx].item()) if 0 <= target_idx < len(group) else float("nan")
        top_score = float(group[top_index].item()) if 0 <= top_index < len(group) else float("nan")
        meta = dict(sample_meta[sample_idx]) if sample_meta and sample_idx < len(sample_meta) else {}
        rows.append(
            {
                **meta,
                "loss": -math.log(max(target_prob, 1e-12)),
                "rank": int(rank),
                "top1": 1.0 if rank <= 1 else 0.0,
                "top3": 1.0 if rank <= 3 else 0.0,
                "top5": 1.0 if rank <= 5 else 0.0,
                "mrr": 1.0 / rank,
                "ndcg10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
                "predicted_index": top_index,
                "score_margin_to_top": top_score - target_score,
            }
        )
        start += size
    return rows


def summarize_ranking_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    denom = max(len(rows), 1)
    return {
        "loss": sum(float(row.get("loss", 0.0)) for row in rows) / denom,
        "top1": sum(float(row.get("top1", 0.0)) for row in rows) / denom,
        "top3": sum(float(row.get("top3", 0.0)) for row in rows) / denom,
        "top5": sum(float(row.get("top5", 0.0)) for row in rows) / denom,
        "mrr": sum(float(row.get("mrr", 0.0)) for row in rows) / denom,
        "ndcg10": sum(float(row.get("ndcg10", 0.0)) for row in rows) / denom,
        "samples": float(len(rows)),
    }


def ranking_metrics(scores: Any, group_sizes: List[int], target_indices: List[int]) -> Dict[str, float]:
    rows = ranking_result_rows(scores, group_sizes, target_indices)
    summary = summarize_ranking_rows(rows)
    return {key: value for key, value in summary.items() if key != "loss"}


def build_cleaning_feedback(
    rows: Sequence[Dict[str, Any]],
    *,
    top_n: int,
    min_slice_samples: int,
) -> Dict[str, Any]:
    slice_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for field in CLEANING_FEEDBACK_SLICE_FIELDS:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(field, "unknown") or "unknown")].append(row)
        field_metrics: Dict[str, Dict[str, float]] = {}
        for value, group_rows in sorted(grouped.items()):
            if len(group_rows) < min_slice_samples:
                continue
            field_metrics[value] = summarize_ranking_rows(group_rows)
        slice_metrics[field] = field_metrics

    high_loss_keys = (
        "sample_id",
        "source_doc_key_hash",
        "source_doc_prefix",
        "sample_source",
        "license_bucket",
        "domain",
        "candidate_count",
        "candidate_count_bucket",
        "composing_len",
        "composing_len_bucket",
        "target_len",
        "target_len_bucket",
        "target_index",
        "target_index_bucket",
        "loss",
        "rank",
        "top1",
        "top3",
        "predicted_index",
        "score_margin_to_top",
    )
    high_loss_samples = [
        {key: row.get(key) for key in high_loss_keys if key in row}
        for row in sorted(rows, key=lambda item: float(item.get("loss", 0.0)), reverse=True)[
            : max(top_n, 0)
        ]
    ]
    return {
        "schema_version": CLEANING_FEEDBACK_SCHEMA_VERSION,
        "privacy_note": (
            "Feedback rows intentionally omit raw context, target text, and candidate text. "
            "Use sample_id/source_doc_key_hash plus slice metrics to guide the next cleaning pass."
        ),
        "slice_fields": list(CLEANING_FEEDBACK_SLICE_FIELDS),
        "min_slice_samples": int(min_slice_samples),
        "slice_metrics": slice_metrics,
        "high_loss_samples": high_loss_samples,
    }


def evaluate_with_cleaning_feedback(
    model: Any,
    loader: Any,
    device: Any,
    *,
    collect_feedback: bool,
    feedback_top_n: int,
    feedback_min_slice_samples: int,
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    require_training_deps()
    model.eval()
    all_rows: List[Dict[str, Any]] = []
    totals = {"loss": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0, "ndcg10": 0.0}
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            scores = model_scores(model, batch)
            rows = ranking_result_rows(
                scores,
                batch["group_sizes"],
                batch["target_indices"],
                batch.get("sample_meta"),
            )
            metrics = summarize_ranking_rows(rows)
            count = int(metrics["samples"])
            sample_count += count
            for key in totals:
                totals[key] += metrics[key] * count
            if collect_feedback:
                all_rows.extend(rows)
    denom = max(sample_count, 1)
    metrics_out = {
        **{key: value / denom for key, value in totals.items()},
        "samples": float(sample_count),
    }
    feedback = (
        build_cleaning_feedback(
            all_rows,
            top_n=feedback_top_n,
            min_slice_samples=feedback_min_slice_samples,
        )
        if collect_feedback
        else None
    )
    return metrics_out, feedback


def evaluate(model: Any, loader: Any, device: Any) -> Dict[str, float]:
    metrics, _ = evaluate_with_cleaning_feedback(
        model,
        loader,
        device,
        collect_feedback=False,
        feedback_top_n=0,
        feedback_min_slice_samples=1,
    )
    return metrics


# -----------------------------
# TRAINING
# -----------------------------


RESUME_ARG_KEYS = (
    "train_path",
    "val_path",
    "test_path",
    "cache_data_dir",
    "init_from_checkpoint",
    "checkpoint_label_space_policy",
    "encoder",
    "max_length",
    "context_before_chars",
    "context_after_chars",
    "per_device_batch",
    "eval_batch",
    "grad_accum",
    "max_epochs",
    "min_epochs",
    "patience",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "dropout",
    "fp16",
    "gradient_checkpointing",
    "num_workers",
    "device",
    "colab_pro_mode",
    "batch_candidates",
    "target_effective_batch",
    "checkpoint_every_steps",
    "checkpoint_every_seconds",
    "max_runtime_seconds",
    "shuffle_seed",
    "no_cleaning_feedback",
    "cleaning_feedback_top_n",
    "cleaning_feedback_min_slice_samples",
)

RESUME_COMPAT_ARG_KEYS = (
    "encoder",
    "checkpoint_label_space_policy",
    "max_length",
    "context_before_chars",
    "context_after_chars",
    "per_device_batch",
    "eval_batch",
    "grad_accum",
    "max_epochs",
    "min_epochs",
    "patience",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "dropout",
    "fp16",
    "gradient_checkpointing",
    "shuffle_seed",
)


def apply_saved_resume_args(args: argparse.Namespace, model_root: Path) -> None:
    if not args.auto_resume:
        return
    config_path = model_root / "training_config.json"
    if not config_path.is_file():
        return
    with config_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    for key in RESUME_ARG_KEYS:
        if key in saved and saved[key] is not None:
            setattr(args, key, saved[key])


def resolve_resume_state_path(args: argparse.Namespace, model_root: Path) -> Path:
    if args.resume_state:
        return Path(args.resume_state)
    return model_root / "last_state.pt"


def apply_colab_training_defaults(args: argparse.Namespace) -> None:
    if args.num_workers is None:
        args.num_workers = 2 if args.colab_pro_mode else Hyperparameters.num_workers
    if args.colab_pro_mode:
        if not args.checkpoint_every_steps:
            args.checkpoint_every_steps = 1000
        if not args.checkpoint_every_seconds:
            args.checkpoint_every_seconds = 1800
        if not args.max_runtime_seconds:
            args.max_runtime_seconds = 39000
    if args.target_effective_batch is not None and args.target_effective_batch <= 0:
        raise ValueError("--target-effective-batch must be positive when provided")
    if args.cleaning_feedback_top_n < 0:
        raise ValueError("--cleaning-feedback-top-n must be non-negative")
    if args.cleaning_feedback_min_slice_samples <= 0:
        raise ValueError("--cleaning-feedback-min-slice-samples must be positive")


def resolve_split_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    processed = Path(args.data_root) / "processed"
    train_path = Path(args.train_path or processed / "plan1_ranking_train.jsonl")
    val_path = Path(args.val_path or processed / "plan1_ranking_val.jsonl")
    test_path = Path(args.test_path or processed / "plan1_ranking_test.jsonl")
    return train_path, val_path, test_path


def cache_split_paths(
    args: argparse.Namespace,
    split_paths: Tuple[Path, Path, Path],
) -> Tuple[Path, Path, Path]:
    args.original_split_paths = {
        "train": str(split_paths[0]),
        "val": str(split_paths[1]),
        "test": str(split_paths[2]),
    }
    if not args.cache_data_dir:
        args.effective_split_paths = dict(args.original_split_paths)
        args.cache_data_enabled = False
        return split_paths

    cache_dir = Path(args.cache_data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_paths: List[Path] = []
    for split_name_value, src in zip(("train", "val", "test"), split_paths):
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = cache_dir / src.name
        same_path = False
        try:
            same_path = src.resolve() == dst.resolve()
        except OSError:
            same_path = False
        if not same_path:
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                print(f"cache hit split={split_name_value} path={dst}", flush=True)
            else:
                print(f"caching split={split_name_value} {src} -> {dst}", flush=True)
                shutil.copy2(src, dst)
        cached_paths.append(dst)

    args.effective_split_paths = {
        "train": str(cached_paths[0]),
        "val": str(cached_paths[1]),
        "test": str(cached_paths[2]),
    }
    args.cache_data_enabled = True
    return tuple(cached_paths)  # type: ignore[return-value]


def parse_batch_candidates(value: str) -> List[int]:
    candidates: List[int] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        batch = int(part)
        if batch <= 0:
            raise ValueError(f"Batch candidates must be positive, got {batch}")
        candidates.append(batch)
    if not candidates:
        raise ValueError("--batch-candidates must contain at least one positive integer")
    return candidates


def cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or ("cuda error" in text and "memory" in text)


def build_training_config(args: argparse.Namespace, split_paths: Tuple[Path, Path, Path]) -> Dict[str, Any]:
    maps = collect_label_maps(split_paths)
    return {
        "backend": "transformer_cross_encoder",
        "encoder_name": args.encoder,
        "max_length": args.max_length,
        "context_before_chars": args.context_before_chars,
        "context_after_chars": args.context_after_chars,
        "numeric_feature_names": list(NUMERIC_FEATURE_NAMES),
        "dropout": args.dropout,
        **maps,
        "license_note": (
            "This Plan1 model may include CC BY-SA and research_only samples. "
            "Do not publish without a separate license audit."
        ),
    }


def build_loaders(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    config: Dict[str, Any],
    split_paths: Tuple[Path, Path, Path],
    device: Any,
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    require_training_deps()
    train_path, val_path, test_path = split_paths
    train_ds = JsonlRankingDataset(train_path)
    val_ds = JsonlRankingDataset(val_path)
    test_ds = JsonlRankingDataset(test_path)
    collate = make_collate_fn(tokenizer, config)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def write_training_manifests(
    *,
    model_root: Path,
    args: argparse.Namespace,
    config: Dict[str, Any],
    split_paths: Tuple[Path, Path, Path],
    datasets: Tuple[Any, Any, Any],
) -> None:
    train_path, val_path, test_path = split_paths
    train_ds, val_ds, test_ds = datasets
    data_profile = config.get("data_profile", {})
    with (model_root / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(config | vars(args), file, ensure_ascii=False, indent=2, default=str)
    with (model_root / "data_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "schema_version": "plan1_training_data_manifest_v1",
                "data_version": data_profile.get("data_version", "unknown"),
                "cleaning_rules_version": data_profile.get("cleaning_rules_version", "unknown"),
                "data_profile": data_profile,
                "train_path": str(train_path),
                "val_path": str(val_path),
                "test_path": str(test_path),
                "original_split_paths": getattr(
                    args,
                    "original_split_paths",
                    {"train": str(train_path), "val": str(val_path), "test": str(test_path)},
                ),
                "effective_split_paths": getattr(
                    args,
                    "effective_split_paths",
                    {"train": str(train_path), "val": str(val_path), "test": str(test_path)},
                ),
                "cache_data_enabled": bool(getattr(args, "cache_data_enabled", False)),
                "cache_data_dir": args.cache_data_dir,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "test_samples": len(test_ds),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def write_cleaning_feedback_report(
    *,
    model_root: Path,
    split: str,
    tag: str,
    metrics: Dict[str, float],
    feedback: Optional[Dict[str, Any]],
    data_profile: Dict[str, Any],
    epoch: Optional[int],
    global_step: int,
) -> Optional[Path]:
    if feedback is None:
        return None
    feedback_dir = model_root / "cleaning_feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CLEANING_FEEDBACK_SCHEMA_VERSION,
        "split": split,
        "tag": tag,
        "epoch": epoch,
        "global_step": int(global_step),
        "data_version": data_profile.get("data_version", "unknown"),
        "cleaning_rules_version": data_profile.get("cleaning_rules_version", "unknown"),
        "data_manifest_path": data_profile.get("data_manifest_path", ""),
        "metrics": metrics,
        **feedback,
    }
    path = feedback_dir / f"{split}_{tag}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
    return path


def build_train_loader(
    *,
    args: argparse.Namespace,
    train_ds: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    device: Any,
    epoch: int,
) -> Any:
    require_training_deps()
    collate = make_collate_fn(tokenizer, config)
    generator = torch.Generator()
    generator.manual_seed(int(args.shuffle_seed) + int(epoch))
    return DataLoader(
        train_ds,
        batch_size=args.per_device_batch,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )


def apply_target_effective_batch(args: argparse.Namespace, *, original_effective_batch: int) -> None:
    target = args.target_effective_batch
    if target is None and args.colab_pro_mode and args.auto_batch_probe:
        target = max(original_effective_batch, 1)
        args.target_effective_batch = target
    if target is None:
        return
    args.grad_accum = max(1, math.ceil(int(target) / max(int(args.per_device_batch), 1)))
    effective = int(args.per_device_batch) * int(args.grad_accum)
    print(
        json.dumps(
            {
                "batch_policy": "target_effective_batch",
                "per_device_batch": args.per_device_batch,
                "grad_accum": args.grad_accum,
                "effective_batch": effective,
                "target_effective_batch": target,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def probe_per_device_batch(
    *,
    args: argparse.Namespace,
    model: Any,
    train_ds: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    device: Any,
) -> int:
    require_training_deps()
    if device.type != "cuda":
        print("auto-batch-probe skipped: CUDA is not available", flush=True)
        return int(args.per_device_batch)

    collate = make_collate_fn(tokenizer, config)
    candidates = parse_batch_candidates(args.batch_candidates)
    model.train()
    selected = int(args.per_device_batch)
    for batch_size in candidates:
        if batch_size > len(train_ds):
            continue
        try:
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            samples = [train_ds[index] for index in range(batch_size)]
            batch = move_batch(collate(samples), device)
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                scores = model_scores(model, batch)
                loss = listwise_loss(scores, batch["group_sizes"], batch["target_indices"])
            loss.backward()
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            selected = batch_size
            print(f"auto-batch-probe selected per_device_batch={selected}", flush=True)
            break
        except RuntimeError as exc:
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cuda_oom(exc):
                print(f"auto-batch-probe OOM at per_device_batch={batch_size}", flush=True)
                continue
            raise
    args.per_device_batch = selected
    return selected


def capture_rng_state() -> Dict[str, Any]:
    require_training_deps()
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    require_training_deps()
    if not state:
        return
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "torch_cpu" in state:
        torch_cpu_state = state["torch_cpu"]
        if isinstance(torch_cpu_state, torch.Tensor):
            torch_cpu_state = torch_cpu_state.detach().to(device="cpu", dtype=torch.uint8)
        else:
            torch_cpu_state = torch.tensor(torch_cpu_state, device="cpu", dtype=torch.uint8)
        torch.set_rng_state(torch_cpu_state)
    cuda_states = state.get("torch_cuda_all")
    if cuda_states is not None and torch.cuda.is_available():
        normalized_cuda_states = []
        for cuda_state in cuda_states:
            if isinstance(cuda_state, torch.Tensor):
                normalized_cuda_states.append(cuda_state.detach().to(device="cpu", dtype=torch.uint8))
            else:
                normalized_cuda_states.append(torch.tensor(cuda_state, device="cpu", dtype=torch.uint8))
        torch.cuda.set_rng_state_all(normalized_cuda_states)


def write_heartbeat(model_root: Path, payload: Dict[str, Any]) -> None:
    heartbeat = {
        **payload,
        "updated_at_unix": time.time(),
        "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (model_root / "heartbeat.json").open("w", encoding="utf-8") as file:
        json.dump(heartbeat, file, ensure_ascii=False, indent=2, default=str)


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def save_training_state(
    *,
    path: Path,
    model_root: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    config: Dict[str, Any],
    args: argparse.Namespace,
    split_paths: Tuple[Path, Path, Path],
    epoch: int,
    batch_index: int,
    global_step: int,
    best_mrr: float,
    best_epoch: int,
    stale_epochs: int,
    reason: str,
) -> None:
    require_training_deps()
    payload = {
        "state_kind": "plan1_exact_training_state",
        "saved_at_unix": time.time(),
        "reason": reason,
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "global_step": int(global_step),
        "best_mrr": float(best_mrr),
        "best_epoch": int(best_epoch),
        "stale_epochs": int(stale_epochs),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "config": config,
        "args": dict(vars(args)),
        "split_paths": {"train": str(split_paths[0]), "val": str(split_paths[1]), "test": str(split_paths[2])},
        "original_split_paths": getattr(args, "original_split_paths", None),
        "effective_split_paths": getattr(args, "effective_split_paths", None),
    }
    atomic_torch_save(payload, path)
    write_heartbeat(
        model_root,
        {
            "state_path": str(path),
            "reason": reason,
            "epoch": epoch,
            "batch_index": batch_index,
            "global_step": global_step,
            "best_mrr": best_mrr,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
        },
    )
    print(f"training state saved: {path} reason={reason}", flush=True)


def torch_load_state(path: Path, device: Any) -> Dict[str, Any]:
    require_training_deps()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def require_training_state_compatible(
    *,
    state: Dict[str, Any],
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    state_config = state.get("config") or {}
    require_checkpoint_compatible(config, state_config)
    for key in ("encoder_name", "max_length", "context_before_chars", "context_after_chars"):
        if state_config.get(key) != config.get(key):
            raise RuntimeError(f"Resume state config mismatch for {key}")
    state_original = state.get("original_split_paths")
    current_original = getattr(args, "original_split_paths", None)
    if state_original and current_original and state_original != current_original:
        raise RuntimeError(
            "Resume state split paths do not match current split paths. "
            f"state={state_original} current={current_original}"
        )
    state_args = state.get("args") or {}
    mismatched_args = []
    for key in RESUME_COMPAT_ARG_KEYS:
        if key in state_args and state_args.get(key) != getattr(args, key, None):
            mismatched_args.append(
                f"{key}: state={state_args.get(key)!r} current={getattr(args, key, None)!r}"
            )
    if mismatched_args:
        raise RuntimeError(
            "Resume state training arguments do not match the current run: "
            + "; ".join(mismatched_args)
        )


def restore_training_state(
    *,
    state_path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    config: Dict[str, Any],
    args: argparse.Namespace,
    device: Any,
) -> Dict[str, Any]:
    state = torch_load_state(state_path, device)
    if state.get("state_kind") != "plan1_exact_training_state":
        raise RuntimeError(f"Unsupported training state kind in {state_path}")
    require_training_state_compatible(state=state, config=config, args=args)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    restore_rng_state(state.get("rng_state") or {})
    print(
        json.dumps(
            {
                "resumed_from": str(state_path),
                "epoch": state.get("epoch"),
                "batch_index": state.get("batch_index"),
                "global_step": state.get("global_step"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return state


def train_one_epoch(
    *,
    model: Any,
    train_loader: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    device: Any,
    grad_accum: int,
    fp16: bool,
    max_grad_norm: float,
    start_batch_index: int = 0,
    checkpoint_callback: Optional[Callable[[int, int, float, int, str], bool]] = None,
) -> Tuple[float, int, int, bool]:
    require_training_deps()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_loss_sum = 0.0
    train_samples = 0
    optimizer_steps = 0
    completed_batches = int(start_batch_index)
    stop_requested = False

    for step, batch in enumerate(train_loader, start=1):
        if step <= start_batch_index:
            continue
        batch = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
            scores = model_scores(model, batch)
            loss = listwise_loss(scores, batch["group_sizes"], batch["target_indices"])
            scaled_loss = loss / grad_accum
        scaler.scale(scaled_loss).backward()
        train_loss_sum += float(loss.detach().item()) * len(batch["group_sizes"])
        train_samples += len(batch["group_sizes"])

        if step % grad_accum == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            completed_batches = step
            if checkpoint_callback is not None:
                loss_so_far = train_loss_sum / max(train_samples, 1)
                stop_requested = checkpoint_callback(
                    completed_batches,
                    optimizer_steps,
                    loss_so_far,
                    train_samples,
                    "periodic",
                )
                if stop_requested:
                    break
        else:
            completed_batches = step

    return train_loss_sum / max(train_samples, 1), optimizer_steps, completed_batches, stop_requested


def resolve_model_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.data_root) / "models" / "plan1-reranker"


def output_has_training_artifacts(model_root: Path) -> bool:
    artifacts = (
        model_root / "metrics.jsonl",
        model_root / "best_metrics.json",
        model_root / "final_test_metrics.json",
        model_root / "best" / "model.safetensors",
        model_root / "best" / "plan1_head.pt",
    )
    return any(path.exists() for path in artifacts)


def read_checkpoint_metrics(checkpoint_dir: Optional[str]) -> Dict[str, Any]:
    if not checkpoint_dir:
        return {}
    metrics_path = Path(checkpoint_dir).parent / "best_metrics.json"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_checkpoint_compatible(current: Dict[str, Any], checkpoint: Dict[str, Any]) -> None:
    keys = ("match_type_to_id", "source_to_id", "domain_to_id", "numeric_feature_names")
    mismatched = [key for key in keys if current.get(key) != checkpoint.get(key)]
    if mismatched:
        raise RuntimeError(
            "Plan1 checkpoint label space does not match the current training data: "
            + ", ".join(mismatched)
        )


def require_checkpoint_expand_compatible(current: Dict[str, Any], checkpoint: Dict[str, Any]) -> None:
    exact_keys = (
        "encoder_name",
        "max_length",
        "context_before_chars",
        "context_after_chars",
        "numeric_feature_names",
    )
    mismatched_exact = [
        key for key in exact_keys if current.get(key) != checkpoint.get(key)
    ]
    if mismatched_exact:
        raise RuntimeError(
            "Plan1 checkpoint cannot be expanded because fixed config differs: "
            + ", ".join(mismatched_exact)
        )

    for map_key in LABEL_EMBEDDING_SPECS:
        current_map = current.get(map_key)
        checkpoint_map = checkpoint.get(map_key)
        if not isinstance(current_map, dict) or not isinstance(checkpoint_map, dict):
            raise RuntimeError(f"Plan1 checkpoint label map is missing or invalid: {map_key}")
        missing = sorted(set(checkpoint_map) - set(current_map))
        if missing:
            raise RuntimeError(
                "Plan1 checkpoint cannot be expanded because current data is missing "
                f"{map_key} labels from the checkpoint: {missing}"
            )


def migrate_label_embedding_weight(
    *,
    current_weight: Any,
    checkpoint_weight: Any,
    current_map: Dict[str, int],
    checkpoint_map: Dict[str, int],
) -> Tuple[Any, Dict[str, Any]]:
    require_training_deps()
    if checkpoint_weight.ndim != 2 or current_weight.ndim != 2:
        raise RuntimeError("Plan1 label embedding weights must be rank-2 tensors")
    if checkpoint_weight.shape[1] != current_weight.shape[1]:
        raise RuntimeError(
            "Plan1 label embedding width mismatch: "
            f"checkpoint={tuple(checkpoint_weight.shape)} current={tuple(current_weight.shape)}"
        )

    migrated = current_weight.detach().cpu().clone()
    unknown_idx = checkpoint_map.get("unknown")
    copied_labels: List[str] = []
    initialized_from_unknown: List[str] = []
    kept_random: List[str] = []
    for label, current_idx in sorted(current_map.items(), key=lambda item: item[1]):
        if label in checkpoint_map:
            migrated[current_idx] = checkpoint_weight[int(checkpoint_map[label])].detach().cpu()
            copied_labels.append(label)
        elif unknown_idx is not None:
            migrated[current_idx] = checkpoint_weight[int(unknown_idx)].detach().cpu()
            initialized_from_unknown.append(label)
        else:
            kept_random.append(label)

    return migrated, {
        "old_count": len(checkpoint_map),
        "new_count": len(current_map),
        "copied_count": len(copied_labels),
        "copied_labels": copied_labels,
        "added_labels": sorted(set(current_map) - set(checkpoint_map)),
        "initialized_from_unknown": initialized_from_unknown,
        "kept_random_initialization": kept_random,
        "unknown_available": unknown_idx is not None,
    }


def migrate_plan1_head_state_for_label_expansion(
    *,
    model: Any,
    checkpoint_head_state: Dict[str, Any],
    current_config: Dict[str, Any],
    checkpoint_config: Dict[str, Any],
    checkpoint_dir: Path,
    policy: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    require_checkpoint_expand_compatible(current_config, checkpoint_config)
    current_state = model.state_dict()
    current_head_keys = {
        key for key in current_state if not key.startswith("encoder.")
    }
    checkpoint_keys = set(checkpoint_head_state)
    missing = sorted(current_head_keys - checkpoint_keys - set(LABEL_EMBEDDING_SPECS.values()))
    unexpected = sorted(checkpoint_keys - current_head_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Plan1 checkpoint head keys do not match for label-space expansion: "
            f"missing={missing} unexpected={unexpected}"
        )

    migrated: Dict[str, Any] = {}
    report: Dict[str, Any] = {
        "schema_version": "plan1_label_space_migration_v1",
        "policy": policy,
        "checkpoint_dir": str(checkpoint_dir),
        "fixed_config": {
            "encoder_name": current_config.get("encoder_name"),
            "max_length": current_config.get("max_length"),
            "context_before_chars": current_config.get("context_before_chars"),
            "context_after_chars": current_config.get("context_after_chars"),
            "numeric_feature_names": current_config.get("numeric_feature_names"),
        },
        "label_maps": {},
        "copied_exact_shape_keys": [],
    }

    for map_key, weight_key in LABEL_EMBEDDING_SPECS.items():
        weight, weight_report = migrate_label_embedding_weight(
            current_weight=current_state[weight_key],
            checkpoint_weight=checkpoint_head_state[weight_key],
            current_map=current_config[map_key],
            checkpoint_map=checkpoint_config[map_key],
        )
        migrated[weight_key] = weight
        report["label_maps"][map_key] = {
            **weight_report,
            "old_map": checkpoint_config[map_key],
            "new_map": current_config[map_key],
        }

    label_weight_keys = set(LABEL_EMBEDDING_SPECS.values())
    for key in sorted(current_head_keys - label_weight_keys):
        old_value = checkpoint_head_state[key]
        new_value = current_state[key]
        if tuple(old_value.shape) != tuple(new_value.shape):
            raise RuntimeError(
                "Plan1 checkpoint head tensor shape mismatch for label-space expansion: "
                f"{key} checkpoint={tuple(old_value.shape)} current={tuple(new_value.shape)}"
            )
        migrated[key] = old_value.detach().cpu()
        report["copied_exact_shape_keys"].append(key)

    return migrated, report


def write_label_space_migration_report(model_root: Path, report: Dict[str, Any]) -> Path:
    path = model_root / "label_space_migration.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=str)
    return path


def load_train_model_and_tokenizer(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: Any,
    model_root: Path,
) -> Tuple[Any, Any, Dict[str, Any]]:
    if not args.init_from_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder)
        encoder = AutoModel.from_pretrained(args.encoder)
        if args.gradient_checkpointing and hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
        model = Plan1RerankerModel(
            encoder,
            match_type_count=len(config["match_type_to_id"]),
            source_count=len(config["source_to_id"]),
            domain_count=len(config["domain_to_id"]),
            numeric_dim=len(NUMERIC_FEATURE_NAMES),
            dropout=args.dropout,
        )
        model.to(device)
        return tokenizer, model, {}

    if args.checkpoint_label_space_policy == "strict":
        checkpoint_session = Plan1InferenceSession.load(args.init_from_checkpoint, device_name=str(device))
        require_checkpoint_compatible(config, checkpoint_session.config)
        if args.gradient_checkpointing and hasattr(
            checkpoint_session.model.encoder, "gradient_checkpointing_enable"
        ):
            checkpoint_session.model.encoder.gradient_checkpointing_enable()
        checkpoint_session.model.train()
        return checkpoint_session.tokenizer, checkpoint_session.model, checkpoint_session.config

    checkpoint_dir = Path(args.init_from_checkpoint)
    config_path = checkpoint_dir / "plan1_config.json"
    head_path = checkpoint_dir / "plan1_head.pt"
    if not config_path.is_file() or not head_path.is_file():
        raise FileNotFoundError(
            f"Plan1 checkpoint requires plan1_config.json and plan1_head.pt in {checkpoint_dir}"
        )
    with config_path.open("r", encoding="utf-8") as file:
        checkpoint_config = json.load(file)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    encoder = AutoModel.from_pretrained(checkpoint_dir)
    if args.gradient_checkpointing and hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable()
    model = Plan1RerankerModel(
        encoder,
        match_type_count=len(config["match_type_to_id"]),
        source_count=len(config["source_to_id"]),
        domain_count=len(config["domain_to_id"]),
        numeric_dim=len(NUMERIC_FEATURE_NAMES),
        dropout=args.dropout,
    )
    head_state = torch.load(head_path, map_location="cpu")
    migrated_head, migration_report = migrate_plan1_head_state_for_label_expansion(
        model=model,
        checkpoint_head_state=head_state,
        current_config=config,
        checkpoint_config=checkpoint_config,
        checkpoint_dir=checkpoint_dir,
        policy=args.checkpoint_label_space_policy,
    )
    missing, unexpected = model.load_state_dict(migrated_head, strict=False)
    missing_non_encoder = [name for name in missing if not name.startswith("encoder.")]
    unexpected_non_encoder = [name for name in unexpected if not name.startswith("encoder.")]
    if missing_non_encoder or unexpected_non_encoder:
        raise RuntimeError(
            "Plan1 migrated head state mismatch "
            f"missing={missing_non_encoder} unexpected={unexpected_non_encoder}"
        )
    model.to(device)
    model.train()
    report_path = write_label_space_migration_report(model_root, migration_report)
    print(f"label space migration written: {report_path}", flush=True)
    return tokenizer, model, checkpoint_config


def write_dry_run_manifest(
    *,
    model_root: Path,
    args: argparse.Namespace,
    datasets: Tuple[Any, Any, Any],
    updates_per_epoch: int,
    total_updates: int,
    warmup_steps: int,
    epoch_offset: int,
    global_step_offset: int,
    checkpoint_metrics: Dict[str, Any],
) -> None:
    train_ds, val_ds, test_ds = datasets
    manifest = {
        "dry_run": True,
        "output_dir": str(model_root),
        "init_from_checkpoint": args.init_from_checkpoint,
        "checkpoint_label_space_policy": args.checkpoint_label_space_policy,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "additional_max_epochs": args.max_epochs,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "warmup_steps": warmup_steps,
        "epoch_offset": epoch_offset,
        "global_step_offset": global_step_offset,
        "checkpoint_metrics": checkpoint_metrics,
        "original_split_paths": getattr(args, "original_split_paths", None),
        "effective_split_paths": getattr(args, "effective_split_paths", None),
        "cache_data_enabled": bool(getattr(args, "cache_data_enabled", False)),
        "cache_data_dir": args.cache_data_dir,
        "colab_pro_mode": args.colab_pro_mode,
        "auto_batch_probe": args.auto_batch_probe,
        "batch_candidates": args.batch_candidates,
        "target_effective_batch": args.target_effective_batch,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "checkpoint_every_seconds": args.checkpoint_every_seconds,
        "max_runtime_seconds": args.max_runtime_seconds,
        "resume_state": args.resume_state,
        "auto_resume": args.auto_resume,
        "shuffle_seed": args.shuffle_seed,
        "cleaning_feedback_enabled": not args.no_cleaning_feedback,
        "cleaning_feedback_schema_version": CLEANING_FEEDBACK_SCHEMA_VERSION,
        "cleaning_feedback_top_n": args.cleaning_feedback_top_n,
        "cleaning_feedback_min_slice_samples": args.cleaning_feedback_min_slice_samples,
    }
    with (model_root / "dry_run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def train(args: argparse.Namespace) -> None:
    require_training_deps()
    model_root = resolve_model_root(args)
    apply_saved_resume_args(args, model_root)
    apply_colab_training_defaults(args)
    model_root = resolve_model_root(args)
    resume_state_path = resolve_resume_state_path(args, model_root)
    resume_requested = bool(args.auto_resume or args.resume_state)
    if output_has_training_artifacts(model_root) and not args.allow_existing_output:
        if resume_requested:
            if not resume_state_path.is_file():
                raise RuntimeError(
                    f"Resume requested, but training state does not exist: {resume_state_path}"
                )
        else:
            raise RuntimeError(
                f"Training output already contains artifacts: {model_root}. "
                "Use --output-dir for an isolated run, --auto-resume, or --allow-existing-output deliberately."
            )
    model_root.mkdir(parents=True, exist_ok=True)
    original_split_paths = resolve_split_paths(args)
    split_paths = cache_split_paths(args, original_split_paths)
    config = build_training_config(args, split_paths)
    data_profile = build_training_data_profile(args, split_paths)
    config["data_profile"] = data_profile
    config["cleaning_feedback_schema_version"] = CLEANING_FEEDBACK_SCHEMA_VERSION
    checkpoint_metrics = read_checkpoint_metrics(args.init_from_checkpoint)
    if args.init_from_checkpoint:
        config["init_from_checkpoint"] = str(Path(args.init_from_checkpoint))
        config["checkpoint_label_space_policy"] = args.checkpoint_label_space_policy
        config["init_kind"] = (
            "warm_start_checkpoint_expand_compatible"
            if args.checkpoint_label_space_policy == "expand-compatible"
            else "warm_start_checkpoint_weights_only"
        )
        config["init_warning"] = (
            "Optimizer, scheduler, and GradScaler state are intentionally reset; "
            "this is not an exact interrupted-run resume."
        )
    if resume_requested:
        config["resume_state"] = str(resume_state_path)
        config["init_kind"] = "exact_training_state_resume"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer, model, checkpoint_config = load_train_model_and_tokenizer(
        args=args,
        config=config,
        device=device,
        model_root=model_root,
    )

    original_effective_batch = max(int(args.per_device_batch) * int(args.grad_accum), 1)
    train_ds_for_probe = JsonlRankingDataset(split_paths[0])
    if args.auto_batch_probe:
        probe_per_device_batch(
            args=args,
            model=model,
            train_ds=train_ds_for_probe,
            tokenizer=tokenizer,
            config=config,
            device=device,
        )
    apply_target_effective_batch(args, original_effective_batch=original_effective_batch)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        args=args,
        tokenizer=tokenizer,
        config=config,
        split_paths=split_paths,
        device=device,
    )
    write_training_manifests(
        model_root=model_root,
        args=args,
        config=config,
        split_paths=split_paths,
        datasets=(train_ds, val_ds, test_ds),
    )

    grad_accum = max(args.grad_accum, 1)
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_updates = max(updates_per_epoch * args.max_epochs, 1)
    warmup_steps = max(1, int(total_updates * args.warmup_ratio))
    epoch_offset = (
        args.epoch_offset
        if args.epoch_offset is not None
        else int(checkpoint_metrics.get("epoch", 0) or 0)
    )
    global_step = (
        args.global_step_offset
        if args.global_step_offset is not None
        else int(checkpoint_metrics.get("global_step", 0) or 0)
    )
    if args.dry_run:
        write_dry_run_manifest(
            model_root=model_root,
            args=args,
            datasets=(train_ds, val_ds, test_ds),
            updates_per_epoch=updates_per_epoch,
            total_updates=total_updates,
            warmup_steps=warmup_steps,
            epoch_offset=epoch_offset,
            global_step_offset=global_step,
            checkpoint_metrics=checkpoint_metrics,
        )
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    metrics_path = model_root / "metrics.jsonl"
    best_metrics_path = model_root / "best_metrics.json"
    best_mrr = -1.0
    best_epoch = 0
    stale_epochs = 0
    start_epoch = int(epoch_offset) + 1
    resume_batch_index = 0

    if resume_requested:
        state = restore_training_state(
            state_path=resume_state_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            args=args,
            device=device,
        )
        start_epoch = int(state["epoch"])
        resume_batch_index = int(state.get("batch_index", 0) or 0)
        global_step = int(state.get("global_step", global_step) or 0)
        best_mrr = float(state.get("best_mrr", best_mrr))
        best_epoch = int(state.get("best_epoch", best_epoch) or 0)
        stale_epochs = int(state.get("stale_epochs", stale_epochs) or 0)

    state_path = model_root / "last_state.pt"
    run_started_at = time.perf_counter()
    last_checkpoint_global_step = int(global_step)
    last_checkpoint_time = time.perf_counter()
    current_epoch = start_epoch
    current_batch_index = resume_batch_index
    end_epoch = int(epoch_offset) + int(args.max_epochs)

    def save_state(
        *,
        epoch: int,
        batch_index: int,
        state_global_step: int,
        reason: str,
    ) -> None:
        save_training_state(
            path=state_path,
            model_root=model_root,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            args=args,
            split_paths=split_paths,
            epoch=epoch,
            batch_index=batch_index,
            global_step=state_global_step,
            best_mrr=best_mrr,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            reason=reason,
        )

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            current_epoch = epoch
            started_at = time.perf_counter()
            train_loader = build_train_loader(
                args=args,
                train_ds=train_ds,
                tokenizer=tokenizer,
                config=config,
                device=device,
                epoch=epoch,
            )
            epoch_start_global_step = int(global_step)
            epoch_start_batch = resume_batch_index if epoch == start_epoch else 0

            def checkpoint_callback(
                batch_index: int,
                optimizer_steps_so_far: int,
                train_loss_so_far: float,
                train_samples_so_far: int,
                reason: str,
            ) -> bool:
                nonlocal last_checkpoint_global_step, last_checkpoint_time, current_batch_index
                current_batch_index = batch_index
                state_global_step = epoch_start_global_step + optimizer_steps_so_far
                now = time.perf_counter()
                step_due = (
                    args.checkpoint_every_steps
                    and state_global_step - last_checkpoint_global_step >= args.checkpoint_every_steps
                )
                time_due = (
                    args.checkpoint_every_seconds
                    and now - last_checkpoint_time >= args.checkpoint_every_seconds
                )
                runtime_due = bool(args.max_runtime_seconds and now - run_started_at >= args.max_runtime_seconds)
                if step_due or time_due or runtime_due:
                    save_state(
                        epoch=epoch,
                        batch_index=batch_index,
                        state_global_step=state_global_step,
                        reason="runtime_limit" if runtime_due else reason,
                    )
                    last_checkpoint_global_step = state_global_step
                    last_checkpoint_time = now
                return runtime_due

            train_loss, optimizer_steps, completed_batches, stop_requested = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                grad_accum=grad_accum,
                fp16=args.fp16,
                max_grad_norm=args.max_grad_norm,
                start_batch_index=epoch_start_batch,
                checkpoint_callback=checkpoint_callback,
            )
            global_step += optimizer_steps
            current_batch_index = completed_batches
            save_state(
                epoch=epoch,
                batch_index=completed_batches,
                state_global_step=global_step,
                reason="runtime_limit" if stop_requested else "epoch_train_complete",
            )
            if stop_requested:
                print(
                    json.dumps(
                        {
                            "stopped": "runtime_limit",
                            "epoch": epoch,
                            "batch_index": completed_batches,
                            "global_step": global_step,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return

            val_metrics, val_feedback = evaluate_with_cleaning_feedback(
                model,
                val_loader,
                device,
                collect_feedback=not args.no_cleaning_feedback,
                feedback_top_n=args.cleaning_feedback_top_n,
                feedback_min_slice_samples=args.cleaning_feedback_min_slice_samples,
            )
            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "elapsed_seconds": time.perf_counter() - started_at,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            print(json.dumps(epoch_metrics, ensure_ascii=False), flush=True)
            feedback_path = write_cleaning_feedback_report(
                model_root=model_root,
                split="val",
                tag=f"epoch_{epoch:04d}",
                metrics=val_metrics,
                feedback=val_feedback,
                data_profile=data_profile,
                epoch=epoch,
                global_step=global_step,
            )
            if feedback_path is not None:
                print(f"cleaning feedback written: {feedback_path}", flush=True)

            if val_metrics["mrr"] > best_mrr:
                best_mrr = val_metrics["mrr"]
                best_epoch = epoch
                stale_epochs = 0
                best_dir = model_root / "best"
                save_plan1_checkpoint(model, tokenizer, best_dir, config)
                with best_metrics_path.open("w", encoding="utf-8") as file:
                    json.dump(epoch_metrics, file, ensure_ascii=False, indent=2)
            else:
                stale_epochs += 1
            save_state(
                epoch=epoch + 1,
                batch_index=0,
                state_global_step=global_step,
                reason="epoch_complete",
            )
            resume_batch_index = 0
            if epoch >= args.min_epochs and stale_epochs >= args.patience:
                break
    except KeyboardInterrupt:
        save_state(
            epoch=current_epoch,
            batch_index=current_batch_index,
            state_global_step=global_step,
            reason="keyboard_interrupt",
        )
        raise

    best_dir = model_root / "best"
    if best_dir.is_dir():
        session = Plan1InferenceSession.load(best_dir, device_name=str(device))
        model = session.model
    test_metrics, test_feedback = evaluate_with_cleaning_feedback(
        model,
        test_loader,
        device,
        collect_feedback=not args.no_cleaning_feedback,
        feedback_top_n=args.cleaning_feedback_top_n,
        feedback_min_slice_samples=args.cleaning_feedback_min_slice_samples,
    )
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_mrr": best_mrr,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    with (model_root / "final_test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(final_metrics, file, ensure_ascii=False, indent=2)
    feedback_path = write_cleaning_feedback_report(
        model_root=model_root,
        split="test",
        tag="final",
        metrics=test_metrics,
        feedback=test_feedback,
        data_profile=data_profile,
        epoch=None,
        global_step=global_step,
    )
    if feedback_path is not None:
        final_metrics["test_cleaning_feedback_path"] = str(feedback_path)
        with (model_root / "final_test_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(final_metrics, file, ensure_ascii=False, indent=2)
        print(f"cleaning feedback written: {feedback_path}", flush=True)
    print(json.dumps(final_metrics, ensure_ascii=False), flush=True)


# -----------------------------
# CLI
# -----------------------------


def add_build_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=str(Hyperparameters.data_root))
    parser.add_argument("--output-prefix", default="plan1_ranking")
    parser.add_argument("--max-candidates", type=int, default=Hyperparameters.max_candidates)
    parser.add_argument("--max-generated", type=int, default=0, help="0 means full corpus")
    parser.add_argument(
        "--max-generated-per-corpus",
        type=int,
        default=0,
        help="0 means no per-corpus cap; useful when building balanced multi-corpus data.",
    )
    parser.add_argument("--max-docs", type=int, default=0, help="0 means full corpus")
    parser.add_argument(
        "--corpus-file",
        action="append",
        default=None,
        help=(
            "Corpus JSONL path or filename under DATA_ROOT/processed. "
            "Can be repeated; defaults to zhwiki_corpus.jsonl."
        ),
    )
    parser.add_argument(
        "--train-split-percent",
        type=int,
        default=Hyperparameters.train_split_percent,
        help="Stable source_doc_id hash split percentage for training data.",
    )
    parser.add_argument(
        "--val-split-percent",
        type=int,
        default=Hyperparameters.val_split_percent,
        help="Stable source_doc_id hash split percentage for validation data.",
    )
    parser.add_argument(
        "--test-split-percent",
        type=int,
        default=Hyperparameters.test_split_percent,
        help="Stable source_doc_id hash split percentage for test data.",
    )
    parser.add_argument("--skip-corpus-generation", action="store_true")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=str(Hyperparameters.data_root))
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training output directory. Defaults to DATA_ROOT/models/plan1-reranker.",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        default=None,
        help="Warm-start model weights and tokenizer from a Plan1 checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-label-space-policy",
        choices=CHECKPOINT_LABEL_SPACE_POLICIES,
        default="strict",
        help=(
            "How to handle checkpoint label maps during warm-start. "
            "strict requires exact label maps; expand-compatible copies labels by name "
            "and initializes new labels from the old unknown row."
        ),
    )
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=None,
        help="Epoch number to continue from. Defaults to sibling best_metrics.json when available.",
    )
    parser.add_argument(
        "--global-step-offset",
        type=int,
        default=None,
        help="Global step to continue from. Defaults to sibling best_metrics.json when available.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an output directory that already contains training artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/model loading and write manifests without starting optimization.",
    )
    parser.add_argument("--encoder", default=Hyperparameters.encoder)
    parser.add_argument("--max-length", type=int, default=Hyperparameters.max_length)
    parser.add_argument("--context-before-chars", type=int, default=Hyperparameters.context_before_chars)
    parser.add_argument("--context-after-chars", type=int, default=Hyperparameters.context_after_chars)
    parser.add_argument("--per-device-batch", type=int, default=Hyperparameters.per_device_batch)
    parser.add_argument("--eval-batch", type=int, default=Hyperparameters.eval_batch)
    parser.add_argument("--grad-accum", type=int, default=Hyperparameters.grad_accum)
    parser.add_argument("--max-epochs", type=int, default=Hyperparameters.max_epochs)
    parser.add_argument("--min-epochs", type=int, default=Hyperparameters.min_epochs)
    parser.add_argument("--patience", type=int, default=Hyperparameters.patience)
    parser.add_argument("--lr", type=float, default=Hyperparameters.lr)
    parser.add_argument("--weight-decay", type=float, default=Hyperparameters.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=Hyperparameters.warmup_ratio)
    parser.add_argument("--max-grad-norm", type=float, default=Hyperparameters.max_grad_norm)
    parser.add_argument("--dropout", type=float, default=Hyperparameters.dropout)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--colab-pro-mode",
        action="store_true",
        help=(
            "Enable Colab Pro friendly defaults: local data cache, periodic exact-state "
            "checkpoints, and runtime-limit graceful exit when paired with the related flags."
        ),
    )
    parser.add_argument(
        "--cache-data-dir",
        default=None,
        help="Copy train/val/test JSONL files into this local directory before training.",
    )
    parser.add_argument(
        "--auto-batch-probe",
        action="store_true",
        help="On CUDA, probe candidate per-device batch sizes with one forward/backward pass.",
    )
    parser.add_argument(
        "--batch-candidates",
        default=DEFAULT_COLAB_BATCH_CANDIDATES,
        help="Comma-separated per-device batch candidates, tested from largest to smallest.",
    )
    parser.add_argument(
        "--target-effective-batch",
        type=int,
        default=None,
        help="Adjust grad accumulation so per_device_batch * grad_accum is at least this value.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help="Save last_state.pt after this many optimizer steps; 0 disables step interval.",
    )
    parser.add_argument(
        "--checkpoint-every-seconds",
        type=float,
        default=0.0,
        help="Save last_state.pt after this many seconds; 0 disables time interval.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="Save exact state and exit when this runtime is reached; 0 disables runtime guard.",
    )
    parser.add_argument(
        "--resume-state",
        default=None,
        help="Resume from a full training state file such as output_dir/last_state.pt.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from output_dir/last_state.pt and reuse saved training arguments when available.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=1337,
        help="Base seed for deterministic epoch shuffling, needed for exact mid-epoch resume.",
    )
    parser.add_argument(
        "--no-cleaning-feedback",
        action="store_true",
        help="Disable validation/test cleaning feedback reports.",
    )
    parser.add_argument(
        "--cleaning-feedback-top-n",
        type=int,
        default=50,
        help="Number of anonymized high-loss samples to keep per feedback report.",
    )
    parser.add_argument(
        "--cleaning-feedback-min-slice-samples",
        type=int,
        default=25,
        help="Minimum samples required before reporting a slice metric.",
    )


def add_audit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=str(Hyperparameters.data_root))
    parser.add_argument("--input-prefix", default="plan1_ranking_multi_1m_90_5_5")


def add_clean_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=str(Hyperparameters.data_root))
    parser.add_argument("--input-prefix", default="plan1_ranking_multi_1m_90_5_5")
    parser.add_argument("--output-prefix", default="plan1_ranking_colab_ready")
    parser.add_argument("--max-candidates", type=int, default=Hyperparameters.max_candidates)
    parser.add_argument("--target-train-samples", type=int, default=930000)
    parser.add_argument("--target-val-samples", type=int, default=55000)
    parser.add_argument("--target-test-samples", type=int, default=55000)
    parser.add_argument("--min-train-samples", type=int, default=900000)
    parser.add_argument("--min-val-samples", type=int, default=50000)
    parser.add_argument("--min-test-samples", type=int, default=50000)
    parser.add_argument(
        "--corpus-file",
        action="append",
        default=None,
        help="Corpus JSONL for supplementation. Can be repeated.",
    )
    parser.add_argument(
        "--train-split-percent",
        type=int,
        default=Hyperparameters.train_split_percent,
    )
    parser.add_argument(
        "--val-split-percent",
        type=int,
        default=Hyperparameters.val_split_percent,
    )
    parser.add_argument(
        "--test-split-percent",
        type=int,
        default=Hyperparameters.test_split_percent,
    )


def add_package_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=str(Hyperparameters.data_root))
    parser.add_argument("--input-prefix", default="plan1_ranking_colab_ready_20260602")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan1 golf IME reranker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-data", help="build Plan1 ranking jsonl files")
    add_build_data_args(build_parser)

    train_parser = subparsers.add_parser("train", help="train Plan1 reranker")
    add_train_args(train_parser)

    audit_parser = subparsers.add_parser("audit-data", help="audit Plan1 ranking data")
    add_audit_args(audit_parser)

    clean_parser = subparsers.add_parser("clean-data", help="clean and re-split Plan1 ranking data")
    add_clean_args(clean_parser)

    package_parser = subparsers.add_parser("package-data", help="package Plan1 data for export")
    add_package_args(package_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-data":
        build_plan1_data(args)
    elif args.command == "train":
        train(args)
    elif args.command == "audit-data":
        audit_plan1_data(args)
    elif args.command == "clean-data":
        clean_plan1_data(args)
    elif args.command == "package-data":
        package_plan1_data(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
