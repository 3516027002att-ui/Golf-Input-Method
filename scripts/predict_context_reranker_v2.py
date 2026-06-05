from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.context_reranker_v2 import (  # noqa: E402
    Defaults,
    build_pair_text,
    load_checkpoint,
    require_training_deps,
    validate_choice,
    CONTEXT_MODES,
    CONTEXT_PERTURBS,
)


def parse_candidates(candidate_args: Sequence[str] | None, candidates_json: str | None) -> List[str]:
    candidates: List[str] = []
    if candidates_json:
        loaded = json.loads(candidates_json)
        if not isinstance(loaded, list):
            raise ValueError("--candidates-json must decode to a JSON list")
        candidates.extend(str(item).strip() for item in loaded)
    if candidate_args:
        candidates.extend(str(item).strip() for item in candidate_args)
    candidates = [item for item in candidates if item]
    if not candidates:
        raise ValueError("Provide at least one --candidate or --candidates-json item")
    return candidates


def load_text_config(checkpoint: Path) -> Dict[str, Any]:
    config_path = checkpoint / "context_reranker_config.json"
    if not config_path.is_file():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_prediction_texts(
    *,
    context_before: str,
    composing: str,
    candidates: Sequence[str],
    config: Dict[str, Any],
    context_mode: str = "online",
    context_perturb: str = "none",
) -> List[str]:
    context_mode = validate_choice(context_mode, CONTEXT_MODES, "context_mode")
    context_perturb = validate_choice(context_perturb, CONTEXT_PERTURBS, "context_perturb")
    before_chars = int(config.get("context_before_chars", Defaults.context_before_chars))
    after_chars = int(config.get("context_after_chars", Defaults.context_after_chars))
    return [
        build_pair_text(
            context_before=context_before,
            context_after="",
            composing=composing,
            candidate=candidate,
            before_chars=before_chars,
            after_chars=after_chars,
            include_context=context_mode != "none",
            context_mode=context_mode,
            context_perturb=context_perturb,
            sample_id=f"predict:{index}",
        )
        for index, candidate in enumerate(candidates)
    ]


def score_texts(args: argparse.Namespace, texts: Sequence[str]) -> List[float]:
    require_training_deps()
    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, config = load_checkpoint(Path(args.checkpoint), device)
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(config.get("max_length", Defaults.max_length)),
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        scores = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids"),
        )
    return [float(score) for score in scores.detach().cpu().tolist()]


def rank_candidates(candidates: Sequence[str], scores: Sequence[float]) -> List[Dict[str, Any]]:
    if len(candidates) != len(scores):
        raise ValueError(f"Got {len(scores)} scores for {len(candidates)} candidates")
    rows = [
        {"candidate": candidate, "score": float(score), "input_index": index}
        for index, (candidate, score) in enumerate(zip(candidates, scores))
    ]
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def run_prediction(
    args: argparse.Namespace,
    *,
    scorer: Callable[[argparse.Namespace, Sequence[str]], Sequence[float]] = score_texts,
) -> Dict[str, Any]:
    candidates = parse_candidates(args.candidate, args.candidates_json)
    config = load_text_config(Path(args.checkpoint))
    texts = build_prediction_texts(
        context_before=args.context_before,
        composing=args.composing,
        candidates=candidates,
        config=config,
        context_mode=args.context_mode,
        context_perturb=args.context_perturb,
    )
    scores = list(scorer(args, texts))
    return {
        "checkpoint": str(args.checkpoint),
        "context_mode": args.context_mode,
        "context_perturb": args.context_perturb,
        "side_features_used_by_model": False,
        "ranked_candidates": rank_candidates(candidates, scores),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline prediction for context reranker v2")
    parser.add_argument("--checkpoint", required=True, help="Directory containing a context reranker v2 checkpoint")
    parser.add_argument("--context-before", required=True, help="Text before the cursor")
    parser.add_argument("--composing", required=True, help="Current composing pinyin/input string")
    parser.add_argument("--candidate", action="append", help="Candidate text; repeat for multiple candidates")
    parser.add_argument("--candidates-json", help="JSON list of candidate strings")
    parser.add_argument("--device", help="cpu, cuda, or another torch device")
    parser.add_argument("--context-mode", choices=sorted(CONTEXT_MODES), default="online")
    parser.add_argument("--context-perturb", choices=sorted(CONTEXT_PERTURBS), default="none")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_prediction(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
