"""Read-only watcher for Plan1 Colab/Drive training runs.

The watcher does not mutate the run directory. It summarizes heartbeat,
metrics, checkpoints, final results, and cleaning feedback so a local agent can
supervise a Colab Pro run from a synced Google Drive folder.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


METRIC_KEYS = (
    "val_loss",
    "val_top1",
    "val_top3",
    "val_top5",
    "val_mrr",
    "val_ndcg10",
)
WATCHED_FEEDBACK_FIELDS = ("sample_source", "license_bucket", "domain")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def age_seconds(path: Path, now: float) -> Optional[float]:
    if not path.exists():
        return None
    return max(now - path.stat().st_mtime, 0.0)


def format_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "missing"
    if seconds < 120:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 120:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"


def is_bad_number(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return not math.isfinite(float(value))


def latest_file(paths: Iterable[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def last_log_lines(path: Path, count: int = 5) -> List[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()
    return [line.rstrip("\n") for line in lines[-count:]]


def summarize_metrics(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics_rows:
        return {"available": False}
    latest = metrics_rows[-1]
    summary = {
        "available": True,
        "epochs": len(metrics_rows),
        "latest": latest,
        "bad_values": [
            key for row in metrics_rows for key, value in row.items() if is_bad_number(value)
        ],
        "degraded": False,
        "degradation_note": "",
    }
    if len(metrics_rows) >= 2:
        prev = metrics_rows[-2]
        latest_mrr = float(latest.get("val_mrr", 0.0) or 0.0)
        prev_mrr = float(prev.get("val_mrr", 0.0) or 0.0)
        latest_loss = float(latest.get("val_loss", 0.0) or 0.0)
        prev_loss = float(prev.get("val_loss", 0.0) or 0.0)
        if latest_mrr + 0.01 < prev_mrr or latest_loss > prev_loss + 0.05:
            summary["degraded"] = True
            summary["degradation_note"] = (
                f"val_mrr {prev_mrr:.4f}->{latest_mrr:.4f}, "
                f"val_loss {prev_loss:.4f}->{latest_loss:.4f}"
            )
    return summary


def collect_worst_slices(feedback: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    overall = feedback.get("metrics") or {}
    overall_top1 = float(overall.get("top1", overall.get("val_top1", 0.0)) or 0.0)
    overall_loss = float(overall.get("loss", overall.get("val_loss", 0.0)) or 0.0)
    rows: List[Dict[str, Any]] = []
    slice_metrics = feedback.get("slice_metrics") or {}
    for field in WATCHED_FEEDBACK_FIELDS:
        values = slice_metrics.get(field) or {}
        for value, metrics in values.items():
            samples = float(metrics.get("samples", 0.0) or 0.0)
            top1 = float(metrics.get("top1", 0.0) or 0.0)
            loss = float(metrics.get("loss", 0.0) or 0.0)
            rows.append(
                {
                    "field": field,
                    "value": value,
                    "samples": int(samples),
                    "top1": top1,
                    "loss": loss,
                    "top1_gap": overall_top1 - top1,
                    "loss_gap": loss - overall_loss,
                }
            )
    rows.sort(key=lambda item: (item["top1_gap"], item["loss_gap"], item["samples"]), reverse=True)
    return rows[:limit]


def analyze_run(run_dir: Path, stale_minutes: float) -> Dict[str, Any]:
    now = time.time()
    heartbeat_path = run_dir / "heartbeat.json"
    metrics_path = run_dir / "metrics.jsonl"
    final_path = run_dir / "final_test_metrics.json"
    last_state_path = run_dir / "last_state.pt"
    train_log_path = run_dir / "train.log"
    feedback_dir = run_dir / "cleaning_feedback"
    latest_feedback_path = latest_file(feedback_dir.glob("*.json")) if feedback_dir.is_dir() else None

    heartbeat = read_json(heartbeat_path)
    metrics_rows = read_jsonl(metrics_path)
    metrics_summary = summarize_metrics(metrics_rows)
    final_metrics = read_json(final_path)
    latest_feedback = read_json(latest_feedback_path) if latest_feedback_path else None

    stale_seconds = stale_minutes * 60.0
    alerts: List[str] = []
    heartbeat_age = age_seconds(heartbeat_path, now)
    log_age = age_seconds(train_log_path, now)
    state_age = age_seconds(last_state_path, now)
    if heartbeat_age is None:
        alerts.append("heartbeat.json missing")
    elif heartbeat_age > stale_seconds:
        alerts.append(f"heartbeat stale: {format_age(heartbeat_age)}")
    if log_age is None:
        alerts.append("train.log missing")
    elif log_age > stale_seconds:
        alerts.append(f"train.log stale: {format_age(log_age)}")
    if state_age is None:
        alerts.append("last_state.pt missing")
    elif state_age > stale_seconds:
        alerts.append(f"last_state.pt stale: {format_age(state_age)}")
    if metrics_summary.get("bad_values"):
        alerts.append("metrics contain NaN/Inf values")
    if metrics_summary.get("degraded"):
        alerts.append(f"validation degraded: {metrics_summary.get('degradation_note')}")

    worst_slices = collect_worst_slices(latest_feedback) if latest_feedback else []
    for item in worst_slices:
        if item["samples"] >= 25 and (item["top1_gap"] >= 0.05 or item["loss_gap"] >= 0.2):
            alerts.append(
                "weak cleaning slice: "
                f"{item['field']}={item['value']} samples={item['samples']} "
                f"top1={item['top1']:.4f} loss={item['loss']:.4f}"
            )

    return {
        "run_dir": str(run_dir),
        "heartbeat": heartbeat,
        "heartbeat_age": format_age(heartbeat_age),
        "train_log_age": format_age(log_age),
        "last_state_age": format_age(state_age),
        "last_state_bytes": last_state_path.stat().st_size if last_state_path.is_file() else 0,
        "metrics": metrics_summary,
        "final_metrics": final_metrics,
        "latest_feedback_path": str(latest_feedback_path) if latest_feedback_path else "",
        "worst_slices": worst_slices,
        "last_log_lines": last_log_lines(train_log_path),
        "alerts": alerts,
    }


def print_report(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print("Plan1 Colab Watch")
    print("=" * 72)
    print(f"run_dir: {report['run_dir']}")
    heartbeat = report.get("heartbeat") or {}
    if heartbeat:
        print(
            "heartbeat: "
            f"reason={heartbeat.get('reason')} epoch={heartbeat.get('epoch')} "
            f"batch_index={heartbeat.get('batch_index')} global_step={heartbeat.get('global_step')} "
            f"age={report['heartbeat_age']}"
        )
    else:
        print("heartbeat: missing")
    print(
        f"last_state: age={report['last_state_age']} "
        f"bytes={report['last_state_bytes']}"
    )
    print(f"train_log: age={report['train_log_age']}")

    metrics = report.get("metrics") or {}
    if metrics.get("available"):
        latest = metrics.get("latest") or {}
        metric_text = " ".join(
            f"{key}={float(latest[key]):.4f}"
            for key in METRIC_KEYS
            if key in latest and isinstance(latest[key], (int, float))
        )
        print(
            f"metrics: epochs={metrics.get('epochs')} "
            f"epoch={latest.get('epoch')} global_step={latest.get('global_step')} {metric_text}"
        )
    else:
        print("metrics: missing")

    final_metrics = report.get("final_metrics")
    if final_metrics:
        print(
            "final: "
            f"best_epoch={final_metrics.get('best_epoch')} "
            f"best_val_mrr={final_metrics.get('best_val_mrr')} "
            f"test_top1={final_metrics.get('test_top1')} "
            f"test_mrr={final_metrics.get('test_mrr')}"
        )

    if report.get("latest_feedback_path"):
        print(f"latest_feedback: {report['latest_feedback_path']}")
        for item in report.get("worst_slices", [])[:5]:
            print(
                "  weak-slice-candidate: "
                f"{item['field']}={item['value']} samples={item['samples']} "
                f"top1={item['top1']:.4f} loss={item['loss']:.4f} "
                f"top1_gap={item['top1_gap']:.4f}"
            )

    alerts = report.get("alerts") or []
    if alerts:
        print("alerts:")
        for alert in alerts:
            print(f"- {alert}")
    else:
        print("alerts: none")

    log_lines = report.get("last_log_lines") or []
    if log_lines:
        print("last train.log lines:")
        for line in log_lines:
            print(f"  {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a synced Plan1 Colab training run")
    parser.add_argument("--run-dir", required=True, help="Synced Google Drive run directory")
    parser.add_argument("--once", action="store_true", help="Print one report and exit")
    parser.add_argument("--stale-minutes", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    while True:
        report = analyze_run(run_dir, args.stale_minutes)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            print_report(report)
        if args.once:
            break
        time.sleep(max(float(args.poll_seconds), 1.0))


if __name__ == "__main__":
    main()
