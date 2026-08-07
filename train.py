from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RUNNERS = {
    "base": ROOT / "scripts" / "base.py",
    "factor": ROOT / "scripts" / "factor.py",
    "noise_aware": ROOT / "scripts" / "noise_aware.py",
}
CONFIG_NAMES = {"base": "base", "factor": "factor", "noise_aware": "noise"}


def run_stage(market: str, stage: str) -> None:
    run_directory = ROOT / "runs" / market / stage
    summary_file_path = run_directory / "pipeline_summary.json"
    if summary_file_path.exists():
        return
    run_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(RUNNERS[stage]),
            "--config-file",
            str(ROOT / "configs" / f"{market}_{CONFIG_NAMES[stage]}.yaml"),
            "--run-directory",
            str(run_directory),
        ],
        cwd=ROOT,
        check=True,
    )


def train_market(market: str) -> dict[str, float]:
    for stage in ("base", "factor", "noise_aware"):
        run_stage(market, stage)
    metrics = json.loads(
        (ROOT / "runs" / market / "noise_aware" / "test_full_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "accuracy": float(metrics["accuracy"]),
        "mcc": float(metrics["mcc"]),
        "macro_f1": float(metrics["macro_f1"]),
        "roc_auc": float(metrics["roc_auc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("all", "cmin_us", "cmin_cn"), default="all")
    arguments = parser.parse_args()
    markets = ("cmin_us", "cmin_cn") if arguments.market == "all" else (arguments.market,)
    results = {market: train_market(market) for market in markets}
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
