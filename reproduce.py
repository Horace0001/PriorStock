from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FACTOR_RUNNER = ROOT / "scripts" / "run_train_and_evaluate_ohlcv124_group_token_mixer_attention_side_adapter_factor_concat_logit_classifier_single_logit.py"
NOISE_RUNNER = ROOT / "scripts" / "run_train_and_evaluate_ohlcv124_group_token_mixer_attention_side_adapter_factor_concat_logit_classifier_noise_aware_single_logit.py"


def run_stage(runner: Path, config: Path, output: Path) -> None:
    summary = output / "pipeline_summary.json"
    if summary.exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(runner), "--config-file", str(config), "--run-directory", str(output)],
        cwd=ROOT,
        check=True,
    )


def read_result(market: str) -> dict[str, float]:
    observed = json.loads((ROOT / "runs" / market / "noise" / "test_full_metrics.json").read_text(encoding="utf-8"))
    return {"accuracy": float(observed["accuracy"]), "mcc": float(observed["mcc"])}


def reproduce_market(market: str) -> dict[str, float]:
    run_root = ROOT / "runs" / market
    run_stage(FACTOR_RUNNER, ROOT / "configs" / f"{market}_factor.yaml", run_root / "factor")
    run_stage(NOISE_RUNNER, ROOT / "configs" / f"{market}_noise.yaml", run_root / "noise")
    return read_result(market)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("all", "cmin_us", "cmin_cn"), default="all")
    arguments = parser.parse_args()
    markets = ("cmin_us", "cmin_cn") if arguments.market == "all" else (arguments.market,)
    results = {market: reproduce_market(market) for market in markets}
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
