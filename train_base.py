from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "scripts" / "run_train_and_evaluate_ohlcv124_group_token_mixer_attention_side_adapter_single_logit.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("cmin_us", "cmin_cn"), required=True)
    arguments = parser.parse_args()
    output = ROOT / "runs" / arguments.market / "base_training"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(RUNNER), "--config-file", str(ROOT / "configs" / f"{arguments.market}_base.yaml"), "--run-directory", str(output)],
        cwd=ROOT,
        check=True,
    )
    summary = json.loads((output / "pipeline_summary.json").read_text(encoding="utf-8"))
    metrics = summary["test_unfiltered_metrics"]
    result = {"accuracy": float(metrics["accuracy"]), "mcc": float(metrics["mcc"])}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
