from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from priorstock.utils.io import write_json_file
from priorstock.utils.seed import set_global_seed
from scripts.factor import _build_data_loader, _build_model
from scripts.noise_aware import _evaluate_checkpoint, _load_config_with_noise_aware_extensions


ROOT = Path(__file__).resolve().parent


def reproduce_market(market: str) -> dict[str, dict[str, float]]:
    run_directory = ROOT / "runs" / market / "quick_reproduction"
    run_directory.mkdir(parents=True, exist_ok=True)
    config_file_path = ROOT / "configs" / f"{market}_noise.yaml"
    (
        experiment_config,
        attention_side_adapter_config,
        objective_config,
        factor_classifier_config,
        noise_aware_config,
    ) = _load_config_with_noise_aware_extensions(config_file_path, run_directory)
    set_global_seed(experiment_config.experiment.random_seed)
    factor_classifier_config = replace(
        factor_classifier_config,
        base_checkpoint_file_path=str(ROOT / "weights" / market / "base.pt"),
    )
    full_objective_config = replace(objective_config, significant_return_absolute_threshold=0.0)
    train_significant_loader = _build_data_loader(
        experiment_config,
        objective_config,
        factor_classifier_config,
        "train",
        False,
    )
    test_significant_loader = _build_data_loader(
        experiment_config,
        objective_config,
        factor_classifier_config,
        "test",
        False,
    )
    test_full_loader = _build_data_loader(
        experiment_config,
        full_objective_config,
        factor_classifier_config,
        "test",
        False,
    )
    checkpoint_file_path = ROOT / "weights" / market / "noise_aware.pt"

    def build_model():
        return _build_model(
            experiment_config,
            attention_side_adapter_config,
            factor_classifier_config,
        )

    test_significant_metrics = _evaluate_checkpoint(
        checkpoint_file_path,
        build_model(),
        test_significant_loader,
        train_significant_loader,
        experiment_config,
        objective_config,
        noise_aware_config,
        "test_significant",
        run_directory,
        None,
    )
    test_full_metrics = _evaluate_checkpoint(
        checkpoint_file_path,
        build_model(),
        test_full_loader,
        train_significant_loader,
        experiment_config,
        full_objective_config,
        noise_aware_config,
        "test_full",
        run_directory,
        None,
    )
    result = {
        "test_significant": test_significant_metrics,
        "test_full": test_full_metrics,
    }
    write_json_file(run_directory / "pipeline_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("all", "cmin_us", "cmin_cn"), default="all")
    arguments = parser.parse_args()
    markets = ("cmin_us", "cmin_cn") if arguments.market == "all" else (arguments.market,)
    results = {market: reproduce_market(market) for market in markets}
    compact_results = {
        market: {
            split_name: {
                "accuracy": float(metrics["accuracy"]),
                "mcc": float(metrics["mcc"]),
                "macro_f1": float(metrics["macro_f1"]),
                "roc_auc": float(metrics["roc_auc"]),
            }
            for split_name, metrics in result.items()
        }
        for market, result in results.items()
    }
    print(json.dumps(compact_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
