"""MLP probe for comparing LLM-extracted factor quality."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from priorstock.news_factors.config import NewsFactorExperimentConfig
from priorstock.news_factors.metrics import compute_binary_probe_metrics


@dataclass(frozen=True)
class ProbeSplitArrays:
    """Feature, label, and soft-target arrays for one split."""

    embeddings: np.ndarray
    labels: np.ndarray
    soft_targets: np.ndarray


class FactorProbeModel(nn.Module):
    """Shared factor MLP with mean, max, and attention pooling."""

    def __init__(
        self,
        reduced_dimension: int,
        hidden_dimension: int,
        attention_hidden_dimension: int,
        dropout_probability: float,
    ) -> None:
        """Build the factor probe network."""

        super().__init__()
        self.factor_mlp = nn.Sequential(
            nn.LayerNorm(reduced_dimension),
            nn.Linear(reduced_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.rank_embedding = nn.Parameter(torch.zeros(5, hidden_dimension))
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dimension, attention_hidden_dimension),
            nn.GELU(),
            nn.Linear(attention_hidden_dimension, 1),
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dimension * 3),
            nn.Linear(hidden_dimension * 3, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, factor_embeddings: torch.Tensor) -> torch.Tensor:
        """Return one logit for each sample from five reduced factor embeddings."""

        hidden_states = self.factor_mlp(factor_embeddings)
        hidden_states = hidden_states + self.rank_embedding.unsqueeze(0)
        mean_pool = hidden_states.mean(dim=1)
        max_pool = hidden_states.max(dim=1).values
        attention_weights = torch.softmax(self.attention_score(hidden_states), dim=1)
        attention_pool = (attention_weights * hidden_states).sum(dim=1)
        pooled = torch.cat([mean_pool, max_pool, attention_pool], dim=-1)
        return self.output_mlp(pooled).squeeze(dim=-1)


def run_factor_probe_training(
    config: NewsFactorExperimentConfig,
    sample_pool_file_path: Path,
    embeddings_file_path: Path,
    output_file_path: Path,
) -> None:
    """Train one probe per LLM and seed, then persist aggregate metrics."""

    sample_rows = {}
    with sample_pool_file_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.strip():
                record = json.loads(line)
                sample_rows[str(record["sample_id"])] = record
    embedding_npz = np.load(embeddings_file_path, allow_pickle=False)
    embeddings = embedding_npz["embeddings"]
    sample_ids = embedding_npz["sample_ids"].astype(str)
    split_names = embedding_npz["split_names"].astype(str)
    llm_models = embedding_npz["llm_models"].astype(str)
    enabled_model_names = tuple(
        model_name
        for model_name in config.chat.models
        if model_name not in set(config.chat.disabled_models)
    )
    common_sample_ids = _find_common_valid_sample_ids(sample_ids, llm_models, enabled_model_names)
    all_results: list[dict[str, object]] = []
    for llm_model in enabled_model_names:
        model_mask = (llm_models == llm_model) & np.isin(sample_ids, list(common_sample_ids))
        split_arrays = _build_split_arrays(
            embeddings=embeddings[model_mask],
            sample_ids=sample_ids[model_mask],
            split_names=split_names[model_mask],
            sample_rows=sample_rows,
            config=config,
        )
        llm_results = [
            {
                **_train_single_seed(config, split_arrays, llm_model, seed),
                "common_sample_count": len(common_sample_ids),
                "common_split_counts": {
                    split_name: int(split_arrays[split_name].labels.shape[0])
                    for split_name in ("train", "validation", "test")
                },
            }
            for seed in config.probe.seeds
        ]
        all_results.extend(llm_results)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(all_results, file_handle, ensure_ascii=False, indent=2, sort_keys=True)


def _find_common_valid_sample_ids(
    sample_ids: np.ndarray,
    llm_models: np.ndarray,
    enabled_model_names: tuple[str, ...],
) -> set[str]:
    """Return sample IDs with valid records for every enabled model."""

    sample_ids_by_model = {
        model_name: set(sample_ids[llm_models == model_name].tolist())
        for model_name in enabled_model_names
    }
    missing_models = [
        model_name for model_name, model_sample_ids in sample_ids_by_model.items() if not model_sample_ids
    ]
    if missing_models:
        raise RuntimeError(f"No valid factor records for models: {missing_models}")
    common_sample_ids = set.intersection(*sample_ids_by_model.values())
    if not common_sample_ids:
        raise RuntimeError("Enabled models have no common valid sample IDs.")
    return common_sample_ids


def _build_split_arrays(
    embeddings: np.ndarray,
    sample_ids: np.ndarray,
    split_names: np.ndarray,
    sample_rows: dict[str, dict[str, object]],
    config: NewsFactorExperimentConfig,
) -> dict[str, ProbeSplitArrays]:
    """Reduce embeddings with train-only PCA and return arrays by split."""

    train_embeddings = embeddings[split_names == "train"].reshape(-1, embeddings.shape[-1])
    pca = PCA(
        n_components=config.embedding.reduced_dimension,
        random_state=config.sampling.random_seed,
        svd_solver="randomized",
    )
    pca.fit(train_embeddings)
    reduced_embeddings = pca.transform(embeddings.reshape(-1, embeddings.shape[-1])).reshape(
        embeddings.shape[0],
        embeddings.shape[1],
        config.embedding.reduced_dimension,
    )
    arrays_by_split: dict[str, ProbeSplitArrays] = {}
    for split_name in ("train", "validation", "test"):
        split_mask = split_names == split_name
        split_sample_ids = sample_ids[split_mask]
        labels = np.asarray(
            [int(sample_rows[sample_id]["hard_label"]) for sample_id in split_sample_ids],
            dtype=np.int64,
        )
        returns = np.asarray(
            [float(sample_rows[sample_id]["target_return"]) for sample_id in split_sample_ids],
            dtype=np.float32,
        )
        soft_targets = 0.5 + 0.5 * np.tanh(returns / config.probe.return_soft_label_temperature)
        arrays_by_split[split_name] = ProbeSplitArrays(
            embeddings=reduced_embeddings[split_mask].astype(np.float32),
            labels=labels,
            soft_targets=soft_targets.astype(np.float32),
        )
    return arrays_by_split


def _train_single_seed(
    config: NewsFactorExperimentConfig,
    split_arrays: dict[str, ProbeSplitArrays],
    llm_model: str,
    seed: int,
) -> dict[str, object]:
    """Train a factor probe for one LLM and seed."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = FactorProbeModel(
        reduced_dimension=config.embedding.reduced_dimension,
        hidden_dimension=config.probe.hidden_dimension,
        attention_hidden_dimension=config.probe.attention_hidden_dimension,
        dropout_probability=config.probe.dropout_probability,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.probe.learning_rate,
        weight_decay=config.probe.weight_decay,
    )
    train_dataset = TensorDataset(
        torch.tensor(split_arrays["train"].embeddings),
        torch.tensor(split_arrays["train"].soft_targets),
        torch.tensor(split_arrays["train"].labels),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.probe.batch_size,
        shuffle=True,
        generator=generator,
    )
    best_state_dict = None
    best_validation_metrics: dict[str, float] | None = None
    best_epoch = 0
    patience_counter = 0
    positive_count = max(int(split_arrays["train"].labels.sum()), 1)
    negative_count = max(int(split_arrays["train"].labels.shape[0] - positive_count), 1)
    positive_class_weight = torch.tensor(float(negative_count / positive_count), dtype=torch.float32)
    for epoch_index in range(1, config.probe.max_epochs + 1):
        model.train()
        for batch_embeddings, batch_targets, _ in train_loader:
            logits = model(batch_embeddings)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch_targets,
                pos_weight=positive_class_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation_metrics = _evaluate_probe(config, model, split_arrays["validation"])
        if best_validation_metrics is None or (
            validation_metrics["selection_score"] > best_validation_metrics["selection_score"]
        ):
            best_validation_metrics = validation_metrics
            best_state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch_index
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.probe.early_stopping_patience:
                break
    if best_state_dict is None or best_validation_metrics is None:
        raise RuntimeError("Probe training did not produce a best checkpoint.")
    model.load_state_dict(best_state_dict)
    return {
        "llm_model": llm_model,
        "seed": seed,
        "best_epoch": best_epoch,
        "train": _evaluate_probe(config, model, split_arrays["train"]),
        "validation": _evaluate_probe(config, model, split_arrays["validation"]),
        "test": _evaluate_probe(config, model, split_arrays["test"]),
    }


def _evaluate_probe(
    config: NewsFactorExperimentConfig,
    model: FactorProbeModel,
    split_arrays: ProbeSplitArrays,
) -> dict[str, float]:
    """Evaluate a trained factor probe on one split."""

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(split_arrays.embeddings))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return compute_binary_probe_metrics(
        labels=split_arrays.labels,
        probabilities=probabilities,
        mcc_score_weight=config.probe.score_mcc_weight,
        balanced_accuracy_score_weight=config.probe.score_balanced_accuracy_weight,
    )
