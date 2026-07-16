"""Factor prompt rendering, parsing, and extraction checkpointing."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from priorstock.news_factors.api_client import OpenAICompatibleClient
from priorstock.news_factors.config import NewsFactorExperimentConfig


@dataclass(frozen=True)
class FactorExtractionRecord:
    """One persisted factor extraction result."""

    sample_id: str
    split_name: str
    llm_model: str
    prompt_version: str
    raw_response: str
    parsed_factors: list[str]
    retry_count: int
    is_valid: bool
    error_message: str


class RequestRateLimiter:
    """Thread-safe fixed-interval limiter for outbound API requests."""

    def __init__(self, requests_per_second: float) -> None:
        """Create a limiter from a positive request rate."""

        if requests_per_second <= 0.0:
            raise ValueError("requests_per_second must be positive.")
        self._minimum_interval_seconds = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def wait_for_slot(self) -> None:
        """Block until the next request slot is available."""

        with self._lock:
            current_time = time.monotonic()
            sleep_seconds = max(0.0, self._next_allowed_time - current_time)
            self._next_allowed_time = max(current_time, self._next_allowed_time) + (
                self._minimum_interval_seconds
            )
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)


def render_factor_prompt(
    prompt_template: str,
    company_name: str,
    ticker: str,
    industry: str,
    formatted_news: str,
) -> str:
    """Render the factor-extraction prompt for one sample."""

    return prompt_template.format(
        stocktarget=f"{company_name} ({ticker})",
        company_name=company_name,
        ticker=ticker,
        industry=industry,
        dated_news_titles=formatted_news,
    )


def parse_factor_response(raw_response: str, expected_factor_count: int) -> list[str]:
    """Parse concise factor lines from one LLM response."""

    parsed_factors: list[str] = []
    for raw_line in _split_factor_candidates(raw_response):
        cleaned_line = raw_line.strip()
        if not cleaned_line:
            continue
        cleaned_line = _clean_factor_line(cleaned_line)
        if cleaned_line:
            parsed_factors.append(cleaned_line)
        if len(parsed_factors) >= expected_factor_count:
            break
    return parsed_factors


def _split_factor_candidates(raw_response: str) -> list[str]:
    """Split factor candidates from either line-based or inline numbered output."""

    line_candidates = [line for line in raw_response.splitlines() if line.strip()]
    if len(line_candidates) >= 5:
        return line_candidates
    normalized_response = raw_response.replace("\r", "\n")
    inline_split_pattern = re.compile(
        r"(?:^|\n|\s)(?:[-*]|\d+[\).\:]|factor\s+\d+[\).\:]?)\s+",
        flags=re.IGNORECASE,
    )
    inline_candidates = [
        candidate.strip()
        for candidate in inline_split_pattern.split(normalized_response)
        if candidate.strip()
    ]
    if len(inline_candidates) >= len(line_candidates):
        return inline_candidates
    return line_candidates


def _clean_factor_line(raw_line: str) -> str:
    """Remove list markers and shallow markdown decoration from one factor line."""

    cleaned_line = re.sub(r"^\s*(?:[-*]|\d+[\).\:]?|factor\s+\d+[\).\:]?)\s*", "", raw_line)
    cleaned_line = cleaned_line.strip("\"' ")
    cleaned_line = re.sub(r"^\*+|\*+$", "", cleaned_line).strip()
    return cleaned_line


def load_existing_factor_records(
    output_file_path: Path,
    disabled_model_names: set[str],
) -> set[tuple[str, str]]:
    """Load completed sample-model keys from an existing JSONL checkpoint."""

    completed_keys: set[tuple[str, str]] = set()
    if not output_file_path.exists():
        return completed_keys
    with output_file_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            error_message = str(record.get("error_message", ""))
            record_model_name = str(record["llm_model"])
            if bool(record.get("is_valid")) or (
                error_message.startswith("DISABLED_MODEL:")
                and record_model_name in disabled_model_names
            ):
                completed_keys.add((str(record["sample_id"]), str(record["llm_model"])))
    return completed_keys


def extract_factors_for_round(
    config: NewsFactorExperimentConfig,
    sample_pool_file_path: Path,
    output_file_path: Path,
) -> None:
    """Extract and checkpoint factors for every selected sample and configured LLM."""

    sample_frame = pd.read_json(sample_pool_file_path, lines=True)
    disabled_model_names = set(config.chat.disabled_models)
    completed_keys = load_existing_factor_records(output_file_path, disabled_model_names)
    client = OpenAICompatibleClient(config.api)
    rate_limiter = RequestRateLimiter(config.api.requests_per_second)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_lock = threading.Lock()
    pending_tasks: list[tuple[pd.Series, str]] = []
    for _, sample_row in sample_frame.iterrows():
        for model_name in config.chat.models:
            task_key = (str(sample_row["sample_id"]), model_name)
            if task_key not in completed_keys:
                pending_tasks.append((sample_row, model_name))

    def _run_one_task(sample_row: pd.Series, model_name: str) -> FactorExtractionRecord:
        """Execute one sample-model factor extraction with one parse-level retry."""

        prompt = render_factor_prompt(
            prompt_template=config.chat.prompt_template,
            company_name=str(sample_row["stock_name"]),
            ticker=str(sample_row["stock_id"]),
            industry=str(sample_row.get("industry", "Unknown")),
            formatted_news=str(sample_row["formatted_recent_news"]),
        )
        if model_name in disabled_model_names:
            return FactorExtractionRecord(
                sample_id=str(sample_row["sample_id"]),
                split_name=str(sample_row["split_name"]),
                llm_model=model_name,
                prompt_version=config.chat.prompt_version,
                raw_response="",
                parsed_factors=[],
                retry_count=0,
                is_valid=False,
                error_message=f"DISABLED_MODEL: {model_name}",
            )
        last_error_message = ""
        last_response_content = ""
        primary_max_output_tokens = config.chat.max_output_tokens_by_model.get(
            model_name,
            config.chat.max_output_tokens,
        )
        retry_max_output_tokens = config.chat.retry_max_output_tokens_by_model.get(
            model_name,
        )
        max_output_token_sequence = [primary_max_output_tokens]
        if retry_max_output_tokens is not None and retry_max_output_tokens != primary_max_output_tokens:
            max_output_token_sequence.append(retry_max_output_tokens)
        else:
            max_output_token_sequence.append(primary_max_output_tokens)
        for parse_attempt_index, max_output_tokens in enumerate(max_output_token_sequence):
            try:
                rate_limiter.wait_for_slot()
                response = client.create_chat_completion(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=config.chat.temperature,
                    top_p=config.chat.top_p,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=config.chat.reasoning_effort_by_model.get(model_name),
                )
                last_response_content = response.content
                parsed_factors = parse_factor_response(response.content, 5)
                is_valid = len(parsed_factors) >= 5
                if is_valid:
                    return FactorExtractionRecord(
                        sample_id=str(sample_row["sample_id"]),
                        split_name=str(sample_row["split_name"]),
                        llm_model=model_name,
                        prompt_version=config.chat.prompt_version,
                        raw_response=response.content,
                        parsed_factors=parsed_factors[:5],
                        retry_count=parse_attempt_index,
                        is_valid=True,
                        error_message="",
                    )
                last_error_message = "Parsed fewer than five factors."
            except (RuntimeError, PermissionError, ValueError) as error:
                last_error_message = str(error)
        return FactorExtractionRecord(
            sample_id=str(sample_row["sample_id"]),
            split_name=str(sample_row["split_name"]),
            llm_model=model_name,
            prompt_version=config.chat.prompt_version,
            raw_response=last_response_content,
            parsed_factors=[],
            retry_count=1,
            is_valid=False,
            error_message=last_error_message,
        )

    completed_since_flush = 0
    completed_count = 0
    total_pending_count = len(pending_tasks)
    with ThreadPoolExecutor(max_workers=config.api.max_workers) as executor:
        future_to_key = {
            executor.submit(_run_one_task, sample_row, model_name): (
                str(sample_row["sample_id"]),
                model_name,
            )
            for sample_row, model_name in pending_tasks
        }
        for future in as_completed(future_to_key):
            record = future.result()
            with output_lock:
                with output_file_path.open("a", encoding="utf-8") as file_handle:
                    file_handle.write(
                        json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    completed_since_flush += 1
                    completed_count += 1
                    if completed_since_flush >= config.output.save_every_n_records:
                        file_handle.flush()
                        completed_since_flush = 0
                if completed_count % config.output.save_every_n_records == 0:
                    print(
                        f"factor_extraction_progress completed={completed_count} "
                        f"pending_total={total_pending_count}",
                        flush=True,
                    )
