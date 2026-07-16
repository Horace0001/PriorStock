"""Lightweight filesystem helpers for JSON, JSONL, and CSV-like artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable


JSONL_APPEND_MAX_ATTEMPTS = 5
JSONL_APPEND_RETRY_DELAY_SECONDS = 0.2


def ensure_directory(directory_path: Path) -> Path:
    """Create one directory tree if it does not exist and return the same path."""

    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def path_from_serialized_value(path_value: str | Path) -> Path:
    """Convert one stored manifest path into a platform-correct Path object."""

    return Path(str(path_value).replace("\\", "/"))


def serialize_path_for_manifest(path_value: Path) -> str:
    """Serialize one path into POSIX form so manifests stay portable across operating systems."""

    return path_value.as_posix()


def write_json_file(file_path: Path, payload: dict) -> None:
    """Persist one JSON object to disk with UTF-8 encoding."""

    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


def read_json_file(file_path: Path) -> dict:
    """Read one JSON object from disk."""

    with file_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def append_jsonl_records(file_path: Path, records: Iterable[dict]) -> None:
    """Append multiple JSON-serializable records to one JSONL file."""

    ensure_directory(file_path.parent)
    materialized_records = list(records)
    last_permission_error: PermissionError | None = None
    for attempt_index in range(JSONL_APPEND_MAX_ATTEMPTS):
        try:
            with file_path.open("a", encoding="utf-8") as file_handle:
                for record in materialized_records:
                    file_handle.write(json.dumps(record, ensure_ascii=False))
                    file_handle.write("\n")
            return
        except PermissionError as error:
            last_permission_error = error
            is_last_attempt = attempt_index == JSONL_APPEND_MAX_ATTEMPTS - 1
            if is_last_attempt:
                break
            time.sleep(JSONL_APPEND_RETRY_DELAY_SECONDS)

    if last_permission_error is not None:
        raise last_permission_error


def write_jsonl_records(file_path: Path, records: Iterable[dict]) -> None:
    """Overwrite one JSONL file with a complete ordered record sequence."""

    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False))
            file_handle.write("\n")


def read_jsonl_records(file_path: Path) -> list[dict]:
    """Load all JSONL records from one file into memory."""

    records: list[dict] = []
    with file_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped_line = line.strip()
            if stripped_line:
                records.append(json.loads(stripped_line))
    return records
