"""Helpers for loading local untracked environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def _strip_optional_quotes(raw_value: str) -> str:
    """Remove one layer of matching single or double quotes from a dotenv value."""

    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {"'", '"'}:
        return raw_value[1:-1]
    return raw_value


def load_local_environment_file(environment_file_path: Path) -> None:
    """Read one local dotenv-style file and populate missing process environment variables."""

    if not environment_file_path.exists():
        return

    with environment_file_path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
                continue

            variable_name, variable_value = stripped_line.split("=", maxsplit=1)
            normalized_name = variable_name.strip()
            normalized_value = _strip_optional_quotes(variable_value.strip())
            if normalized_name and normalized_name not in os.environ:
                os.environ[normalized_name] = normalized_value


def load_project_local_environment_files(project_root: Path | None = None) -> None:
    """Populate missing environment variables from project-local ignored dotenv files."""

    resolved_project_root = project_root or Path(__file__).resolve().parents[2]
    for file_name in [".env.local", ".env"]:
        load_local_environment_file(resolved_project_root / file_name)
