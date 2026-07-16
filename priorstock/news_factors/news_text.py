"""Raw CMIN-US news loading and prompt formatting."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from priorstock.news_factors.config import NewsTextConfig


@dataclass(frozen=True)
class NewsWindow:
    """Bounded recent-news text for one stock-date sample."""

    formatted_news: str
    news_item_count: int
    source_dates: tuple[str, ...]


class RawNewsRepository:
    """Lazy reader for CMIN-US raw news CSV files."""

    def __init__(self, news_text_config: NewsTextConfig) -> None:
        """Create the repository and initialize the per-ticker cache."""

        self._config = news_text_config
        self._cache_by_stock_id: dict[str, pd.DataFrame] = {}

    def build_recent_news_window(
        self,
        stock_id: str,
        candidate_dates: list[str],
    ) -> NewsWindow:
        """Return bounded formatted news from the candidate trading dates."""

        news_frame = self._load_stock_news_frame(stock_id)
        selected_lines: list[str] = []
        selected_source_dates: list[str] = []
        total_character_count = 0
        for candidate_date in candidate_dates:
            day_frame = news_frame.loc[news_frame["date"] == candidate_date]
            if day_frame.empty:
                continue
            day_item_count = 0
            for _, row in day_frame.iterrows():
                if day_item_count >= self._config.max_news_items_per_day:
                    break
                if len(selected_lines) >= self._config.max_news_items_per_sample:
                    break
                normalized_text = self._normalize_news_text(
                    str(row[self._config.news_text_source_field])
                )
                if not normalized_text:
                    continue
                bounded_text = normalized_text[: self._config.max_news_characters_per_item]
                line = f"{candidate_date}: {bounded_text}"
                if total_character_count + len(line) > self._config.max_prompt_news_characters:
                    break
                selected_lines.append(line)
                selected_source_dates.append(candidate_date)
                total_character_count += len(line)
                day_item_count += 1
            if len(selected_lines) >= self._config.max_news_items_per_sample:
                break
            if total_character_count >= self._config.max_prompt_news_characters:
                break
        return NewsWindow(
            formatted_news="\n".join(selected_lines),
            news_item_count=len(selected_lines),
            source_dates=tuple(selected_source_dates),
        )

    def _load_stock_news_frame(self, stock_id: str) -> pd.DataFrame:
        """Load and normalize one raw news CSV."""

        if stock_id in self._cache_by_stock_id:
            return self._cache_by_stock_id[stock_id]
        csv_file_path = self._config.raw_news_directory / f"{stock_id}.csv"
        if not csv_file_path.exists():
            empty_frame = pd.DataFrame(columns=["date", self._config.news_text_source_field])
            self._cache_by_stock_id[stock_id] = empty_frame
            return empty_frame
        news_frame = pd.read_csv(csv_file_path, sep="\t")
        required_columns = {"date", self._config.news_text_source_field}
        missing_columns = required_columns - set(news_frame.columns)
        if missing_columns:
            raise ValueError(
                f"Raw news file {csv_file_path} is missing columns: {sorted(missing_columns)}"
            )
        news_frame = news_frame.dropna(subset=["date", self._config.news_text_source_field]).copy()
        if "time" in news_frame.columns:
            news_frame = news_frame.sort_values(["date", "time"], ascending=[True, False])
        else:
            news_frame = news_frame.sort_values(["date"], ascending=[True])
        self._cache_by_stock_id[stock_id] = news_frame
        return news_frame

    @staticmethod
    def _normalize_news_text(raw_text: str) -> str:
        """Normalize HTML entities and whitespace in one raw news text."""

        decoded_text = html.unescape(raw_text)
        collapsed_text = " ".join(decoded_text.split())
        return collapsed_text.strip()
