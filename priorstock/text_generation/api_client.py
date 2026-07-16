"""HTTP client for third-party OpenAI-compatible technical-text generation."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, Timeout

from priorstock.config import TextGenerationConfig
from priorstock.utils.environment import load_project_local_environment_files
from priorstock.utils.rate_limit import FixedIntervalRateLimiter


ERROR_RESPONSE_TEXT_MAX_CHARACTERS = 1000
INSUFFICIENT_USER_QUOTA_ERROR_CODE = "insufficient_user_quota"


@dataclass(frozen=True)
class TechnicalTextRequestRecord:
    """One technical-indicator generation request."""

    record_id: str
    stock_id: str
    stock_name: str
    trade_date: str
    system_prompt: str
    user_prompt: str
    indicator_snapshot: dict


class TextGenerationRequestError(RuntimeError):
    """Raised when one technical-text request fails after all configured retries."""


class TextGenerationQuotaExceededError(TextGenerationRequestError):
    """Raised when the provider reports that the account quota is exhausted."""


class OpenAICompatibleTextGenerationClient:
    """Thin requests-based client for the configured chat-completions endpoint."""

    def __init__(self, text_generation_config: TextGenerationConfig) -> None:
        """Create the client and validate the required API key environment variable."""

        load_project_local_environment_files()
        api_key = os.getenv(text_generation_config.api_key_environment_variable)
        if not api_key:
            raise RuntimeError(
                "The text-generation API key is missing. "
                f"Set environment variable '{text_generation_config.api_key_environment_variable}'."
            )

        self._text_generation_config = text_generation_config
        self._rate_limiter = FixedIntervalRateLimiter(text_generation_config.requests_per_second)
        self._thread_local = threading.local()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _get_thread_local_session(self) -> requests.Session:
        """Return one per-thread requests session with connection pooling enabled."""

        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            http_adapter = requests.adapters.HTTPAdapter(
                pool_connections=self._text_generation_config.max_worker_count,
                pool_maxsize=self._text_generation_config.max_worker_count,
            )
            session.mount("http://", http_adapter)
            session.mount("https://", http_adapter)
            self._thread_local.session = session
        return session

    @staticmethod
    def _build_http_error_message(record_id: str, response: Response) -> str:
        """Format one actionable HTTP error message with truncated response text."""

        response_text_excerpt = response.text[:ERROR_RESPONSE_TEXT_MAX_CHARACTERS]
        return (
            f"Technical-text request for '{record_id}' failed with HTTP {response.status_code}. "
            f"Response body excerpt: {response_text_excerpt}"
        )

    @staticmethod
    def _extract_provider_error_code(response: Response) -> str | None:
        """Return the provider-specific error code when the response body is JSON."""

        try:
            payload = response.json()
        except ValueError:
            return None

        if not isinstance(payload, dict):
            return None

        error_object = payload.get("error")
        if not isinstance(error_object, dict):
            return None

        error_code = error_object.get("code")
        if isinstance(error_code, str) and error_code:
            return error_code
        return None

    def generate_text(self, request_record: TechnicalTextRequestRecord) -> dict:
        """Send one chat-completions request and return the parsed content plus response metadata."""

        retry_delay_seconds = self._text_generation_config.initial_retry_delay_seconds
        last_error_message: str | None = None
        session = self._get_thread_local_session()

        for attempt_index in range(1, self._text_generation_config.max_retry_attempt_count + 1):
            self._rate_limiter.acquire()
            try:
                response = session.post(
                    self._text_generation_config.api_base_url,
                    headers=self._headers,
                    json={
                        "model": self._text_generation_config.api_model_name,
                        "messages": [
                            {"role": "system", "content": request_record.system_prompt},
                            {"role": "user", "content": request_record.user_prompt},
                        ],
                    },
                    timeout=self._text_generation_config.request_timeout_seconds,
                )
                if response.status_code >= 400:
                    raise HTTPError(
                        self._build_http_error_message(request_record.record_id, response),
                        response=response,
                    )
                payload = response.json()
                generated_text = payload["choices"][0]["message"]["content"].strip()
                return {
                    "record_id": request_record.record_id,
                    "stock_id": request_record.stock_id,
                    "stock_name": request_record.stock_name,
                    "trade_date": request_record.trade_date,
                    "indicator_snapshot": request_record.indicator_snapshot,
                    "generated_text": generated_text,
                    "raw_response": payload,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            except Timeout as error:
                last_error_message = (
                    f"Technical-text request for '{request_record.record_id}' timed out on attempt "
                    f"{attempt_index}/{self._text_generation_config.max_retry_attempt_count}: {error}"
                )
            except RequestsConnectionError as error:
                last_error_message = (
                    f"Technical-text request for '{request_record.record_id}' hit a network connection error on "
                    f"attempt {attempt_index}/{self._text_generation_config.max_retry_attempt_count}: {error}"
                )
            except HTTPError as error:
                status_code = error.response.status_code if error.response is not None else None
                provider_error_code = (
                    self._extract_provider_error_code(error.response) if error.response is not None else None
                )
                if provider_error_code == INSUFFICIENT_USER_QUOTA_ERROR_CODE:
                    raise TextGenerationQuotaExceededError(
                        f"Technical-text request for '{request_record.record_id}' failed because the provider "
                        f"reported exhausted account quota ({INSUFFICIENT_USER_QUOTA_ERROR_CODE}): {error}"
                    ) from error
                if status_code not in self._text_generation_config.retryable_status_codes:
                    raise TextGenerationRequestError(
                        f"Technical-text request for '{request_record.record_id}' failed with a non-retryable "
                        f"HTTP status code {status_code}: {error}"
                    ) from error
                last_error_message = (
                    f"Technical-text request for '{request_record.record_id}' received a retryable HTTP status "
                    f"code {status_code} on attempt "
                    f"{attempt_index}/{self._text_generation_config.max_retry_attempt_count}: {error}"
                )
            except ValueError as error:
                last_error_message = (
                    f"Technical-text request for '{request_record.record_id}' returned a non-JSON payload on "
                    f"attempt {attempt_index}/{self._text_generation_config.max_retry_attempt_count}: {error}"
                )

            is_last_attempt = attempt_index == self._text_generation_config.max_retry_attempt_count
            if is_last_attempt:
                break
            time.sleep(retry_delay_seconds)
            retry_delay_seconds *= self._text_generation_config.retry_backoff_multiplier

        raise TextGenerationRequestError(last_error_message or f"Technical-text request for '{request_record.record_id}' failed.")
