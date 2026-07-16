"""OpenAI-compatible HTTP client for news-factor extraction and embeddings."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from priorstock.exceptions import ConfigurationError
from priorstock.news_factors.config import ApiEndpointConfig


@dataclass(frozen=True)
class ChatCompletionResponse:
    """Minimal chat-completion response content and raw payload."""

    content: str
    raw_response: dict[str, Any]


class OpenAICompatibleClient:
    """Small requests-based client for OpenAI-compatible chat and embedding endpoints."""

    def __init__(self, api_config: ApiEndpointConfig) -> None:
        """Create a client using an API key read from the configured environment variable."""

        api_key = os.environ.get(api_config.api_key_environment_variable)
        if not api_key:
            raise ConfigurationError(
                f"Environment variable {api_config.api_key_environment_variable} is required."
            )
        base_url = api_config.base_url.rstrip("/")
        self._chat_url = f"{base_url}/chat/completions"
        self._embedding_url = f"{base_url}/embeddings"
        self._timeout_seconds = api_config.request_timeout_seconds
        self._retry_count = api_config.retry_count
        self._retry_wait_seconds = api_config.retry_wait_seconds
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        """Return a per-thread HTTP session so embedding workers can reuse connections."""

        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._thread_local.session = session
        return session

    def create_chat_completion(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        reasoning_effort: str | None,
    ) -> ChatCompletionResponse:
        """Call a chat-completion endpoint and return message content."""

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_output_tokens > 0:
            payload["max_tokens"] = max_output_tokens
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        raw_response = self._post_json_with_retry(self._chat_url, payload)
        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Unexpected chat response format: {raw_response}") from error
        if not isinstance(content, str):
            raise RuntimeError(f"Chat response content is not a string: {raw_response}")
        return ChatCompletionResponse(content=content, raw_response=raw_response)

    def create_embeddings(
        self,
        model_name: str,
        input_texts: list[str],
    ) -> list[list[float] | np.ndarray]:
        """Call an embedding endpoint for a batch of texts."""

        if not input_texts:
            raise ValueError("input_texts must not be empty.")
        payload = {
            "model": model_name,
            "input": input_texts,
            "encoding_format": "base64",
        }
        raw_response = self._post_json_with_retry(self._embedding_url, payload)
        try:
            data = raw_response["data"]
            embeddings_by_index = sorted(data, key=lambda item: int(item["index"]))
            embeddings = [
                self._decode_embedding_payload(item["embedding"])
                for item in embeddings_by_index
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Unexpected embedding response format: {raw_response}") from error
        if len(embeddings) != len(input_texts):
            raise RuntimeError("Embedding response count does not match input count.")
        return embeddings

    @staticmethod
    def _decode_embedding_payload(raw_embedding: object) -> list[float] | np.ndarray:
        """Decode either OpenAI base64 embeddings or float-list fallback payloads."""

        if isinstance(raw_embedding, str):
            return np.frombuffer(base64.b64decode(raw_embedding), dtype=np.float32).copy()
        if isinstance(raw_embedding, list):
            return raw_embedding
        raise RuntimeError(f"Embedding payload has unsupported type: {type(raw_embedding).__name__}")

    def _post_json_with_retry(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON with bounded retry and concrete error reporting."""

        last_error: BaseException | None = None
        for attempt_index in range(self._retry_count + 1):
            try:
                response = self._get_session().post(
                    url,
                    headers=self._headers,
                    data=json.dumps(payload),
                    timeout=self._timeout_seconds,
                )
                if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Retryable status {response.status_code}: {response.text[:500]}",
                        response=response,
                    )
                if response.status_code in {401, 403}:
                    raise PermissionError(
                        f"Authentication or permission failure status {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                response.raise_for_status()
                raw_json = response.json()
                if not isinstance(raw_json, dict):
                    raise RuntimeError("JSON response root is not an object.")
                return raw_json
            except PermissionError:
                raise
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError,
                requests.RequestException,
                requests.JSONDecodeError,
                RuntimeError,
            ) as error:
                last_error = error
                if attempt_index >= self._retry_count:
                    break
                time.sleep(self._retry_wait_seconds)
        raise RuntimeError(f"API request failed after retries: {last_error}") from last_error
