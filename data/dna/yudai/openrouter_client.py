import json
import logging
import random
import time
from typing import Dict, List, Optional

import requests


class OpenRouterClient:
    """Minimal OpenRouter chat client with retry and backoff.

    Network-level retries (HTTP 429/5xx, connection timeouts) are handled here.
    Content-level validation/retries should be handled by callers.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        seed: Optional[int] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: int = 60,
        max_retries: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)

        if not self.api_key:
            raise ValueError("OPENROUTER API key is required")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat completion request and return assistant content.

        Retries on 429/5xx with exponential backoff and jitter.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, object] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        attempt = 0
        backoff_seconds = 1.0
        last_error: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout_seconds,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    # OpenAI-compatible schema
                    choices = data.get("choices") or []
                    if not choices:
                        raise ValueError("No choices in response")
                    content = choices[0]["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError("Assistant content is not a string")
                    return content

                if resp.status_code in (429, 500, 502, 503, 504):
                    self._log_warning(
                        f"HTTP {resp.status_code} from OpenRouter; will retry (attempt={attempt})"
                    )
                    last_error = RuntimeError(
                        f"Transient HTTP error {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    # Non-retryable
                    try:
                        err_text = resp.text
                    except Exception:
                        err_text = "<no body>"
                    raise RuntimeError(
                        f"OpenRouter error {resp.status_code}: {err_text[:500]}"
                    )

            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                self._log_warning(
                    f"Network error communicating with OpenRouter; will retry (attempt={attempt}): {e}"
                )
            except Exception as e:  # Non-retryable or JSON issues
                raise

            # Backoff before next retry
            attempt += 1
            sleep_seconds = backoff_seconds * (1.5 ** attempt)
            # Full jitter between 0.5x and 1.5x
            jitter = random.uniform(0.5, 1.5)
            time.sleep(sleep_seconds * jitter)

        # Exhausted
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to get response from OpenRouter after retries")

    def _log_warning(self, message: str) -> None:
        try:
            self.logger.warning(message)
        except Exception:
            pass


