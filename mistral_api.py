import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

import openai
from retry import retry

import clemcore.backends as backends
import clemcore.backends.openai_api as openai_api

logger = logging.getLogger(__name__)

NAME = "mistral"

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_MIN_INTERVAL_SEC = 0.3  # free tier is ~1 req/s; throttle proactively

# Hybrid-reasoning models (e.g. mistral-medium-3.5 with reasoning_effort>none) can
# exhaust max_tokens entirely on hidden thinking, leaving the visible text part empty
# and finish_reason="length". When that happens we bump max_tokens and retry — up to
# MAX_TOKENS_RETRY_ATTEMPTS times, doubling each time, never exceeding MAX_TOKENS_RETRY_CAP.
MAX_TOKENS_RETRY_ATTEMPTS = 2
MAX_TOKENS_RETRY_FACTOR = 2
MAX_TOKENS_RETRY_CAP = 131072

# Bypass openai_api's @retry(tries=3, delay=10) so we control retry timing for Mistral.
_base_generate_response = openai_api.OpenAIModel.generate_response.__wrapped__


def _flatten_message_content(content):
    # Some Mistral models (e.g. magistral / reasoning) return message.content as a
    # list of typed parts instead of a string, which breaks clemcore's downstream
    # ``(message.content or "").strip()``. Keep "text" parts; drop "thinking" etc.
    if isinstance(content, list):
        parts = []
        for part in content:
            get = part.get if isinstance(part, dict) else lambda k: getattr(part, k, None)
            if get("type") == "text":
                parts.append(get("text") or "")
        return "".join(parts)
    return content


def _normalize_chat_completion(api_response):
    for choice in getattr(api_response, "choices", None) or ():
        message = getattr(choice, "message", None)
        if message is not None and isinstance(getattr(message, "content", None), list):
            message.content = _flatten_message_content(message.content)
    return api_response


class MistralAI(openai_api.OpenAI):
    """Backend for Mistral's native API (api.mistral.ai/v1)."""

    def __init__(self):
        super().__init__(key_name="mistral")

    def _make_api_client(self):
        client = openai.OpenAI(
            base_url=MISTRAL_BASE_URL,
            api_key=self.key["api_key"],
        )
        original_create = client.chat.completions.create

        def create_with_string_content(*args, **kwargs):
            return _normalize_chat_completion(original_create(*args, **kwargs))

        client.chat.completions.create = create_with_string_content
        return client

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return MistralModel(self.client, model_spec)


class MistralModel(openai_api.OpenAIModel, backends.BatchGenerativeModel):

    _last_call_ts: float = 0.0

    @retry(tries=10, delay=15, backoff=2, max_delay=120, logger=logger)
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        original_max_tokens = self.gen_args.get("max_tokens")
        try:
            for attempt in range(MAX_TOKENS_RETRY_ATTEMPTS + 1):
                wait = MISTRAL_MIN_INTERVAL_SEC - (time.monotonic() - MistralModel._last_call_ts)
                if wait > 0:
                    time.sleep(wait)
                try:
                    prompt, response, response_text = _base_generate_response(self, messages)
                finally:
                    MistralModel._last_call_ts = time.monotonic()

                if response_text:
                    return prompt, response, response_text

                choice = (response.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason")

                truncated_thinking = finish_reason == "length"
                if not truncated_thinking or attempt == MAX_TOKENS_RETRY_ATTEMPTS:
                    logger.warning(
                        "Mistral returned empty content (finish_reason=%s).",
                        finish_reason,
                    )
                    return prompt, response, response_text

                current = self.gen_args.get("max_tokens") or 0
                bumped = min(current * MAX_TOKENS_RETRY_FACTOR, MAX_TOKENS_RETRY_CAP)
                if bumped <= current:
                    logger.warning(
                        "Mistral truncated mid-think but max_tokens already at cap (%d) — giving up.",
                        current,
                    )
                    return prompt, response, response_text

                logger.warning(
                    "Mistral truncated mid-think (finish_reason=length, empty visible content). "
                    "Retrying with max_tokens %d -> %d (retry %d/%d).",
                    current, bumped, attempt + 1, MAX_TOKENS_RETRY_ATTEMPTS,
                )
                self.gen_args["max_tokens"] = bumped

            return prompt, response, response_text
        finally:
            if original_max_tokens is None:
                self.gen_args.pop("max_tokens", None)
            else:
                self.gen_args["max_tokens"] = original_max_tokens

    def generate_batch_response(self, batch_messages: List[List[Dict]]) -> List[Tuple[Any, Any, str]]:
        results = [None] * len(batch_messages)
        with ThreadPoolExecutor(max_workers=len(batch_messages)) as executor:
            futures = {
                executor.submit(self.generate_response, messages): i
                for i, messages in enumerate(batch_messages)
            }
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
        return results
