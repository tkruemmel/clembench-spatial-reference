import logging
import time
from typing import Any, Dict, List, Tuple

import openai
from retry import retry

import clemcore.backends as backends
import clemcore.backends.openai_api as openai_api
from clemcore.backends.utils import augment_response_object, ensure_messages_format

logger = logging.getLogger(__name__)

NAME = "moonshot"

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
# Moonshot does not publish hard rate limits, but the API can return 429/5xx
# under load. Keep a light inter-request gap to avoid thundering-herd on batch
# calls, and rely on retry+backoff for the 429/5xx/transient cases.
MOONSHOT_MIN_INTERVAL_SEC = 0.1


class MoonshotModel(openai_api.OpenAIModel):
    """OpenAI-compatible model for Moonshot AI that forces temperature=1 (API requirement)."""

    _last_call_ts: float = 0.0

    @retry(tries=10, delay=15, backoff=2, max_delay=120, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        wait = MOONSHOT_MIN_INTERVAL_SEC - (time.monotonic() - MoonshotModel._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        try:
            prompt = self.encode_messages(messages)
            gen_kwargs = dict(model=self.model_spec.model_id, messages=prompt)
            gen_kwargs = {**gen_kwargs, **self.gen_args}
            gen_kwargs["temperature"] = 1  # Moonshot API only accepts temperature=1
            api_response = self.client.chat.completions.create(**gen_kwargs)
        finally:
            MoonshotModel._last_call_ts = time.monotonic()
        message = api_response.choices[0].message
        if message.role != "assistant":
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
        response_text = (message.content or "").strip()
        if not response_text:
            raise ValueError("Moonshot API response message content is None or empty")
        return prompt, api_response.model_dump(mode="json"), response_text


class MoonshotAI(openai_api.OpenAI):
    """Backend for Moonshot AI's native API (api.moonshot.ai/v1)."""

    def __init__(self):
        super().__init__(key_name="moonshot")

    def _make_api_client(self):
        return openai.OpenAI(
            base_url=MOONSHOT_BASE_URL,
            api_key=self.key["api_key"],
        )

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return MoonshotModel(self.client, model_spec)
