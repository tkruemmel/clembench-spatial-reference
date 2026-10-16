import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

import openai
from concurrent.futures import TimeoutError as FuturesTimeoutError
from retry import retry

import clemcore.backends as backends
import clemcore.backends.openai_api as openai_api

logger = logging.getLogger(__name__)

NAME = "deepseek"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# DeepSeek does not publish hard rate limits, but api.deepseek.com frequently
# returns 503 "server busy" under load. Keep a light inter-request gap to avoid
# thundering-herd on batch calls (e.g. 6-player turns), and rely on retry+backoff
# for the 503/429/transient cases.
DEEPSEEK_MIN_INTERVAL_SEC = 0.1

# Reasoning models (thinking enabled) can exhaust max_tokens entirely on
# reasoning_content, leaving content="" and finish_reason="length". When that
# happens we bump max_tokens and retry — up to MAX_TOKENS_RETRY_ATTEMPTS times,
# doubling each time, never exceeding MAX_TOKENS_RETRY_CAP.
MAX_TOKENS_RETRY_ATTEMPTS = 2
MAX_TOKENS_RETRY_FACTOR = 2
MAX_TOKENS_RETRY_CAP = 131072

# Bypass openai_api's @retry(tries=3, delay=10) so we control retry timing for DeepSeek.
_base_generate_response = openai_api.OpenAIModel.generate_response.__wrapped__

# A black-holed request can dribble CDN keep-alive bytes that keep resetting
# httpx's read timeout, so the client-level timeout=1800 never fires (observed
# 2026-08-18: three suites frozen 3-6h in poll() on one ESTAB connection, tiny
# CPU ticks from the dribble). Cap each attempt's wall clock from outside; the
# worker thread leaks on timeout, acceptable for the rare hang.
REQUEST_HARD_CAP_SEC = 1500


class InsufficientBalanceError(BaseException):
    """Account out of credit. Derives from BaseException so the @retry ladder and
    clemcore's broad ``except Exception`` handlers let it through: on 2026-08-22 a
    drained account turned into 10 retries per request, which advanced the progress
    bar over every remaining instance while writing no episodes at all."""


def _call_with_hard_cap(fn, *args):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args)
    try:
        return future.result(timeout=REQUEST_HARD_CAP_SEC)
    except FuturesTimeoutError:
        raise TimeoutError(
            f"DeepSeek request exceeded {REQUEST_HARD_CAP_SEC}s hard cap (hung connection)"
        )
    finally:
        executor.shutdown(wait=False)


class DeepSeekAI(openai_api.OpenAI):
    """Backend for DeepSeek's native API (api.deepseek.com)."""

    def __init__(self):
        super().__init__(key_name="deepseek")

    def _make_api_client(self):
        # Non-streaming requests send headers only after generation completes;
        # thinking-mode turns can exceed the SDK's 600s default, which would
        # retry the same slow prompt until the @retry budget burns for hours.
        # max_retries=0 leaves retrying to the decorator on generate_response.
        return openai.OpenAI(
            base_url=DEEPSEEK_BASE_URL,
            api_key=self.key["api_key"],
            timeout=1800.0,
            max_retries=0,
        )

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return DeepSeekModel(self.client, model_spec)


class DeepSeekModel(openai_api.OpenAIModel, backends.BatchGenerativeModel):

    _last_call_ts: float = 0.0

    @retry(tries=10, delay=15, backoff=2, max_delay=120, logger=logger)
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        original_max_tokens = self.gen_args.get("max_tokens")
        try:
            for attempt in range(MAX_TOKENS_RETRY_ATTEMPTS + 1):
                wait = DEEPSEEK_MIN_INTERVAL_SEC - (time.monotonic() - DeepSeekModel._last_call_ts)
                if wait > 0:
                    time.sleep(wait)
                try:
                    prompt, response, response_text = _call_with_hard_cap(
                        _base_generate_response, self, messages
                    )
                except openai.APIStatusError as exc:
                    if exc.status_code == 402:
                        logger.critical("DeepSeek account out of credit (402) — aborting run")
                        raise InsufficientBalanceError(
                            "DeepSeek returned 402 Insufficient Balance; top up the account"
                        ) from exc
                    raise
                finally:
                    DeepSeekModel._last_call_ts = time.monotonic()

                if response_text:
                    return prompt, response, response_text

                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                reasoning = message.get("reasoning_content")
                finish_reason = choice.get("finish_reason")

                truncated_thinking = bool(reasoning) and finish_reason == "length"
                if not truncated_thinking or attempt == MAX_TOKENS_RETRY_ATTEMPTS:
                    if reasoning:
                        logger.warning(
                            "DeepSeek returned empty content but reasoning_content has %d chars "
                            "(finish_reason=%s). Likely truncated mid-think — raise max_tokens or "
                            "use reasoning_effort=max.",
                            len(reasoning), finish_reason,
                        )
                    else:
                        logger.warning(
                            "DeepSeek returned empty content and no reasoning_content "
                            "(finish_reason=%s).", finish_reason,
                        )
                    return prompt, response, response_text

                current = self.gen_args.get("max_tokens") or 0
                bumped = min(current * MAX_TOKENS_RETRY_FACTOR, MAX_TOKENS_RETRY_CAP)
                if bumped <= current:
                    logger.warning(
                        "DeepSeek truncated mid-think (reasoning_content %d chars) but max_tokens "
                        "already at cap (%d) — giving up.",
                        len(reasoning), current,
                    )
                    return prompt, response, response_text

                logger.warning(
                    "DeepSeek truncated mid-think (reasoning_content %d chars, finish_reason=length). "
                    "Retrying with max_tokens %d -> %d (retry %d/%d).",
                    len(reasoning), current, bumped, attempt + 1, MAX_TOKENS_RETRY_ATTEMPTS,
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
