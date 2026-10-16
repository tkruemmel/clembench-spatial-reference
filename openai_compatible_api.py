"""Repo-local openai_compatible backend that streams responses.

Why this exists: the Potsdam CLP endpoint sits behind an Apache proxy with a
hard 300s read timeout. A non-streaming request returns nothing until the whole
generation is finished, so any turn whose thinking runs past 300s is killed with
`502 Proxy Error` - measured deterministically at 300.1s, six times out of six.
Retries do not help (the timeout is not transient; attempts only succeed when
temperature resampling happens to produce a shorter response), and each failed
attempt burns a full 300s of wall clock.

Streaming fixes it outright: the server emits every token as it is generated,
including the hidden thinking tokens, so the connection never falls silent and
the proxy never trips. Measured: a 27,269-token generation completed in 1018s,
3.4x past the non-streaming ceiling.

This is a transport change only - same model, same sampling parameters - so
episodes played before and after remain comparable. clemcore resolves backends
from the working directory before its packaged ones, which is how this file
takes precedence (the same mechanism this repo already uses for mistral_api.py).
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Tuple

import httpx
import openai
from retry import retry

import clemcore.backends as backends
import clemcore.backends.openai_api as openai_api
from clemcore.backends.utils import augment_response_object, ensure_messages_format

logger = logging.getLogger(__name__)

# Generous per-read budget. With streaming this bounds silence between chunks,
# not total generation time, so long reasoning turns are unaffected.
REQUEST_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=120.0, pool=30.0)

# The read timeout above does not fire on a black-holed connection: on 2026-09-08
# 02:32 the proxy dropped all 28 in-flight requests server-side while keeping the
# client sockets open, and every worker sat in an SSL read for over half an hour
# with nothing arriving. So the stream is consumed in a worker thread and watched
# from outside.
#
# The watchdog is IDLE-based, not total-duration based. Calls here legitimately
# run for hours: measured over 21,846 finished calls, 4 % exceed 1800 s and the
# longest was 9,240 s (an 8.3k-token reasoning chain under 28-way contention). A
# 1800 s total cap was tried on 2026-09-08 02:55 and fired 34 times in one hour,
# exhausting the @retry ladder and destroying two nearly finished episodes; a
# duration cap cannot tell a slow call from a dead one. Inter-chunk silence can:
# a hung connection delivers nothing, while a slow call keeps emitting tokens
# (proven by the 300 s proxy timeout never firing on multi-hour calls).
STREAM_IDLE_CAP_SEC = 900      # no chunk at all for 15 min -> connection is dead
STREAM_TOTAL_CAP_SEC = 18000   # 5 h backstop, ~2x the longest call ever observed
POLL_SEC = 30


def _consume_stream(client, gen_kwargs, progress):
    """Consume the stream, recording (last chunk time, chunk count) in ``progress``
    so the calling thread can tell a slow generation from a dead connection."""
    head, usage, finish_reason = None, None, None
    content_parts = []
    for chunk in client.chat.completions.create(**gen_kwargs):
        progress[0] = time.monotonic()
        progress[1] += 1
        if head is None:
            head = chunk
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if getattr(choice.delta, "content", None):
            content_parts.append(choice.delta.content)
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    return head, usage, finish_reason, content_parts


class StreamingOpenAIModel(openai_api.OpenAIModel):
    """OpenAI-compatible model that consumes the response as a token stream."""

    def _reset_client(self):
        """Replace the OpenAI client so the next attempt opens a new connection."""
        try:
            old = self.client
            self.client = openai.OpenAI(
                base_url=str(old.base_url), api_key=old.api_key,
                timeout=REQUEST_TIMEOUT, http_client=httpx.Client(verify=False))
            logger.warning("replaced the OpenAI client after a hung stream")
        except Exception as exc:  # never let recovery break the retry ladder
            logger.warning("could not replace the OpenAI client: %s", exc)

    @retry(tries=3, delay=10, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        prompt = self.encode_messages(messages)
        gen_kwargs = dict(model=self.model_spec.model_id, messages=prompt)
        gen_kwargs = {**gen_kwargs, **self.gen_args}

        model_config = getattr(self.model_spec, "model_config", {})
        if "reasoning_model" in model_config:
            if not self.temperature > 0:
                raise ValueError(f"For reasoning models temperature must be >0, but is {self.temperature}. "
                                 f"Please use the -t option to set a temperature and try again.")
            # Matches the packaged backend: max_tokens is not supported for reasoning models.
            gen_kwargs.pop("max_tokens", None)
        if "extra_body" in model_config:
            gen_kwargs["extra_body"] = model_config["extra_body"]

        gen_kwargs["stream"] = True
        gen_kwargs["stream_options"] = {"include_usage": True}

        progress = [time.monotonic(), 0]
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_consume_stream, self.client, gen_kwargs, progress)
        started = time.monotonic()
        try:
            while True:
                try:
                    head, usage, finish_reason, content_parts = future.result(timeout=POLL_SEC)
                    break
                except FuturesTimeoutError:
                    idle = time.monotonic() - progress[0]
                    total = time.monotonic() - started
                    if idle > STREAM_IDLE_CAP_SEC or total > STREAM_TOTAL_CAP_SEC:
                        # Give @retry a fresh connection pool: the dead connection is
                        # pooled and would be handed straight back to the next attempt.
                        self._reset_client()
                        raise TimeoutError(
                            f"streamed request abandoned after {total:.0f}s, idle {idle:.0f}s, "
                            f"{progress[1]} chunks received (hung connection)") from None
        finally:
            executor.shutdown(wait=False)

        response_text = "".join(content_parts).strip()
        if not response_text:
            logger.warning("Streamed response content is None or empty, returning empty string.")

        # Shaped like a non-streamed ChatCompletion dump so recorders, transcripts
        # and scoring see exactly what they saw before.
        response = {
            "id": getattr(head, "id", None),
            "object": "chat.completion",
            "created": getattr(head, "created", None),
            "model": getattr(head, "model", self.model_spec.model_id),
            "choices": [{
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": response_text},
            }],
            "usage": usage.model_dump(mode="json") if usage is not None else None,
            "clem_transport": "stream",
        }
        return prompt, response, response_text


class GenericOpenAI(openai_api.OpenAI):
    """Generic backend for OpenAI-compatible remote APIs, streaming variant."""

    def __init__(self):
        super().__init__(key_name="openai_compatible")

    def _make_api_client(self):
        return openai.OpenAI(
            base_url=self.key["base_url"],
            api_key=self.key["api_key"],
            timeout=REQUEST_TIMEOUT,
            # The CLP server ships an incomplete cert chain; the packaged backend
            # disables verification here too.
            http_client=httpx.Client(verify=False),
        )

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return StreamingOpenAIModel(self.client, model_spec)
