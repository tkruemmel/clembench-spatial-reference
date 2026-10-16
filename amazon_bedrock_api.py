import json
import logging
import time
from typing import List, Dict, Tuple, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EventStreamError

from clemcore import backends
from clemcore.backends.utils import ensure_messages_format

logger = logging.getLogger(__name__)

NAME = "amazon_bedrock"

# Whole-call retries for failures that surface while iterating the event
# stream, where botocore's own retry config no longer applies.
STREAM_ATTEMPTS = 4
_TRANSIENT_CODES = {
    "internalServerException", "throttlingException", "serviceUnavailableException",
    "modelTimeoutException", "modelNotReadyException", "ThrottlingException",
    "InternalServerException", "ServiceUnavailableException",
}


def _is_transient(error: Exception) -> bool:
    """True for server-side hiccups worth retrying, not for bad requests."""
    response = getattr(error, "response", None) or {}
    code = (response.get("Error") or {}).get("Code") or ""
    if code in _TRANSIENT_CODES:
        return True
    # EventStreamError stringifies the modeled error name when Error.Code is absent.
    text = str(error)
    return any(c in text for c in _TRANSIENT_CODES)


EMPTY_RESPONSE_SENTINEL = "[[EMPTY_MODEL_RESPONSE]]"
THINK_CLOSE_TAG = "</think>"


def extract_response_text(blocks: List[Dict[str, Any]]) -> str:
    """Return the model's final answer from Converse content blocks.

    Reasoning models (e.g. gpt-oss) return [text, reasoningContent, text] — the
    answer is the text AFTER the last reasoning block; mixing reasoning into
    response_text makes game parsers reject every turn. Fall back to all text
    blocks, then to reasoning text only if no text block exists.

    Nemotron 3 Super's chat template opens `<think>` inside the prompt, so its
    reasoning arrives inline in the text block and only the closing tag is
    visible: [draft answer][thinking]</think>[final answer]. Bedrock never
    splits this into a reasoningContent block. The answer is what follows the
    LAST `</think>`; the draft before it is text the model itself withdrew, and
    the start-anchored game parsers must never see it (2026-09-06: 658 Nemotron
    turns were scored on a draft that differed from the model's final move).
    """
    last_reasoning_idx = -1
    for i, block in enumerate(blocks):
        if isinstance(block.get("reasoningContent"), dict):
            last_reasoning_idx = i

    def _texts(items):
        return [t for t in (str(b["text"]).strip() for b in items if "text" in b) if t]

    text_parts = _texts(blocks[last_reasoning_idx + 1:])
    if not text_parts:
        text_parts = _texts(blocks)
    if not text_parts:
        text_parts = [
            t for t in (
                str(b["reasoningContent"]["text"]).strip()
                for b in blocks
                if isinstance(b.get("reasoningContent"), dict) and "text" in b["reasoningContent"]
            ) if t
        ]
    response_text = "\n".join(text_parts).strip()
    if THINK_CLOSE_TAG in response_text:
        response_text = response_text.rsplit(THINK_CLOSE_TAG, 1)[1].strip()
    if not response_text:
        # Prevent downstream request builders from emitting blank text blocks.
        response_text = EMPTY_RESPONSE_SENTINEL
    return response_text


def inference_config(temperature: float, max_tokens) -> Dict[str, Any]:
    """Converse inferenceConfig. `max_tokens=None` means no cap: the field is
    omitted and the model's own output ceiling applies."""
    cfg: Dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        cfg["maxTokens"] = max_tokens
    return cfg


def output_cap(model_spec: backends.ModelSpec, gen_max_tokens):
    """The cap to send, or None to send none.

    clem always supplies gen_args["max_tokens"] (CLI default 300). A registry
    entry with `"model_config": {"uncapped_output": true}` lifts it: reasoning
    models write their trace into the same token budget as the answer, and any
    cap truncates the answer first (Nemotron 3 Super hit a 16,384 cap 140 times
    in the 2026-08 campaign). The flag travels into run.json via model_spec, so
    a run records that gen_args["max_tokens"] was not applied.
    """
    config = getattr(model_spec, "model_config", None) or {}
    if isinstance(config, dict) and config.get("uncapped_output"):
        return None
    return gen_max_tokens


class AmazonBedrock(backends.Backend):

    def __init__(self):
        # Responses are streamed, so read_timeout applies between chunks
        # rather than to the whole generation: reasoning models can run to the
        # 16k-token cap and exceed any end-to-end timeout (nemotron 3p prompts
        # blew through 1800s on 2026-08-19). A 600s inter-chunk gap means a
        # genuine stall. Adaptive retries absorb transient throttling.
        self.client = boto3.client("bedrock-runtime", config=Config(
            read_timeout=600,
            retries={"max_attempts": 8, "mode": "adaptive"},
        ))

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return AmazonBedrockModel(self.client, model_spec)


class AmazonBedrockModel(backends.Model):

    def __init__(self, client, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.client = client

    # Merge back-to-back user messages (rules + turn context) into one, joined by
    # a blank line, exactly as the OpenAI-based backends (DeepSeek, Qwen) do.
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        prompt = messages
        system_parts = []
        converse_messages = []

        def _normalize_text(value: Any) -> str:
            if isinstance(value, str):
                return value.strip()
            if value is None:
                return ""
            # Keep non-string payloads serializable and non-empty if present.
            return json.dumps(value).strip()

        for msg in messages:
            role = msg.get("role")
            content_text = _normalize_text(msg.get("content"))

            if not content_text:
                logger.debug("Skipping empty Bedrock message for role=%s", role)
                continue

            if role == "system":
                system_parts.append({"text": content_text})
            elif role in {"user", "assistant"}:
                converse_messages.append({
                    "role": role,
                    "content": [{"text": content_text}]
                })
            else:
                logger.debug("Skipping unsupported Bedrock role=%s", role)

        if not converse_messages:
            raise ValueError("Cannot call Bedrock converse with no non-empty non-system messages")

        kwargs = dict(
            modelId=self.model_spec.model_id,
            messages=converse_messages,
            inferenceConfig=inference_config(
                self.temperature, output_cap(self.model_spec, self.max_tokens)),
        )
        if system_parts:
            kwargs["system"] = system_parts

        # botocore's retry config only covers the initial ConverseStream call.
        # A transient internalServerException raised while ITERATING the event
        # stream escapes it and kills the episode (cost one qwen3-235b episode
        # on 2026-08-21), so retry the whole call and rebuild from scratch.
        assembled: Dict[int, Dict[str, List[str]]] = {}
        stop_reason = None
        usage: Dict[str, Any] = {}
        for attempt in range(STREAM_ATTEMPTS):
            assembled = {}
            stop_reason = None
            usage = {}
            try:
                stream = self.client.converse_stream(**kwargs)
                # Reassemble streamed deltas into converse()-shaped content
                # blocks so the extraction below and the logged response keep
                # the exact shape the non-streaming call produced.
                for event in stream["stream"]:
                    if "contentBlockDelta" in event:
                        part = event["contentBlockDelta"]
                        idx = part.get("contentBlockIndex", 0)
                        delta = part.get("delta", {})
                        slot = assembled.setdefault(idx, {"text": [], "reasoning": []})
                        if "text" in delta:
                            slot["text"].append(delta["text"])
                        reasoning_delta = delta.get("reasoningContent")
                        if isinstance(reasoning_delta, dict) and "text" in reasoning_delta:
                            slot["reasoning"].append(reasoning_delta["text"])
                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason")
                    elif "metadata" in event:
                        usage = event["metadata"].get("usage", {})
                break
            except (EventStreamError, ClientError) as e:
                if attempt + 1 >= STREAM_ATTEMPTS or not _is_transient(e):
                    raise
                backoff = 2 ** attempt
                logger.warning(
                    "Bedrock stream failed (attempt %d/%d): %s; retrying in %ds",
                    attempt + 1, STREAM_ATTEMPTS, e, backoff,
                )
                time.sleep(backoff)

        content = []
        for idx in sorted(assembled):
            block: Dict[str, Any] = {}
            if assembled[idx]["text"]:
                block["text"] = "".join(assembled[idx]["text"])
            if assembled[idx]["reasoning"]:
                block["reasoningContent"] = {"text": "".join(assembled[idx]["reasoning"])}
            if block:
                content.append(block)
        response = {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": stop_reason,
            "usage": usage,
        }

        response_text = extract_response_text(response["output"]["message"]["content"])

        return prompt, response, response_text
