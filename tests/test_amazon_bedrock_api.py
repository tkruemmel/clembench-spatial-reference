"""Response-text extraction of the repo-root Bedrock backend.

The game parsers are anchored to the start of the response and stop at the
first `move:` line, so anything a model emits before its final answer —
separated reasoning blocks (gpt-oss) or inline thinking closed by `</think>`
(Nemotron 3 Super, whose chat template opens `<think>` inside the prompt so
only the closing tag reaches the stream) — must never reach them.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from amazon_bedrock_api import EMPTY_RESPONSE_SENTINEL, extract_response_text  # noqa: E402


def test_plain_text_block_is_returned_unchanged():
    blocks = [{"text": "message: hi\nmove: A up"}]
    assert extract_response_text(blocks) == "message: hi\nmove: A up"


def test_answer_after_last_separated_reasoning_block_wins():
    blocks = [
        {"text": "draft that must be ignored"},
        {"reasoningContent": {"text": "thinking"}},
        {"text": "message: final\nmove: B down"},
    ]
    assert extract_response_text(blocks) == "message: final\nmove: B down"


def test_reasoning_only_response_falls_back_to_reasoning_text():
    blocks = [{"reasoningContent": {"text": "only thinking"}}]
    assert extract_response_text(blocks) == "only thinking"


def test_inline_think_close_tag_drops_the_draft_before_it():
    text = (
        "message: I will wait for my teammate.\nmove: A wait\n\n"
        "Wait — correction: I cannot \"wait\" as a move...\n"
        "</think>\n\n"
        "message: A is blocked right by X, moving left.\nmove: A left"
    )
    assert extract_response_text([{"text": text}]) == (
        "message: A is blocked right by X, moving left.\nmove: A left"
    )


def test_last_think_close_tag_is_the_boundary():
    text = "first draft</think>second draft</think>message: x\nmove: C up"
    assert extract_response_text([{"text": text}]) == "message: x\nmove: C up"


def test_nothing_after_think_close_tag_yields_the_empty_sentinel():
    # The model withdrew its draft and gave no final answer; scoring the
    # draft would credit an answer the model itself discarded.
    text = "message: draft\nmove: A up\nthinking...</think>   \n"
    assert extract_response_text([{"text": text}]) == EMPTY_RESPONSE_SENTINEL


def test_text_without_think_tag_is_left_alone_even_if_long():
    text = "message: draft\nmove: A up\n\nWait — actually...\n" + "more thinking\n" * 50
    assert extract_response_text([{"text": text}]) == text.strip()


def test_empty_blocks_yield_the_empty_sentinel():
    assert extract_response_text([]) == EMPTY_RESPONSE_SENTINEL


# --- output length cap ------------------------------------------------------
# clem always supplies gen_args["max_tokens"] (CLI default 300). A registry entry
# can lift the cap for reasoning models, in which case the Converse request must
# carry no maxTokens at all and the model's own ceiling applies.
from clemcore import backends  # noqa: E402

from amazon_bedrock_api import inference_config, output_cap  # noqa: E402


def _spec(**extra):
    return backends.ModelSpec.from_dict(
        {"model_name": "m", "backend": "amazon_bedrock", "model_id": "vendor.m", **extra})


def test_inference_config_carries_the_cap_by_default():
    assert inference_config(0.6, 300) == {"temperature": 0.6, "maxTokens": 300}


def test_inference_config_omits_max_tokens_when_uncapped():
    assert inference_config(0.6, None) == {"temperature": 0.6}


def test_output_cap_is_the_gen_arg_without_a_registry_flag():
    assert output_cap(_spec(), 300) == 300


def test_registry_flag_uncapped_output_lifts_the_cap():
    assert output_cap(_spec(model_config={"uncapped_output": True}), 300) is None


def test_registry_flag_set_to_false_keeps_the_cap():
    assert output_cap(_spec(model_config={"uncapped_output": False}), 300) == 300
