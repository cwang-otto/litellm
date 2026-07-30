import json

import pytest

import litellm
from litellm.router_utils.fallback_event_handlers import (
    MID_STREAM_CONTINUATION_SYSTEM_PROMPT,
    build_mid_stream_continuation_messages,
    get_fallback_model_group,
    run_async_fallback,
)


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {}}


class FakeRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        return StreamingWrapper()


class AlwaysFailRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_adds_errors_when_opted_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert json.loads(additional_headers["x-litellm-fallback-errors"]) == [
        {
            "message": "upstream limited request",
            "type": "RuntimeError",
            "param": None,
            "code": None,
        }
    ]


@pytest.mark.asyncio
async def test_run_async_fallback_omits_errors_without_opt_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert "x-litellm-fallback-errors" not in additional_headers


@pytest.mark.asyncio
async def test_run_async_fallback_raises_when_all_fallbacks_fail():
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=AlwaysFailRouter(),
            fallback_model_group=["fallback-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original request failed"),
            max_fallbacks=3,
            fallback_depth=0,
            include_fallback_errors=True,
        )


class RecordingRouter:
    def __init__(self):
        self.received_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return StreamingWrapper()


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_include_fallback_errors_to_nested_call():
    """A nested fallback (multi-hop) must keep collecting errors, so the opt-in
    flag has to reach the nested async_function_with_fallbacks call."""
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    assert router.received_kwargs.get("include_fallback_errors") is True


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_forward_flag_without_opt_in():
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert "include_fallback_errors" not in router.received_kwargs


@pytest.mark.asyncio
async def test_run_async_fallback_skips_original_model_group():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["primary-model", "fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(
        fallbacks=fallbacks, model_group="unmatched-model"
    )

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]


# --------------------------------------------------------------------------
# Mid-stream fallback continuation message building.
#
# Anthropic removed assistant prefill starting with Claude Sonnet 4.6 / Opus 4.6
# (a prefilled assistant message returns a 400 error), so the mid-stream fallback
# must use the documented user-message continuation pattern for those models:
# https://platform.claude.com/docs/en/about-claude/models/migration-guide
#
# Models whose registry entry says supports_assistant_prefill=true, has no value,
# or is unknown keep the legacy prefill-resume behavior.
# --------------------------------------------------------------------------




@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    """Pin capability lookups to the in-repo cost map so the tests exercise this
    PR's registry changes instead of the remote map."""
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()

MESSAGES = [{"role": "user", "content": "Plan my trip to Tokyo"}]
PARTIAL = "Here are the best flight options I found so"


def _build(model_group):
    return build_mid_stream_continuation_messages(
        messages=MESSAGES,
        generated_content=PARTIAL,
        model_group=model_group,
    )


def _assert_legacy_prefill(result):
    assert len(result) == 3
    assert result[0] == MESSAGES[0]
    assert result[1] == {
        "role": "system",
        "content": MID_STREAM_CONTINUATION_SYSTEM_PROMPT,
    }
    assert result[2] == {
        "role": "assistant",
        "content": PARTIAL,
        "prefix": True,
    }


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "anthropic/claude-sonnet-4-6",
        "vertex_ai/claude-sonnet-4-6",
        "claude-opus-4-6",
        "openrouter/anthropic/claude-sonnet-4.6",  # dot-variant registry key
    ],
)
def test_prefill_rejecting_models_get_user_continuation(model):
    """Claude Sonnet 4.6+/Opus 4.6+ reject assistant prefill with a 400 —
    the continuation must ride a user message and carry the partial text."""
    result = _build(model)
    assert len(result) == 2
    assert result[0] == MESSAGES[0]
    assert result[1]["role"] == "user"
    assert PARTIAL in result[1]["content"]
    assert "Continue from where you left off" in result[1]["content"]
    # No prefill anywhere — the conversation must end with a user message.
    assert all(m.get("prefix") is not True for m in result)


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-5",  # registry: supports_assistant_prefill=true
        "gpt-4",  # registry entry exists, capability field absent
        "definitely-not-a-real-model",  # unknown model → capability lookup fails
        None,  # no model group available
    ],
)
def test_other_models_keep_legacy_prefill_resume(model):
    """Anything not explicitly marked supports_assistant_prefill=false keeps the
    pre-existing prefill-resume behavior (back-compat)."""
    _assert_legacy_prefill(_build(model))


def test_prefill_rejecting_fallback_target_gets_user_continuation():
    """The same continuation messages go to every fallback target — a primary
    that supports prefill must still use the user continuation when any
    configured fallback target rejects it (e.g. claude-sonnet-4-5 → claude-sonnet-4-6)."""
    result = build_mid_stream_continuation_messages(
        messages=MESSAGES,
        generated_content=PARTIAL,
        model_group="claude-sonnet-4-5",
        fallbacks=[{"claude-sonnet-4-5": ["claude-sonnet-4-6"]}],
    )
    assert len(result) == 2
    assert result[1]["role"] == "user"
    assert PARTIAL in result[1]["content"]


def test_unrelated_fallback_groups_do_not_affect_prefill():
    """Fallback config for OTHER model groups must not flip this group's behavior."""
    result = build_mid_stream_continuation_messages(
        messages=MESSAGES,
        generated_content=PARTIAL,
        model_group="gpt-4",
        fallbacks=[{"some-other-model": ["claude-sonnet-4-6"]}],
    )
    _assert_legacy_prefill(result)


def test_fallbacks_list_is_not_mutated_by_capability_check():
    """get_fallback_model_group pops a matching entry from flat string-format
    fallback lists — the capability check must operate on a copy, or the actual
    fallback execution silently loses one destination per mid-stream retry."""
    string_format_fallbacks = ["gpt-4", "claude-sonnet-4-6"]
    build_mid_stream_continuation_messages(
        messages=MESSAGES,
        generated_content=PARTIAL,
        model_group="gpt-4",
        fallbacks=string_format_fallbacks,
    )
    assert string_format_fallbacks == ["gpt-4", "claude-sonnet-4-6"]

    dict_format_fallbacks = [{"gpt-4": ["claude-sonnet-4-6"]}]
    build_mid_stream_continuation_messages(
        messages=MESSAGES,
        generated_content=PARTIAL,
        model_group="gpt-4",
        fallbacks=dict_format_fallbacks,
    )
    assert dict_format_fallbacks == [{"gpt-4": ["claude-sonnet-4-6"]}]
