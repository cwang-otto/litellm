"""
Unit tests for the Responses-API streaming-fallback helpers added to Router
in PR #28215 (fix(router): wrap aresponses streaming iterator for mid-stream
fallbacks).

Targets the four helpers introduced on Router:
  - _extract_partial_responses_usage
  - _combine_responses_fallback_usage
  - _build_responses_continuation_input
  - _aresponses_streaming_iterator
"""

import os
import sys
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


def _make_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                },
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-test",
                },
            },
        ]
    )


def _make_completed_event(
    input_tokens: int, output_tokens: int, total_tokens: int
) -> ResponseCompletedEvent:
    response = ResponsesAPIResponse.model_construct(
        usage=ResponseAPIUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    )
    return ResponseCompletedEvent.model_construct(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=response,
    )


# -------- _extract_partial_responses_usage --------


def test_extract_partial_responses_usage_native_completed():
    """Native path: completed_response carries usage → returned as-is."""
    completed = _make_completed_event(11, 7, 18)
    source = MagicMock()
    source.completed_response = completed

    usage = Router._extract_partial_responses_usage(source)
    assert usage is not None
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert usage.total_tokens == 18


def test_extract_partial_responses_usage_no_completed_response():
    """Native path: no completed_response → returns None."""
    source = MagicMock()
    source.completed_response = None

    usage = Router._extract_partial_responses_usage(source)
    assert usage is None


def test_extract_partial_responses_usage_iterator_missing_the_attribute():
    """
    Iterator that never declares completed_response → None, not AttributeError.

    Distinct from the test above, which covers the attribute being *present and
    None*. A MagicMock cannot express "attribute absent" — it fabricates any
    attribute read — so only a real object exercises this, and the bridge
    iterator is exactly such an object (it skips super().__init__()).
    """

    class _NoCompletedResponse:
        pass

    assert Router._extract_partial_responses_usage(_NoCompletedResponse()) is None


def test_bridge_iterator_declares_completed_response():
    """
    LiteLLMCompletionStreamingIterator must declare completed_response even
    though it does not call super().__init__() — Router and the proxy's
    container-ownership hook both read it off any streaming iterator.
    """
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )

    iterator = LiteLLMCompletionStreamingIterator(
        model="claude-sonnet-4-6",
        litellm_custom_stream_wrapper=MagicMock(),
        request_input="hello",
        responses_api_request={},
    )

    assert iterator.completed_response is None


def test_bridge_iterator_matches_base_field_set():
    """
    The bridge iterator does not call super().__init__() (the base wants an
    httpx.Response it never has), yet callers treat it as a
    BaseResponsesAPIStreamingIterator and read base fields off it. Pin the whole
    field set: declaring them one incident at a time has already shipped two
    production failures (completed_response -> AttributeError inside the
    except MidStreamFallbackError handler, killing the fallback;
    _hidden_params -> prepare_response_for_header_attachment swapped the object
    for a HiddenParamsAsyncIteratorWrapper, changing the asserted type).

    Parsing the base __init__ rather than instantiating it keeps this honest
    without fabricating an httpx.Response: any field a future base class adds
    fails here instead of at request time.
    """
    import re
    from pathlib import Path

    from litellm.responses import streaming_iterator as base_module
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )

    base_src = Path(base_module.__file__).read_text()
    base_init = base_src.split("class BaseResponsesAPIStreamingIterator")[1]
    base_init = base_init.split("def __init__")[1].split("\n    def ")[0]
    base_fields = set(re.findall(r"self\.(\w+)\s*[:=]", base_init))
    assert "completed_response" in base_fields, "base __init__ parse looks wrong"

    bridge = LiteLLMCompletionStreamingIterator(
        model="claude-sonnet-4-6",
        litellm_custom_stream_wrapper=MagicMock(),
        request_input="hello",
        responses_api_request={},
    )

    missing = sorted(base_fields - set(vars(bridge)))
    assert not missing, f"bridge iterator is missing base-class fields: {missing}"


def test_extract_partial_responses_usage_bridge_no_chunks_yet():
    """
    Bridge path with nothing accumulated (mid-stream error on the very first
    chunk, e.g. Anthropic 'Overloaded') falls through to the native branch and
    returns None instead of raising.
    """
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )

    iterator = LiteLLMCompletionStreamingIterator(
        model="claude-sonnet-4-6",
        litellm_custom_stream_wrapper=MagicMock(),
        request_input="hello",
        responses_api_request={},
    )
    assert iterator.collected_chat_completion_chunks == []

    assert Router._extract_partial_responses_usage(iterator) is None


# -------- _combine_responses_fallback_usage --------


def test_combine_responses_fallback_usage_sums_completed_event():
    """Partial-stream usage is summed into the fallback event's usage."""
    fallback_event = _make_completed_event(5, 3, 8)
    partial = ResponseAPIUsage(input_tokens=11, output_tokens=7, total_tokens=18)

    Router._combine_responses_fallback_usage(fallback_event, partial)

    combined = fallback_event.response.usage
    assert combined is not None
    assert combined.input_tokens == 16
    assert combined.output_tokens == 10
    assert combined.total_tokens == 26


def test_combine_responses_fallback_usage_passthrough_for_unknown_event():
    """Events that are not completed/failed/incomplete are not mutated."""
    other = MagicMock()  # not a ResponseCompletedEvent etc. → isinstance false
    partial = ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    Router._combine_responses_fallback_usage(other, partial)
    # No mutation expected on the unknown event — call is a no-op.


# -------- _build_responses_continuation_input --------


def test_build_responses_continuation_input_from_string():
    out = Router._build_responses_continuation_input(
        "Hello world", "partial assistant text"
    )
    assert len(out) == 3
    assert out[0]["role"] == "user"
    assert out[0]["content"][0]["text"] == "Hello world"
    assert out[1]["role"] == "developer"
    assert out[2]["role"] == "assistant"
    assert out[2]["content"][0]["text"] == "partial assistant text"


def test_build_responses_continuation_input_from_list_preserves_items():
    existing: List[Any] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "msg1"}],
        }
    ]
    out = Router._build_responses_continuation_input(existing, "partial")
    assert len(out) == 3
    assert out[0]["content"][0]["text"] == "msg1"
    assert out[1]["role"] == "developer"
    assert out[2]["role"] == "assistant"


def test_build_responses_continuation_input_from_none():
    out = Router._build_responses_continuation_input(None, "partial")
    assert len(out) == 2
    assert out[0]["role"] == "developer"
    assert out[1]["role"] == "assistant"


# -------- _aresponses_streaming_iterator (passthrough smoke test) --------


@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_passthrough():
    """
    Without MidStreamFallbackError, the wrapper yields source events
    unchanged and returns a BaseResponsesAPIStreamingIterator subclass.
    """
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    events = [_make_completed_event(1, 1, 2)]

    class _FakeSource:
        """Minimal source iterator. Provides every attribute the wrapper
        constructor reads from source_iterator."""

        def __init__(self) -> None:
            self._i = 0
            self.completed_response = None
            self.response = MagicMock()
            self.model = "openai/gpt-4o-mini"
            self.logging_obj = MagicMock()
            self.responses_api_provider_config = MagicMock()
            self.start_time = 0.0
            self.litellm_metadata = {}
            self.custom_llm_provider = "openai"
            self.request_data = {}
            self.call_type = "aresponses"
            self._hidden_params: dict = {}

        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self):
            if self._i >= len(events):
                raise StopAsyncIteration
            ev = events[self._i]
            self._i += 1
            return ev

        async def aclose(self):
            return None

    router = _make_router()
    source = _FakeSource()

    wrapper = await router._aresponses_streaming_iterator(
        source, initial_kwargs={"model": "primary"}
    )
    assert isinstance(wrapper, BaseResponsesAPIStreamingIterator)

    collected = [ev async for ev in wrapper]
    assert len(collected) == 1
    assert collected[0].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED


# -------- _aresponses_with_streaming_fallbacks --------


@pytest.mark.asyncio
async def test_aresponses_with_streaming_fallbacks_non_streaming_passthrough():
    """Non-streaming response is returned unchanged, no wrap."""
    router = _make_router()
    plain_response = MagicMock()

    async def fake_original(**_kwargs):
        return plain_response

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks",
        new=AsyncMock(return_value=plain_response),
    ):
        out = await router._aresponses_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=False,
        )
    assert out is plain_response


@pytest.mark.asyncio
async def test_aresponses_with_streaming_fallbacks_wraps_streaming_iterator():
    """Streaming response is wrapped via _aresponses_streaming_iterator."""
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    router = _make_router()
    streaming_iter = MagicMock(spec=BaseResponsesAPIStreamingIterator)
    wrapped = MagicMock(spec=BaseResponsesAPIStreamingIterator)

    async def fake_original(**_kwargs):
        return streaming_iter

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks",
        new=AsyncMock(return_value=streaming_iter),
    ), patch.object(
        router,
        "_aresponses_streaming_iterator",
        new=AsyncMock(return_value=wrapped),
    ) as mock_wrap:
        out = await router._aresponses_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=True,
        )
    assert out is wrapped
    mock_wrap.assert_awaited_once()


@pytest.mark.asyncio
async def test_aresponses_fallback_on_in_stream_error_event():
    """A retriable in-stream error event (429) must trigger the router's mid-stream
    fallback path: the wrapper catches MidStreamFallbackError raised by the source
    iterator and yields the fallback stream instead of surfacing the error."""
    import json
    from unittest.mock import Mock

    import litellm
    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
    from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ErrorEvent, ErrorEventError

    router = _make_router()

    error_payload = {
        "type": "error",
        "error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "rate limited"},
    }
    sse_bytes = f"data: {json.dumps(error_payload)}\n\n".encode()

    async def mock_aiter_bytes():
        yield sse_bytes

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=0,
        error=ErrorEventError(type="tokens", code="rate_limit_exceeded", message="rate limited"),
    )

    source = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    fallback_event = _make_completed_event(1, 1, 2)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_FallbackStream()),
    ) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary", "input": "original question"},
        )
        collected = [ev async for ev in wrapped]

    assert collected == [fallback_event]
    mock_fallback.assert_awaited_once()
    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.status_code == 429
    assert isinstance(raised.original_exception, litellm.APIError)
    assert raised.original_exception.status_code == 429
    assert mock_fallback.await_args.kwargs["kwargs"]["input"] == "original question"


@pytest.mark.asyncio
async def test_aresponses_fallback_uses_continuation_input_after_partial_content():
    """When output text was already streamed before the error, the fallback re-entry
    must carry a continuation input with the partial assistant text instead of
    retrying the original input from scratch (which would duplicate streamed content)."""
    import json
    from unittest.mock import Mock

    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
    from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ErrorEvent, ErrorEventError

    router = _make_router()

    events = [
        {"type": "response.output_text.delta", "delta": "partial answer"},
        {"type": "error", "error": {"type": "server_error", "code": "internal_error", "message": "boom"}},
    ]
    sse_payload = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    async def mock_aiter_bytes():
        yield sse_payload

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def transform(model, parsed_chunk, logging_obj):
        if parsed_chunk.get("type") == "error":
            return ErrorEvent(
                type=ResponsesAPIStreamEvents.ERROR,
                sequence_number=0,
                error=ErrorEventError(**parsed_chunk["error"]),
            )
        delta_event = Mock()
        delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        delta_event.delta = parsed_chunk["delta"]
        return delta_event

    mock_config.transform_streaming_response.side_effect = transform

    source = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    fallback_event = _make_completed_event(1, 1, 2)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_FallbackStream()),
    ) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary", "input": "original question"},
        )
        collected = [ev async for ev in wrapped]

    assert collected[-1] == fallback_event
    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.is_pre_first_chunk is False
    assert raised.generated_content == "partial answer"
    continuation = mock_fallback.await_args.kwargs["kwargs"]["input"]
    assert isinstance(continuation, list)
    assert continuation[0]["content"][0]["text"] == "original question"
    assert continuation[-2]["role"] == "developer"
    assert continuation[-1]["role"] == "assistant"
    assert continuation[-1]["content"][0]["text"] == "partial answer"


@pytest.mark.asyncio
async def test_aresponses_client_error_event_skips_fallback():
    """A 400-mapped in-stream error (raised as APIError, not MidStreamFallbackError)
    must surface to the caller without invoking the router's fallback path."""
    import litellm

    router = _make_router()

    class _ClientErrorSource:
        completed_response = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise litellm.APIError(
                status_code=400,
                message="bad request",
                llm_provider="openai",
                model="gpt-5",
            )

    wrapped = await router._aresponses_streaming_iterator(
        response=_ClientErrorSource(),
        initial_kwargs={"model": "primary"},
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        with pytest.raises(litellm.APIError) as exc_info:
            async for _ in wrapped:
                pass

    assert exc_info.value.status_code == 400
    mock_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_aresponses_fallback_survives_source_without_completed_response():
    """
    Regression: a source iterator that never declared completed_response (the
    litellm-completion bridge, which skips super().__init__()) used to abort the
    mid-stream fallback with AttributeError raised from inside the
    `except MidStreamFallbackError` handler — the failover never ran and callers
    saw an opaque error instead of the fallback stream.

    Reproduces the pre-first-chunk shape (Anthropic 'Overloaded' arriving before
    any content), where the bridge has no accumulated chunks and the helper falls
    through to the native branch.
    """
    from litellm.exceptions import MidStreamFallbackError

    class _NoCompletedResponseSource:
        """No completed_response attribute — deliberately not a MagicMock."""

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise MidStreamFallbackError(
                message="Overloaded",
                model="claude-sonnet-4-6",
                llm_provider="anthropic",
                generated_content="",
                is_pre_first_chunk=True,
            )

    assert not hasattr(_NoCompletedResponseSource(), "completed_response")

    fallback_event = _make_completed_event(2, 3, 5)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    router = _make_router()
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_FallbackStream()),
    ) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=_NoCompletedResponseSource(),
            initial_kwargs={"model": "primary", "input": "original question"},
        )
        collected = [ev async for ev in wrapped]

    assert collected == [fallback_event]
    mock_fallback.assert_awaited_once()
    # Pre-first-chunk → retry with the original input, no continuation prompt.
    assert mock_fallback.await_args.kwargs["kwargs"]["input"] == "original question"


@pytest.mark.asyncio
async def test_aresponses_fallback_survives_usage_extraction_failure():
    """
    Partial-usage extraction is best-effort accounting; whatever it raises must
    not replace the failover with an error. Fallback still runs, just without
    partial usage merged in.
    """
    from litellm.exceptions import MidStreamFallbackError

    class _Source:
        completed_response = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise MidStreamFallbackError(
                message="Overloaded",
                model="claude-sonnet-4-6",
                llm_provider="anthropic",
                generated_content="",
                is_pre_first_chunk=True,
            )

    fallback_event = _make_completed_event(4, 6, 10)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    router = _make_router()
    with (
        patch.object(
            Router,
            "_extract_partial_responses_usage",
            side_effect=RuntimeError("accounting blew up"),
        ),
        patch.object(
            router,
            "async_function_with_fallbacks_common_utils",
            new=AsyncMock(return_value=_FallbackStream()),
        ) as mock_fallback,
    ):
        wrapped = await router._aresponses_streaming_iterator(
            response=_Source(),
            initial_kwargs={"model": "primary", "input": "original question"},
        )
        collected = [ev async for ev in wrapped]

    assert collected == [fallback_event]
    mock_fallback.assert_awaited_once()
    # Usage untouched: extraction failed, so nothing was merged into the event.
    assert collected[0].response.usage.total_tokens == 10
