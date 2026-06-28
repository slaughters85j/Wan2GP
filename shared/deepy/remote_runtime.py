"""OpenAI-compatible remote backend for Deepy.

When the Deepy backend is set to ``remote`` (Configuration > Deepy), the local
nano-vLLM Qwen runtime is bypassed entirely and the assistant turn is driven by
an OpenAI-compatible chat-completions endpoint (e.g. LM Studio on the LAN).

The remote path is stateless per request: the whole conversation is re-sent each
round and the server manages its own KV cache. That lets it reuse all of the
engine's tool registry, tool execution, chat transcript streaming and message
store, while skipping the token-level KV-cache choreography of the local engine.

Heavy / potentially circular imports (the engine module, ``requests``, PIL) are
done lazily inside functions so this module stays cheap to import and never
participates in an import cycle with ``shared.deepy.engine``.
"""

from __future__ import annotations

import json
from typing import Any

from shared.deepy.config import (
    DEEPY_REMOTE_API_KEY_KEY,
    DEEPY_REMOTE_BASE_URL_KEY,
    DEEPY_REMOTE_MODEL_KEY,
    deepy_remote_enabled,
    get_deepy_config_value,
    get_deepy_runtime_config,
    normalize_deepy_remote_api_key,
    normalize_deepy_remote_base_url,
    normalize_deepy_remote_model,
)
from shared.deepy.vision import VISION_QA_SYSTEM_PROMPT
from shared.gradio import assistant_chat


# Safety bound so a misbehaving model cannot loop on tool calls forever.
_MAX_TOOL_ROUNDS = 24
# Reasoning models spend part of the output budget thinking before the answer.
# The server's max_tokens caps reasoning + answer together, so add headroom over
# the requested answer budget to keep the final answer from being truncated.
_THINKING_HEADROOM_TOKENS = 1024
_MIN_COMPLETION_TOKENS = 2048
# UI stream throttle (seconds) to avoid flooding the chat with token-level events.
_STREAM_EMIT_INTERVAL = 0.15
# HTTP timeouts: (connect, read). Read is generous for slow first tokens.
_HTTP_TIMEOUT = (10, 600)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def remote_backend_active() -> bool:
    """True when Deepy is configured to use the remote OpenAI-compatible backend."""
    return deepy_remote_enabled(get_deepy_runtime_config())


def _remote_config() -> tuple[str, str, str]:
    base_url = normalize_deepy_remote_base_url(get_deepy_config_value(DEEPY_REMOTE_BASE_URL_KEY, ""))
    model = normalize_deepy_remote_model(get_deepy_config_value(DEEPY_REMOTE_MODEL_KEY, ""))
    api_key = normalize_deepy_remote_api_key(get_deepy_config_value(DEEPY_REMOTE_API_KEY_KEY, ""))
    return base_url, model, api_key


def _endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.lower().endswith("/v1"):
        base = base[:-3].rstrip("/")
    return f"{base}/v1/{path.lstrip('/')}"


def _headers(api_key: str) -> dict[str, str]:
    # LM Studio ignores the bearer token unless "Require Authentication" is on,
    # but some OpenAI-compatible servers reject a missing Authorization header.
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or 'lm-studio'}",
    }


def _split_think(text: str) -> tuple[str, str]:
    """Split inline ``<think>...</think>`` reasoning out of a content string."""
    lowered = text.lower()
    open_idx = lowered.find(_THINK_OPEN)
    if open_idx < 0:
        return "", text
    close_idx = lowered.find(_THINK_CLOSE)
    if close_idx < 0:
        return text[open_idx + len(_THINK_OPEN):], ""
    reasoning = text[open_idx + len(_THINK_OPEN):close_idx]
    answer = text[:open_idx] + text[close_idx + len(_THINK_CLOSE):]
    return reasoning, answer


def _partition(reasoning_accum: str, answer_accum: str) -> tuple[str, str]:
    """Return (reasoning, answer) for display, preferring a dedicated reasoning field."""
    if reasoning_accum.strip():
        return reasoning_accum, answer_accum
    return _split_think(answer_accum)


def _completion_budget(requested_answer_tokens) -> int:
    """Total max_tokens to request: reasoning models burn part of the budget thinking,
    and the server caps reasoning + answer together, so add headroom over the answer."""
    return max(_MIN_COMPLETION_TOKENS, int(requested_answer_tokens or 0) + _THINKING_HEADROOM_TOKENS)


# One-shot enhancer / vision calls don't stream, so a "thinking-only" reasoning model
# that overruns the budget yields an empty answer. Give these a generous ceiling so the
# answer survives even a long reasoning pass (the served model carries a large context
# window; this caps only a single completion, not the context).
_ONESHOT_COMPLETION_TOKENS = 8192


def _compose_raw_text(reasoning: str, answer: str) -> str:
    reasoning = reasoning.strip()
    answer = answer.strip()
    if len(reasoning) > 0:
        return f"<think>\n{reasoning}\n</think>\n\n{answer}".strip()
    return answer


def _build_openai_messages(system_prompt: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the engine's session.messages (already OpenAI-shaped) into a request payload."""
    payload: list[dict[str, Any]] = [{"role": "system", "content": str(system_prompt or "")}]
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        if role == "user":
            content = msg.get("model_content") or msg.get("content") or ""
            payload.append({"role": "user", "content": str(content)})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": str(msg.get("content", "") or "")}
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": str(tc.get("id", "") or ""),
                        "type": "function",
                        "function": {
                            "name": str((tc.get("function", {}) or {}).get("name", "") or ""),
                            "arguments": json.dumps((tc.get("function", {}) or {}).get("arguments", {}) or {}),
                        },
                    }
                    for tc in tool_calls
                ]
            payload.append(entry)
        elif role == "tool":
            payload.append({
                "role": "tool",
                "tool_call_id": str(msg.get("tool_call_id", "") or ""),
                "content": str(msg.get("content", "") or ""),
            })
    return payload


def _emit_stream(engine, reasoning_accum: str, answer_accum: str) -> None:
    reasoning, answer = _partition(reasoning_accum, answer_accum)
    turn_id = engine._ensure_active_turn()
    reasoning = reasoning.strip()
    answer = answer.strip()
    if len(reasoning) > 0:
        engine._stream_reasoning_block_id, reasoning_event = assistant_chat.upsert_reasoning_block(
            engine.session, turn_id, engine._stream_reasoning_block_id, reasoning
        )
        engine._emit_chat_event(reasoning_event)
    if len(answer) > 0:
        engine._emit_chat_event(assistant_chat.set_assistant_content(engine.session, turn_id, answer))


def _collect_tool_calls(tool_accum: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for index in sorted(tool_accum.keys()):
        slot = tool_accum[index]
        name = str(slot.get("name", "") or "").strip()
        if len(name) == 0:
            continue
        args_text = str(slot.get("arguments", "") or "").strip()
        try:
            arguments = json.loads(args_text) if len(args_text) > 0 else {}
        except Exception:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append({"name": name, "arguments": arguments})
    return tool_calls


def _stream_chat_completion(engine, base_url, model, api_key, messages, tools, *, temperature, top_p, max_tokens):
    """Stream one chat completion. Returns (reasoning, answer, tool_calls)."""
    import time

    import requests

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": int(max_tokens),
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    answer = ""
    reasoning = ""
    tool_accum: dict[int, dict[str, str]] = {}
    last_emit = 0.0

    response = requests.post(
        _endpoint(base_url, "chat/completions"),
        json=body,
        headers=_headers(api_key),
        stream=True,
        timeout=_HTTP_TIMEOUT,
    )
    response.raise_for_status()
    try:
        for line in response.iter_lines(decode_unicode=True):
            if engine.session.interrupt_requested:
                break
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            choices = chunk.get("choices") or []
            if len(choices) == 0:
                continue
            delta = choices[0].get("delta") or {}
            reasoning_delta = delta.get("reasoning_content")
            if reasoning_delta is None:
                reasoning_delta = delta.get("reasoning")
            if reasoning_delta:
                reasoning += str(reasoning_delta)
            content_delta = delta.get("content")
            if content_delta:
                answer += str(content_delta)
            for tool_delta in (delta.get("tool_calls") or []):
                index = int(tool_delta.get("index", 0) or 0)
                slot = tool_accum.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if tool_delta.get("id"):
                    slot["id"] = str(tool_delta["id"])
                function = tool_delta.get("function") or {}
                if function.get("name"):
                    slot["name"] = str(function["name"])
                if function.get("arguments"):
                    slot["arguments"] += str(function["arguments"])
            now = time.perf_counter()
            if now - last_emit >= _STREAM_EMIT_INTERVAL:
                _emit_stream(engine, reasoning, answer)
                last_emit = now
    finally:
        response.close()

    _emit_stream(engine, reasoning, answer)
    reasoning_display, answer_display = _partition(reasoning, answer)
    return reasoning_display.strip(), answer_display.strip(), _collect_tool_calls(tool_accum)


def _chat_completion_once(base_url, model, api_key, messages, *, temperature=0.6, top_p=0.9, max_tokens=512) -> str:
    """Non-streaming chat completion. Returns the assistant message content string."""
    import requests

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": int(max_tokens),
    }
    response = requests.post(
        _endpoint(base_url, "chat/completions"),
        json=body,
        headers=_headers(api_key),
        timeout=_HTTP_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    choices = result.get("choices") or []
    message = (choices[0].get("message") or {}) if len(choices) > 0 else {}
    return str(message.get("content", "") or "")


def _pil_to_data_url(image) -> str:
    import base64
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _to_data_url_any(image) -> str:
    """Encode a PIL image, a file path, or a numpy array to a PNG data URL."""
    from PIL import Image

    if isinstance(image, str):
        with Image.open(image) as handle:
            return _pil_to_data_url(handle)
    if hasattr(image, "save"):
        return _pil_to_data_url(image)
    return _pil_to_data_url(Image.fromarray(image))


def run_remote_prompt_enhance(prompts, *, instructions="", images=None, temperature=0.6, top_p=0.9, max_tokens=512) -> list[str]:
    """Enhance prompts on the remote backend, reusing the model's enhancer instructions.

    Returns one enhanced string per input prompt (matching the local
    ``process_prompt_enhancer`` contract). ``images`` (PIL/path/ndarray) are
    attached to every prompt's message for image-conditioned enhancement.
    """
    base_url, model, api_key = _remote_config()
    if len(base_url) == 0 or len(model) == 0:
        raise RuntimeError("Deepy remote backend is not configured (Base URL + Model).")

    system_prompt = str(instructions or "").strip()
    if len(system_prompt) == 0:
        system_prompt = (
            "You are an expert prompt engineer. Rewrite the user's prompt into a single, "
            "vivid, high-quality generation prompt. Stay faithful to the user's intent and "
            "output only the enhanced prompt with no commentary."
        )

    image_blocks = []
    for image in (images or []):
        try:
            image_blocks.append({"type": "image_url", "image_url": {"url": _to_data_url_any(image)}})
        except Exception:
            continue

    enhanced: list[str] = []
    for prompt in prompts:
        messages = [{"role": "system", "content": system_prompt}]
        text = str(prompt or "")
        if image_blocks:
            messages.append({"role": "user", "content": [{"type": "text", "text": text}, *image_blocks]})
        else:
            messages.append({"role": "user", "content": text})
        content = _chat_completion_once(base_url, model, api_key, messages, temperature=temperature, top_p=top_p, max_tokens=_ONESHOT_COMPLETION_TOKENS)
        _reasoning, answer = _partition("", content)
        answer = answer.strip()
        # If a thinking-only reasoning model overruns the budget without emitting an
        # answer, fall back to the original prompt so generation is never blocked.
        enhanced.append(answer if len(answer) > 0 else text)
    return enhanced


def run_remote_turn(engine, user_text: str, *, max_new_tokens: int = 1024, temperature: float | None = 0.6, top_p: float | None = 0.9) -> None:
    """Drive a full assistant turn against the remote backend.

    Mirrors ``AssistantEngine.run_turn`` but with no KV-cache / local-runtime work:
    build OpenAI messages -> stream a completion -> run any tool calls locally ->
    append results -> loop until the model answers without tool calls.
    """
    # Lazy imports avoid an import cycle with shared.deepy.engine at module load.
    from shared.deepy.engine import (
        checkpoint_assistant_turn,
        finish_assistant_turn,
        rollback_assistant_turn,
    )
    from shared.prompt_enhancer.qwen35_assistant_runtime import extract_tool_calls

    session = engine.session
    user_text = str(user_text or "").strip()
    if len(user_text) == 0:
        engine._send_chat("Please enter a request.")
        return

    base_url, model, api_key = _remote_config()
    if len(base_url) == 0 or len(model) == 0:
        engine._send_chat("Deepy remote backend is not configured. Set a Base URL and Model name in Configuration > Deepy.")
        return

    engine._active_turn_id = ""
    engine._stream_reasoning_block_id = ""
    engine._refresh_runtime_status_note()
    session.messages.append(engine._build_pending_user_message(user_text))
    checkpoint_assistant_turn(session)
    turn_completed = False
    try:
        system_prompt = engine._build_reset_base_system_prompt()
        tools = engine.tool_box.get_tool_schemas()
        for _round in range(_MAX_TOOL_ROUNDS):
            if session.interrupt_requested:
                break
            engine._stream_reasoning_block_id = ""
            engine._set_status("Thinking...", kind="thinking")
            payload_messages = _build_openai_messages(system_prompt, session.messages)
            completion_tokens = _completion_budget(max_new_tokens or 1024)
            reasoning_text, answer_text, tool_calls = _stream_chat_completion(
                engine, base_url, model, api_key, payload_messages, tools,
                temperature=0.6 if temperature is None else float(temperature),
                top_p=0.9 if top_p is None else float(top_p),
                max_tokens=completion_tokens,
            )
            if session.interrupt_requested:
                break

            # Fallback for models that emit Qwen-style tool calls as plain text
            # rather than as structured tool_calls.
            if len(tool_calls) == 0 and len(answer_text) > 0:
                text_tool_calls = extract_tool_calls(answer_text)
                if len(text_tool_calls) == 0:
                    text_tool_calls = engine.tool_box.infer_tool_calls(answer_text)
                tool_calls = text_tool_calls

            raw_text = _compose_raw_text(reasoning_text, answer_text)
            if tool_calls:
                stored_tool_calls = engine._append_assistant_message(raw_text, tool_calls=tool_calls)
                checkpoint_assistant_turn(session)
                for tool_call, stored_tool_call in zip(tool_calls, stored_tool_calls):
                    if session.interrupt_requested:
                        break
                    tool_result = engine._execute_tool(tool_call)
                    engine._append_tool_message(tool_result, stored_tool_call.get("id"))
                    checkpoint_assistant_turn(session)
                if session.interrupt_requested:
                    break
                continue

            engine._append_assistant_message(raw_text)
            checkpoint_assistant_turn(session)
            if len(answer_text.strip()) == 0 and len(reasoning_text.strip()) == 0:
                engine._send_chat("Deepy returned an empty response.")
            turn_completed = True
            break
        else:
            engine._send_chat("Deepy stopped after too many tool rounds in a row.")
    except Exception as exc:
        import traceback

        traceback.print_exc()
        engine._send_chat(f"Deepy remote backend error: {exc}")
    finally:
        engine._hide_status()
        if session.interrupt_requested:
            rollback_assistant_turn(session)
        finish_assistant_turn(session)
        engine._emit_stats(force=True)


def _image_to_data_url(media_record: dict[str, Any], frame_no: int | None) -> str:
    import os

    from PIL import Image

    from shared.utils.utils import get_video_frame

    media_path = str(media_record.get("path", "")).strip()
    if len(media_path) == 0 or not os.path.isfile(media_path):
        raise FileNotFoundError(f"Media file not found: {media_path}")
    media_type = str(media_record.get("media_type", "")).strip().lower()
    if media_type == "video":
        image = get_video_frame(
            media_path,
            0 if frame_no is None else int(frame_no),
            return_last_if_missing=True,
            return_PIL=True,
        )
    else:
        with Image.open(media_path) as handle:
            image = handle.copy()
    return _pil_to_data_url(image)


def run_remote_vision_query(engine, media_record: dict[str, Any], question: str, frame_no: int | None = None) -> dict[str, Any]:
    """Answer an Inspect Media question using the remote vision-language model."""
    media_type = str(media_record.get("media_type", "")).strip().lower()
    question_text = str(question or "").strip()
    base_url, model, api_key = _remote_config()

    def _error(message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "media_id": media_record.get("media_id", ""),
            "media_type": media_type,
            "label": media_record.get("label", ""),
            "frame_no": None if media_type != "video" else (0 if frame_no is None else int(frame_no)),
            "question": question_text,
            "answer": "",
            "error": message,
        }

    if len(base_url) == 0 or len(model) == 0:
        return _error("Deepy remote backend is not configured.")
    try:
        data_url = _image_to_data_url(media_record, frame_no)
    except Exception as exc:
        return _error(str(exc))

    messages = [
        {"role": "system", "content": VISION_QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    try:
        content = _chat_completion_once(base_url, model, api_key, messages, temperature=0.2, top_p=0.9, max_tokens=_ONESHOT_COMPLETION_TOKENS)
    except Exception as exc:
        return _error(str(exc))

    _reasoning, answer = _partition("", str(content or "").strip())
    answer = answer.strip()
    if len(answer) == 0:
        return _error("The remote vision model returned an empty answer.")
    return {
        "status": "done",
        "media_id": media_record.get("media_id", ""),
        "media_type": media_type,
        "label": media_record.get("label", ""),
        "frame_no": None if media_type != "video" else (0 if frame_no is None else int(frame_no)),
        "question": question_text,
        "answer": answer,
        "error": "",
    }


__all__ = [
    "remote_backend_active",
    "run_remote_turn",
    "run_remote_vision_query",
    "run_remote_prompt_enhance",
]
