"""Scripted pydantic-ai models for testing the agents without an LLM."""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

CELL = re.compile(r"^- word (\S+) \((.*)\): (\w+) = '(.*)', confidence ([\d.]+)$")
SENTENCE = re.compile(r"^## Sentence (\S+)$")

ToolCalls = list[tuple[str, dict]]


def prompt_of(messages: list[ModelMessage]) -> str:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return str(part.content)
    return ""


def cells(prompt: str) -> list[dict]:
    """The cells listed in a task prompt, with their sentence."""
    found, sent_id = [], None
    for line in prompt.splitlines():
        header = SENTENCE.match(line)
        if header:
            sent_id = header.group(1)
            continue
        match = CELL.match(line)
        if match:
            token_id, form, column, value, confidence = match.groups()
            found.append(
                {"sent_id": sent_id, "token_id": token_id, "form": form, "column": column.lower(), "value": value, "confidence": float(confidence)}
            )
    return found


def sentences(prompt: str) -> list[str]:
    return [match.group(1) for line in prompt.splitlines() if (match := SENTENCE.match(line))]


def retries(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]


Answer = Callable[[str], dict]


def scripted(answer: Answer | None = None, tools: Callable[[str], ToolCalls] | None = None, summary: str = "done") -> FunctionModel:
    """A model that optionally calls ``tools(prompt)`` once, then gives ``answer(prompt)`` as its final result.

    Required answer fields that ``answer`` leaves out are sent empty. Tool
    results and retry messages are recorded on the model (``.returns``,
    ``.retries``); after a retry the answer is sent again, so a test can make
    ``answer`` depend on ``len(model.retries)``.
    """
    model: FunctionModel

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        prompt = prompt_of(messages)
        model.retries[:] = retries(messages)
        first = isinstance(last, ModelRequest) and any(isinstance(p, UserPromptPart) for p in last.parts)
        if first and tools is not None:
            calls = tools(prompt)
            if calls:
                return ModelResponse(parts=[ToolCallPart(name, args) for name, args in calls])
        if not first:
            model.returns.extend(str(getattr(part, "content", "")) for part in last.parts)
        output = info.output_tools[0]
        args = {name: "" for name in output.parameters_json_schema.get("required", [])}
        args.update(answer(prompt) if answer else {})
        args["summary"] = args.get("summary") or summary
        return ModelResponse(parts=[ToolCallPart(output.name, args)])

    model = FunctionModel(respond)
    model.returns = []  # type: ignore[attr-defined]
    model.retries = []  # type: ignore[attr-defined]
    return model


def values(column: str, value: str | Callable[[dict], str], confidence: float = 0.9) -> Answer:
    """An answer setting ``value`` on every listed cell of ``column``."""

    def answer(prompt: str) -> dict:
        lines = [
            f"{c['sent_id']} {c['token_id']} {value(c) if callable(value) else value} {confidence}"
            for c in cells(prompt)
            if c["column"] == column
        ]
        return {"values": "\n".join(lines)}

    return answer


def scored(confidence: Callable[[dict], float] | float = 0.9) -> Answer:
    """A scorer answer giving every listed cell ``confidence`` (or ``confidence(cell)``)."""

    def answer(prompt: str) -> dict:
        lines = [
            f"{c['sent_id']} {c['token_id']} {c['column']} {confidence(c) if callable(confidence) else confidence}"
            for c in cells(prompt)
        ]
        return {"scores": "\n".join(lines)}

    return answer
