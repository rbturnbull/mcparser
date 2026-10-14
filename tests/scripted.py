"""Scripted pydantic-ai models for testing the agents without an LLM."""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

CELL = re.compile(r"^- word (\S+) \((.*)\): (\w+) = '(.*)', confidence ([\d.]+)$")

ToolCalls = list[tuple[str, dict]]


def prompt_of(messages: list[ModelMessage]) -> str:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return str(part.content)
    return ""


def cells(prompt: str) -> list[dict]:
    """The cells listed in a task prompt."""
    found = []
    for line in prompt.splitlines():
        match = CELL.match(line)
        if match:
            token_id, form, column, value, confidence = match.groups()
            found.append({"token_id": token_id, "form": form, "column": column.lower(), "value": value, "confidence": float(confidence)})
    return found


def sent_id(prompt: str) -> str:
    return re.search(r'Sentence "([^"]+)"', prompt).group(1)


def scripted(decide: Callable[[str], ToolCalls], summary: str = "done") -> FunctionModel:
    """A model that makes ``decide(prompt)`` tool calls once, then returns ``summary``.

    Tool results are recorded on the model as ``.returns`` for inspection.
    """
    returns: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(p, UserPromptPart) for p in last.parts):
            calls = decide(prompt_of(messages))
            if calls:
                return ModelResponse(parts=[ToolCallPart(name, args) for name, args in calls])
        else:
            returns.extend(str(getattr(part, "content", "")) for part in last.parts)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"summary": summary})])

    model = FunctionModel(respond)
    model.returns = returns  # type: ignore[attr-defined]
    return model


def nothing(prompt: str) -> ToolCalls:
    return []
