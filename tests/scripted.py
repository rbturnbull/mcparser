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

COLUMNS = ("id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc")
CELLS = re.compile(r"^([A-Z]+) cells to annotate or review: (.*)$")
SCORE = re.compile(r"^Cells to score: (.*)$")
SENTENCE = re.compile(r"^## s\d+ = sentence (\S+)$")

ToolCalls = list[tuple[str, dict]]


def prompt_of(messages: list[ModelMessage]) -> str:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return str(part.content)
    return ""


def _expand(ranges: str, table: dict[str, list[str]]) -> list[str]:
    ids = []
    for part in ranges.split():
        start, _, end = part.partition("..")
        if end:
            ids += [token_id for token_id in table if token_id.isdigit() and int(start) <= int(token_id) <= int(end)]
        else:
            ids.append(start)
    return ids


def cells(prompt: str) -> list[dict]:
    """The cells listed in a task prompt, with their sentence, form and current value."""
    found, sent_id, table = [], None, {}

    def cell(token_id: str, column: str, confidence: float) -> dict:
        row = table[token_id] + ["_"] * 10
        value = row[COLUMNS.index(column)]
        return {"sent_id": sent_id, "token_id": token_id, "form": row[1], "column": column, "value": value, "confidence": confidence}

    for line in prompt.splitlines():
        header = SENTENCE.match(line)
        if header:
            sent_id, table = header.group(1), {}
        elif line[:1].isdigit() and "\t" in line:
            row = line.split("\t")
            table[row[0]] = row
        elif match := CELLS.match(line):
            column = match.group(1).lower()
            for group in match.group(2).split(", "):
                ranges, _, confidence = group.rpartition(" at ")
                found += [cell(token_id, column, float(confidence)) for token_id in _expand(ranges, table)]
        elif match := SCORE.match(line):
            for group in match.group(1).split("; "):
                column, _, ranges = group.partition(" ")
                found += [cell(token_id, column.lower(), 0.0) for token_id in _expand(ranges, table)]
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
