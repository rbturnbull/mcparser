"""The pydantic-ai agents, one per role, sharing one MCP connection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest, RetryPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from ..corpus import CorpusStore
from ..mcp.server import build_server
from ..pricing import estimate
from ..settings import Settings
from .outputs import Annotation, Applier, Batch, Scores
from .prompts import instructions
from .roles import CORPUS_TOOLS, ROLES
from .toolset import role_toolset


class TaskReport(BaseModel):
    """What an agent that edits through tools returns when it finishes a task."""

    summary: str = Field(description="At most 15 words on what you changed, confirmed or flagged.")


Answer = TaskReport | Annotation | Scores


class ModelUsage(BaseModel):
    """Token use of one model over a run."""

    tasks: int = 0
    requests: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = Field(None, description="Estimated US dollars (see mcparser.pricing).")

    def add(self, usage: RunUsage, model: str) -> None:
        self.tasks += 1
        self.requests += usage.requests
        self.input_tokens += usage.input_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        self.output_tokens += usage.output_tokens
        self.cost = estimate(model, self.input_tokens, self.cache_read_tokens, self.cache_write_tokens, self.output_tokens)

    def merge(self, other: ModelUsage) -> None:
        """Add another usage (e.g. of a different model); cost is None if either has no price."""
        for name in ("tasks", "requests", "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.cost = None if self.cost is None or other.cost is None else self.cost + other.cost


class AgentTeam:
    """Builds and runs the role agents.

    By default the agents reach the corpus through an in-process MCP server over
    ``store``; pass ``toolset`` to use another MCP connection (e.g. a
    :class:`~pydantic_ai.mcp.MCPToolset` for a remote server). ``models`` maps a
    role name (or ``"*"`` for all roles) to a model name or instance, overriding
    the settings; tests use it to supply scripted models.
    """

    def __init__(
        self,
        store: CorpusStore,
        target: str,
        settings: Settings,
        models: Mapping[str, Model | str] | None = None,
        toolset: AbstractToolset[Any] | None = None,
    ):
        self.store = store
        self.target = target
        self.settings = settings
        self.models = dict(models or {})
        self.corpus_tool = settings.corpus_tool and bool(store.text_corpora)
        self.toolset = toolset or MCPToolset(build_server(store))
        self._agents: dict[str, Agent[Batch, Answer]] = {}
        self.usage: dict[str, ModelUsage] = {}
        """Token use and estimated cost per model."""
        self.role_usage: dict[str, ModelUsage] = {}
        """Token use and estimated cost per role."""

    def relations(self) -> str:
        """The trusted DEPREL inventory as one line, for the syntax roles' instructions."""
        values = self.store.allowed_values("deprel", limit=200)
        return ", ".join(f"{v.value} {v.count}" for v in values)

    @property
    def trusted(self) -> list[str]:
        return [info.name for info in self.store.list_sources() if info.mode == "ro"]

    def model(self, role: str) -> Model | str:
        return self.models.get(role) or self.models.get("*") or self.settings.model_for(role)

    def agent(self, role: str) -> Agent[Batch, Answer]:
        if role not in self._agents:
            spec = ROLES[role]
            output_type = TaskReport if spec.edits_with_tools else Scores if spec.is_scorer else Annotation
            agent = Agent(
                self.model(role),
                name=role,
                deps_type=Batch,
                instructions=instructions(
                    spec, self.target, self.trusted, self.settings.role(role).instructions, self.relations(),
                    list(self.store.text_corpora),
                    self.corpus_tool,
                ),
                output_type=output_type,
                toolsets=[role_toolset(self.toolset, spec, self.target, CORPUS_TOOLS if self.corpus_tool else frozenset())],
                retries=3,
                model_settings=self.model_settings(role),
                defer_model_check=True,
            )
            if not spec.edits_with_tools:
                applier = Applier(self.store, spec)
                apply = applier.scores if spec.is_scorer else applier.annotation

                @agent.output_validator
                def check_and_apply(ctx: RunContext[Batch], answer: Annotation | Scores) -> Annotation | Scores:
                    return apply(answer, ctx.deps)

            self._agents[role] = agent
        return self._agents[role]

    def model_settings(self, role: str) -> ModelSettings:
        settings = ModelSettings(cache=self.settings.cache)
        thinking = self.settings.thinking_for(role)
        if thinking is not None:
            settings["thinking"] = thinking
        return settings

    @staticmethod
    def model_name(model: Model | str) -> str:
        return model if isinstance(model, str) else model.model_name

    async def __aenter__(self) -> AgentTeam:
        await self.toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.toolset.__aexit__(*args)

    async def run(
        self, role: str, prompt: str, batch: Batch, usage: ModelUsage | None = None, retries: list[str] | None = None
    ) -> Answer:
        """Run ``role`` on a task; for compact answers, the edits are applied when the answer is accepted.

        The task's token use is added to the per-model and per-role totals and,
        if given, to ``usage``; the retry messages sent to the model (why a
        request was repeated) are appended to ``retries``.
        """
        result = await self.agent(role).run(
            prompt, deps=batch, usage_limits=UsageLimits(request_limit=self.settings.request_limit)
        )
        name = self.model_name(self.model(role))
        task = ModelUsage()
        task.add(result.usage, name)
        self.usage.setdefault(name, ModelUsage(cost=0.0)).merge(task)
        self.role_usage.setdefault(role, ModelUsage(cost=0.0)).merge(task)
        if usage is not None:
            usage.merge(task)
        if retries is not None:
            retries += [
                str(part.content)[:500]
                for message in result.all_messages()
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, RetryPromptPart)
            ]
        return result.output

    def total(self) -> ModelUsage:
        """Usage summed over all models (cost is None if any model has no price)."""
        total = ModelUsage(cost=0.0)
        for usage in self.usage.values():
            total.merge(usage)
        return total
