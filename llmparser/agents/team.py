"""The pydantic-ai agents, one per role, sharing one MCP connection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import UsageLimits

from ..corpus import CorpusStore
from ..mcp.server import build_server
from ..settings import Settings
from .prompts import instructions
from .roles import ROLES
from .toolset import role_toolset


class TaskReport(BaseModel):
    """What an agent returns when it finishes a task."""

    summary: str = Field(description="One or two sentences on what you changed, confirmed or flagged.")


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
        self.toolset = toolset or MCPToolset(build_server(store))
        self._agents: dict[str, Agent[None, TaskReport]] = {}
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0

    @property
    def trusted(self) -> list[str]:
        return [info.name for info in self.store.list_sources() if info.mode == "ro"]

    def model(self, role: str) -> Model | str:
        return self.models.get(role) or self.models.get("*") or self.settings.model_for(role)

    def agent(self, role: str) -> Agent[None, TaskReport]:
        if role not in self._agents:
            spec = ROLES[role]
            self._agents[role] = Agent(
                self.model(role),
                name=role,
                instructions=instructions(spec, self.target, self.trusted, self.settings.role(role).instructions),
                output_type=TaskReport,
                toolsets=[role_toolset(self.toolset, spec, self.target)],
                retries=3,
                model_settings=ModelSettings(cache=self.settings.cache),
                defer_model_check=True,
            )
        return self._agents[role]

    async def __aenter__(self) -> AgentTeam:
        await self.toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.toolset.__aexit__(*args)

    async def run(self, role: str, prompt: str) -> TaskReport:
        result = await self.agent(role).run(prompt, usage_limits=UsageLimits(request_limit=self.settings.request_limit))
        usage = result.usage
        self.requests += usage.requests
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        return result.output
