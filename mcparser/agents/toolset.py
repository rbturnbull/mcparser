"""Restrict the MCP tools an agent sees and what it may write."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from ..corpus.store import normalise_column
from .compact import compact_result
from .roles import WRITE_TOOLS, Role


@dataclass
class RoleToolset(WrapperToolset[Any]):
    """The MCP toolset as seen by one role.

    Only the role's tools are listed, and results come back as compact text
    (see :mod:`mcparser.agents.compact`). The ``agent`` argument is hidden from the
    model and filled in with the role name, so edits and flags are attributed
    correctly. Writes are limited to the target source and the role's columns;
    the scorer may only send confidences. A refused call is returned to the
    model as a retry prompt explaining why.
    """

    role: Role
    target: str
    extra: frozenset[str] = frozenset()
    """Tools every role may use besides its own (the corpus tool)."""

    @property
    def allowed(self) -> frozenset[str]:
        return self.role.tools | self.extra

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        tools = await self.wrapped.get_tools(ctx)
        visible: dict[str, ToolsetTool[Any]] = {}
        for name, tool in tools.items():
            if name not in self.allowed:
                continue
            schema = tool.tool_def.parameters_json_schema
            properties = schema.get("properties", {})
            if "agent" in properties:
                schema = {**schema, "properties": {k: v for k, v in properties.items() if k != "agent"}}
                tool = replace(tool, tool_def=replace(tool.tool_def, parameters_json_schema=schema))
            visible[name] = tool
        return visible

    def _check_column(self, column: str, has_value: bool) -> None:
        column = normalise_column(column)
        if self.role.is_scorer and has_value:
            raise ModelRetry("As the scorer you may only send confidences; omit 'value' and flag wrong cells instead.")
        if column not in self.role.columns:
            allowed = ", ".join(self.role.columns)
            raise ModelRetry(
                f"You may only change {allowed}. To report a problem with {column}, call flag_issue with column={column!r}."
            )

    def check(self, name: str, args: dict[str, Any]) -> None:
        """Raise :class:`ModelRetry` if the role may not make this call."""
        if name not in self.allowed:
            raise ModelRetry(f"The tool {name!r} is not available to the {self.role.name} agent.")
        if name in WRITE_TOOLS and name != "resolve_flag" and args.get("source") != self.target:
            raise ModelRetry(f"You may only edit the source {self.target!r}.")
        if name == "update_cells":
            for update in args.get("updates") or []:
                self._check_column(str(update.get("column", "")), has_value=update.get("value") is not None)

    @staticmethod
    def normalise(name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Accept ``updates`` sent as a JSON string, as models sometimes do."""
        if name != "update_cells":
            return args
        updates = args.get("updates")
        if isinstance(updates, str):
            try:
                updates = json.loads(updates)
            except json.JSONDecodeError:
                updates = None
        if not isinstance(updates, list) or not all(isinstance(update, dict) for update in updates):
            raise ModelRetry("'updates' must be a list of objects with token_id, column, and value and/or confidence.")
        return {**args, "updates": updates}

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]) -> Any:
        tool_args = self.normalise(name, tool_args)
        self.check(name, tool_args)
        if name in WRITE_TOOLS:
            tool_args = {**tool_args, "agent": self.role.name}
        return compact_result(name, await self.wrapped.call_tool(name, tool_args, ctx, tool), self.role)


def role_toolset(
    toolset: AbstractToolset[Any], role: Role, target: str, extra: frozenset[str] = frozenset()
) -> RoleToolset:
    return RoleToolset(wrapped=toolset, role=role, target=target, extra=extra)
