"""Run settings: which model each agent uses, thresholds and rounds.

Settings can be loaded from YAML, e.g.::

    model: anthropic:claude-opus-5-5
    threshold: 0.8
    rounds: 3
    roles:
      lemma:
        model: anthropic:claude-sonnet-5-5
      deps:
        enabled: true
      head:
        instructions: Attach particles to the verb they modify.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from .agents.roles import DEFAULT_DISABLED, ROLES

DEFAULT_MODEL = "anthropic:claude-opus-5-5"


class RoleSettings(BaseModel):
    model: str | None = Field(None, description="Model for this role; default the global model.")
    enabled: bool | None = Field(None, description="Run this role; default true except for deps.")
    instructions: str | None = Field(None, description="Extra instructions appended to the role's prompt.")


class Settings(BaseModel):
    model: str = Field(DEFAULT_MODEL, description="pydantic-ai model name used by every role unless overridden.")
    threshold: float = Field(0.8, ge=0, le=1, description="Cells below this confidence are refined.")
    rounds: int = Field(3, ge=1, description="Maximum number of score/refine rounds.")
    concurrency: int = Field(1, ge=1, description="Sentences processed at the same time.")
    request_limit: int = Field(60, ge=1, description="Maximum model requests per agent task.")
    cache: bool = Field(True, description="Use prompt caching (tool definitions, instructions, conversation).")
    roles: dict[str, RoleSettings] = Field(default_factory=dict)

    @field_validator("roles")
    @classmethod
    def known_roles(cls, roles: dict[str, RoleSettings]) -> dict[str, RoleSettings]:
        unknown = set(roles) - set(ROLES)
        if unknown:
            raise ValueError(f"Unknown roles {sorted(unknown)}; expected some of {list(ROLES)}.")
        return roles

    @classmethod
    def from_file(cls, path: Path | str) -> Settings:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True)

    def role(self, name: str) -> RoleSettings:
        return self.roles.get(name, RoleSettings())

    def model_for(self, name: str) -> str:
        return self.role(name).model or self.model

    def enabled(self, name: str) -> bool:
        enabled = self.role(name).enabled
        return name not in DEFAULT_DISABLED if enabled is None else enabled

    def with_overrides(
        self,
        model: str | None = None,
        role_models: dict[str, str] | None = None,
        only_roles: list[str] | None = None,
        **fields: object,
    ) -> Settings:
        """A copy with command-line overrides applied."""
        updated = self.model_copy(deep=True, update={key: value for key, value in fields.items() if value is not None})
        if model:
            updated.model = model
        for name, role_model in (role_models or {}).items():
            updated.roles.setdefault(name, RoleSettings()).model = role_model
        if only_roles:
            for name in ROLES:
                if name != "scorer":
                    updated.roles.setdefault(name, RoleSettings()).enabled = name in only_roles
        return Settings.model_validate(updated.model_dump())
