"""Run settings: which model each agent uses, thresholds and rounds.

Roles run on one of three model tiers: a **strong** model for the hard
judgements (segmentation, lemmas, syntax), an **efficient** model for roles
that mostly apply the evidence (UPOS, FEATS, scoring), and a **light** model
for lookups (XPOS). Settings can be loaded from YAML, e.g.::

    strong_model: anthropic:claude-opus-5-5
    efficient_model: anthropic:claude-sonnet-5-5
    light_model: anthropic:claude-haiku-5-5
    efficient_roles: [upos, feats, scorer]
    light_roles: [xpos]
    strong_thinking: low          # reasoning effort; omit for the model's default
    batch_sentences: 8            # sentences per agent task
    rules:
      min_count: 5                # examples needed before code settles a cell
      min_share: 0.95
    threshold: 0.8
    rounds: 3
    roles:
      head:
        thinking: medium
      deps:
        enabled: true
      lemma:
        model: anthropic:claude-opus-5-5
        instructions: Write Sumerograms as their Hittite reading when known.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .agents.roles import DEFAULT_DISABLED, JOINT_ROLES, ROLES, SEPARATE_SYNTAX_ROLES
from .rules import RuleSettings

DEFAULT_STRONG_MODEL = "anthropic:claude-opus-5-5"
DEFAULT_EFFICIENT_MODEL = "anthropic:claude-sonnet-5-5"
DEFAULT_LIGHT_MODEL = "anthropic:claude-haiku-5-5"
DEFAULT_EFFICIENT_ROLES = ("upos", "feats", "scorer")
"""Roles that mostly apply the evidence given in their prompt."""
DEFAULT_LIGHT_ROLES = ("xpos",)
"""Roles that are close to a lookup once UPOS and FEATS are known."""
Tier = Literal["strong", "efficient", "light"]

THINKING_LEVELS = ("minimal", "low", "medium", "high", "xhigh")
Thinking = bool | Literal["minimal", "low", "medium", "high", "xhigh"]
"""Reasoning effort: ``False`` (off), ``True`` (provider default) or a level."""


def parse_thinking(value: str) -> Thinking:
    """Parse a command-line thinking value: off, on, or a level."""
    text = value.strip().lower()
    if text in ("off", "false", "no", "none"):
        return False
    if text in ("on", "true", "yes"):
        return True
    if text in THINKING_LEVELS:
        return text  # type: ignore[return-value]
    raise ValueError(f"Thinking must be off, on or one of {', '.join(THINKING_LEVELS)}; got {value!r}.")


class RoleSettings(BaseModel):
    model: str | None = Field(None, description="Model for this role; overrides its tier.")
    thinking: Thinking | None = Field(None, description="Reasoning effort for this role; overrides its tier.")
    enabled: bool | None = Field(None, description="Run this role; default true except for deps.")
    instructions: str | None = Field(None, description="Extra instructions appended to the role's prompt.")


class Settings(BaseModel):
    strong_model: str = Field(DEFAULT_STRONG_MODEL, description="Model for the demanding roles.")
    efficient_model: str = Field(DEFAULT_EFFICIENT_MODEL, description="Cheaper model for the roles in efficient_roles.")
    light_model: str = Field(DEFAULT_LIGHT_MODEL, description="Cheapest model for the roles in light_roles.")
    efficient_roles: list[str] = Field(list(DEFAULT_EFFICIENT_ROLES), description="Roles that use efficient_model.")
    light_roles: list[str] = Field(list(DEFAULT_LIGHT_ROLES), description="Roles that use light_model.")
    strong_thinking: Thinking | None = Field(None, description="Reasoning effort for strong roles; None = model default.")
    efficient_thinking: Thinking | None = Field(None, description="Reasoning effort for efficient roles; None = model default.")
    light_thinking: Thinking | None = Field(None, description="Reasoning effort for light roles; None = model default.")
    model: str | None = Field(None, description="If set, one model for every role (overrides both tiers).")
    threshold: float = Field(0.8, ge=0, le=1, description="Cells below this confidence are refined.")
    rounds: int = Field(3, ge=1, description="Maximum number of score/refine rounds.")
    concurrency: int = Field(1, ge=1, description="Sentences processed at the same time.")
    request_limit: int = Field(60, ge=1, description="Maximum model requests per agent task.")
    cache: bool = Field(True, description="Use prompt caching (tool definitions, instructions, conversation).")
    batch_sentences: int = Field(8, ge=1, description="Maximum sentences per agent task.")
    batch_cells: int = Field(120, ge=1, description="Maximum words with cells to work on per agent task (a single sentence may exceed it).")
    joint_syntax: bool = Field(
        True, description="One agent sets HEAD and DEPREL together (false: separate head and deprel agents)."
    )
    final_scoring: bool = Field(
        False, description="Score the cells changed in the last round once more (costs scorer tasks, changes no values)."
    )
    rules: RuleSettings = Field(default_factory=RuleSettings, description="When code settles cells without a model.")
    roles: dict[str, RoleSettings] = Field(default_factory=dict)

    @field_validator("roles")
    @classmethod
    def known_roles(cls, roles: dict[str, RoleSettings]) -> dict[str, RoleSettings]:
        unknown = set(roles) - set(ROLES)
        if unknown:
            raise ValueError(f"Unknown roles {sorted(unknown)}; expected some of {list(ROLES)}.")
        return roles

    @field_validator("efficient_roles", "light_roles")
    @classmethod
    def known_efficient_roles(cls, names: list[str]) -> list[str]:
        unknown = set(names) - set(ROLES)
        if unknown:
            raise ValueError(f"Unknown roles {sorted(unknown)}; expected some of {list(ROLES)}.")
        return names

    @classmethod
    def from_file(cls, path: Path | str) -> Settings:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True)

    def role(self, name: str) -> RoleSettings:
        return self.roles.get(name, RoleSettings())

    def tier(self, name: str) -> Tier:
        if name in self.light_roles:
            return "light"
        return "efficient" if name in self.efficient_roles else "strong"

    def model_for(self, name: str) -> str:
        if self.role(name).model:
            return self.role(name).model
        if self.model:
            return self.model
        return getattr(self, f"{self.tier(name)}_model")

    def thinking_for(self, name: str) -> Thinking | None:
        override = self.role(name).thinking
        if override is not None:
            return override
        return getattr(self, f"{self.tier(name)}_thinking")

    def enabled(self, name: str) -> bool:
        enabled = self.role(name).enabled
        if enabled is not None:
            return enabled
        if name in JOINT_ROLES:
            return self.joint_syntax
        if name in SEPARATE_SYNTAX_ROLES:
            return not self.joint_syntax
        return name not in DEFAULT_DISABLED

    def with_overrides(
        self,
        model: str | None = None,
        role_models: dict[str, str] | None = None,
        only_roles: list[str] | None = None,
        **fields: object,
    ) -> Settings:
        """A copy with command-line overrides applied (``None`` values are ignored)."""
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
