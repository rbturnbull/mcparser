import pytest
from pydantic import ValidationError

from mcparser.settings import (
    DEFAULT_EFFICIENT_MODEL,
    DEFAULT_LIGHT_MODEL,
    DEFAULT_STRONG_MODEL,
    RoleSettings,
    Settings,
    parse_thinking,
)


def test_defaults():
    settings = Settings()
    assert DEFAULT_STRONG_MODEL == "anthropic:claude-opus-5-5"
    assert DEFAULT_EFFICIENT_MODEL == "anthropic:claude-sonnet-5-5"
    assert settings.model_for("head") == DEFAULT_STRONG_MODEL
    assert settings.model_for("scorer") == DEFAULT_EFFICIENT_MODEL
    assert settings.model_for("upos") == DEFAULT_EFFICIENT_MODEL
    assert settings.model_for("xpos") == DEFAULT_LIGHT_MODEL == "anthropic:claude-haiku-5-5"
    assert (settings.tier("feats"), settings.tier("lemma"), settings.tier("xpos")) == ("efficient", "strong", "light")
    assert (settings.batch_sentences, settings.rules.min_count, settings.rules.min_share) == (8, 5, 0.98)
    assert settings.thinking_for("head") is None
    assert settings.enabled("lemma") and not settings.enabled("deps")


def test_from_file_and_role_overrides(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "strong_model: big\nefficient_model: small\nefficient_roles: [lemma]\nlight_roles: [upos]\nthreshold: 0.7\n"
        "strong_thinking: low\nefficient_thinking: false\nlight_thinking: minimal\nrules: {min_count: 2}\n"
        "roles:\n  lemma:\n    model: special\n  head:\n    thinking: high\n  deps:\n    enabled: true\n  upos:\n    enabled: false\n",
        encoding="utf-8",
    )
    settings = Settings.from_file(path)
    assert (settings.model_for("lemma"), settings.model_for("upos"), settings.threshold) == ("special", DEFAULT_LIGHT_MODEL, 0.7)
    assert (settings.model_for("feats"), settings.thinking_for("upos"), settings.rules.min_count) == ("big", "minimal", 2)
    assert (settings.thinking_for("head"), settings.thinking_for("deprel"), settings.thinking_for("lemma")) == ("high", "low", False)
    assert settings.enabled("deps") and not settings.enabled("upos")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert Settings.from_file(empty) == Settings()


def test_one_model_for_everything():
    settings = Settings(model="m", roles={"head": RoleSettings(model="h")})
    assert (settings.model_for("upos"), settings.model_for("lemma"), settings.model_for("head")) == ("m", "m", "h")


@pytest.mark.parametrize("field", ["roles", "efficient_roles"])
def test_unknown_roles_are_rejected(field):
    value = {"parsing": RoleSettings()} if field == "roles" else ["parsing"]
    with pytest.raises(ValidationError, match="Unknown roles"):
        Settings(**{field: value})


def test_parse_thinking():
    assert [parse_thinking(v) for v in ("off", "None", "on", "TRUE", "medium")] == [False, False, True, True, "medium"]
    with pytest.raises(ValueError, match="Thinking must be"):
        parse_thinking("max")


def test_with_overrides():
    base = Settings(roles={"head": RoleSettings(instructions="be careful")})
    updated = base.with_overrides(
        model="m", role_models={"lemma": "l"}, only_roles=["lemma", "head"], rounds=5, threshold=None,
        efficient_roles=["xpos"], strong_thinking="low",
    )  # fmt: skip
    assert (updated.model, updated.model_for("lemma"), updated.rounds, updated.threshold) == ("m", "l", 5, 0.8)
    assert (updated.efficient_roles, updated.thinking_for("head")) == (["xpos"], "low")
    assert [name for name in ("segmentation", "lemma", "head", "deprel") if updated.enabled(name)] == ["lemma", "head"]
    assert updated.enabled("scorer")
    assert updated.role("head").instructions == "be careful"
    assert base.model is None
    assert base.with_overrides() == base


def test_to_yaml_round_trips(tmp_path):
    settings = Settings(roles={"lemma": RoleSettings(model="x", thinking="low")}, strong_thinking=True)
    path = tmp_path / "s.yaml"
    path.write_text(settings.to_yaml(), encoding="utf-8")
    assert Settings.from_file(path) == settings
