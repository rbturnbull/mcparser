import pytest
from pydantic import ValidationError

from llmparser.settings import DEFAULT_MODEL, RoleSettings, Settings


def test_defaults():
    settings = Settings()
    assert settings.model == DEFAULT_MODEL == "anthropic:claude-opus-5-5"
    assert settings.model_for("lemma") == DEFAULT_MODEL
    assert settings.enabled("lemma") and not settings.enabled("deps")


def test_from_file_and_role_overrides(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "model: test-model\nthreshold: 0.7\nroles:\n  lemma:\n    model: small\n  deps:\n    enabled: true\n  upos:\n    enabled: false\n",
        encoding="utf-8",
    )
    settings = Settings.from_file(path)
    assert (settings.model_for("lemma"), settings.model_for("upos"), settings.threshold) == ("small", "test-model", 0.7)
    assert settings.enabled("deps") and not settings.enabled("upos")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert Settings.from_file(empty) == Settings()


def test_unknown_role_is_rejected():
    with pytest.raises(ValidationError, match="Unknown roles"):
        Settings(roles={"syntax": RoleSettings()})


def test_with_overrides():
    base = Settings(roles={"head": RoleSettings(instructions="be careful")})
    updated = base.with_overrides(model="m", role_models={"lemma": "l"}, only_roles=["lemma", "head"], rounds=5, threshold=None)
    assert (updated.model, updated.model_for("lemma"), updated.rounds, updated.threshold) == ("m", "l", 5, 0.8)
    assert [name for name in ("segmentation", "lemma", "head", "deprel") if updated.enabled(name)] == ["lemma", "head"]
    assert updated.enabled("scorer")
    assert updated.role("head").instructions == "be careful"
    assert base.model == DEFAULT_MODEL
    assert base.with_overrides() == base


def test_to_yaml_round_trips(tmp_path):
    settings = Settings(roles={"lemma": RoleSettings(model="x")})
    path = tmp_path / "s.yaml"
    path.write_text(settings.to_yaml(), encoding="utf-8")
    assert Settings.from_file(path) == settings
