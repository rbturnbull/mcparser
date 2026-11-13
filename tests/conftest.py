from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from pydantic_ai import models

from mcparser.corpus import CorpusStore, parse_conllu

# GitHub Actions makes typer/rich force colours and boxes, which breaks substring assertions on CLI output
for _name in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS"):
    os.environ.pop(_name, None)
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

models.ALLOW_MODEL_REQUESTS = False  # never call a real model from the tests

DATA = Path(__file__).parent / "data"


@pytest.fixture
def train_path(tmp_path: Path) -> Path:
    path = tmp_path / "train.conllu"
    shutil.copy(DATA / "train.conllu", path)
    return path


@pytest.fixture
def pred_path(tmp_path: Path) -> Path:
    path = tmp_path / "pred.conllu"
    shutil.copy(DATA / "pred.conllu", path)
    return path


@pytest.fixture
def store(train_path: Path, pred_path: Path) -> CorpusStore:
    store = CorpusStore()
    store.add_source("train", train_path)
    store.add_source("pred", pred_path, mode="rw")
    return store


def sentence(text: str):
    """Parse a single sentence written with spaces between columns (``|`` kept)."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        lines.append(line if line.startswith("#") else "\t".join(line.split()))
    (parsed,) = parse_conllu("\n".join(lines))
    return parsed


@pytest.fixture
def anyio_backend():
    return "asyncio"
