import pytest

from llmparser.database.database_base import DatabaseBase


def test_database_base_methods_raise_not_implemented() -> None:
    base = DatabaseBase.__new__(DatabaseBase)

    with pytest.raises(NotImplementedError):
        base.load("path")

    with pytest.raises(NotImplementedError):
        base("query")

    with pytest.raises(NotImplementedError):
        DatabaseBase.build("input", "output")
