from pathlib import Path

from llmparser.io import write_yaml


def test_write_yaml_single_item_preserves_key_order(tmp_path: Path) -> None:
    output_path = tmp_path / "out.yml"
    data = {
        "item1": {
            "b": 1,
            "a": 2,
            "text": "hello",
        }
    }

    write_yaml(data, output_path)

    content = output_path.read_text()
    assert "item1:" in content
    assert "text: hello" in content
    assert "b: 1" in content
    assert "a: 2" in content
    assert content.index("b: 1") < content.index("a: 2")
    assert "text: |" not in content


def test_write_yaml_multiline_string_uses_literal_block(tmp_path: Path) -> None:
    output_path = tmp_path / "out.yml"
    data = {
        "item1": {
            "text": "line1\nline2",
        }
    }

    write_yaml(data, output_path)

    content = output_path.read_text()
    assert "text: |" in content
    assert "  line1" in content
    assert "  line2" in content


def test_write_yaml_multiple_items_are_separated(tmp_path: Path) -> None:
    output_path = tmp_path / "out.yml"
    data = {
        "first": {"value": 1},
        "second": {"value": 2},
    }

    write_yaml(data, output_path)

    content = output_path.read_text().rstrip()
    blocks = content.split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].lstrip().startswith("first:")
    assert blocks[1].lstrip().startswith("second:")
