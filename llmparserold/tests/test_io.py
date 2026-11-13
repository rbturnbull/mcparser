from pathlib import Path

from llmparser.io import conllu_to_dict, convert_to_yaml, load_file, write_yaml


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


def test_conllu_to_dict_uses_sent_id_and_text() -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"

    data = conllu_to_dict(conllu_path)

    assert "5.7" in data
    assert data["5.7"]["text"].strip() != ""


def test_load_file_reads_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "input.yml"
    yaml_path.write_text("item1:\n  text: hello\n")

    data = load_file(yaml_path)

    assert data["item1"]["text"] == "hello"


def test_load_file_reads_conllu() -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"

    data = load_file(conllu_path)

    assert "5.7" in data
    assert data["5.7"]["text"].strip() != ""


def test_convert_to_yaml_writes_output(tmp_path: Path) -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"
    yaml_path = tmp_path / "out.yml"

    data = convert_to_yaml(conllu_path, yaml_path)

    assert "5.7" in data
    content = yaml_path.read_text()
    assert "'5.7':\n" in content
