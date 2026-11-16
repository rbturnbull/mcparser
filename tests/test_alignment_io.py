import json

import pytest
from typer.testing import CliRunner

from conftest import DATA
from mcparser.cli import app
from mcparser.corpus import EditError
from mcparser.corpus.alignment import Link
from mcparser.corpus.alignment_io import AlignmentFormat, format_links, read_links

runner = CliRunner()
IDS = ["s1", "s2", "s3"]
A, B = str(DATA / "parallel_a.conllu"), str(DATA / "parallel_b.conllu")


def keys(pairs):
    return {sent_id: [(l.source, l.target, l.sure) for l in links] for sent_id, links in pairs.items()}


def test_read_text_bases_and_suffixes():
    zero = read_links("0-0S 2-4P 1-2\n\n0-1\n", IDS)
    assert keys(zero) == {"s1": [("1", "1", True), ("3", "5", False), ("2", "3", True)], "s3": [("1", "2", True)]}
    one = read_links("1-1\n\n\n", IDS, base=1)
    assert keys(one) == {"s1": [("1", "1", True)]}


def test_text_round_trip():
    pairs = {"s1": [Link(source="1", target="1"), Link(source="3", target="5", sure=False)], "s2": [Link(source="2", target="2")]}
    text = format_links(pairs, IDS)
    assert text == "0-0S 2-4P\n1-1S\n\n"
    assert keys(read_links(text, IDS)) == keys(pairs)
    assert format_links(pairs, IDS, base=1, suffix=False) == "1-1 3-5\n2-2\n\n"


def test_yaml_round_trip():
    pairs = {"s1": [Link(source="1", target="2"), Link(source="2", target="3", sure=False)]}
    text = format_links(pairs, IDS, AlignmentFormat.YAML)
    assert text == "s1: 1-2S 2-3P\n"
    assert keys(read_links(text, IDS, "yaml")) == keys(pairs)
    assert read_links("", IDS, "yaml") == {}
    assert read_links("s1:\ns2: 1-1\n", IDS, "yaml").keys() == {"s2"}
    with pytest.raises(EditError, match="should map"):
        read_links("- 1-1", IDS, "yaml")


def test_errors():
    with pytest.raises(EditError, match="has 1 lines"):
        read_links("0-0\n", IDS)
    with pytest.raises(EditError, match="invalid link 'x'"):
        read_links("x\n\n\n", IDS)
    with pytest.raises(EditError, match="below the first word"):
        read_links("0-0\n\n\n", IDS, base=1)
    with pytest.raises(EditError, match="not a plain word number"):
        format_links({"s1": [Link(source="3.1", target="1")]}, IDS)


def test_import_export_commands(tmp_path):
    source = tmp_path / "gold.txt"
    source.write_text("0-0S 1-2P\n\n\n", encoding="utf-8")
    out = tmp_path / "set.jsonl"
    result = runner.invoke(app, ["align", "import", str(source), "-a", A, "-b", B, "-o", str(out), "--confidence", "1"])
    assert result.exit_code == 0 and "2 links in 1 pairs" in result.output
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["sent_id"] == "s1" and record["links"][1] == {"source": "2", "target": "3", "sure": False, "confidence": 1.0}
    assert runner.invoke(app, ["align", "import", str(source), "-a", A, "-b", B, "-o", str(out)]).exit_code == 1  # exists
    assert runner.invoke(app, ["align", "import", str(source), "-a", A, "-b", B, "-o", str(out), "--force"]).exit_code == 0

    shown = runner.invoke(app, ["align", "export", str(out), "-a", A, "-b", B])
    assert shown.output == "0-0S 1-2P\n\n\n"
    yaml_out = tmp_path / "set.yaml"
    assert runner.invoke(app, ["align", "export", str(out), "-a", A, "-b", B, "-f", "yaml", "-o", str(yaml_out)]).exit_code == 0
    assert yaml_out.read_text(encoding="utf-8") == "s1: 1-1S 2-3P\n"
    plain = runner.invoke(app, ["align", "export", str(out), "-a", A, "-b", B, "--plain", "--base", "1"])
    assert plain.output == "1-1 2-3\n\n\n"
    confident = runner.invoke(app, ["align", "export", str(out), "-a", A, "-b", B, "--min-confidence", "1.0"])
    assert confident.exit_code == 0
    low = runner.invoke(app, ["align", "export", str(out), "-a", A, "-b", B, "--min-confidence", "0.5", "-f", "yaml"])
    assert low.output == "s1: 1-1S 2-3P\n"

    back = tmp_path / "back.jsonl"
    assert runner.invoke(app, ["align", "import", str(yaml_out), "-a", A, "-b", B, "-o", str(back), "-f", "yaml"]).exit_code == 0
    assert json.loads(back.read_text(encoding="utf-8"))["links"][0] == {"source": "1", "target": "1", "sure": True}


def test_import_and_export_errors(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("0-9S\n\n\n", encoding="utf-8")
    result = runner.invoke(app, ["align", "import", str(bad), "-a", A, "-b", B, "-o", str(tmp_path / "x.jsonl")])
    assert result.exit_code == 1 and "missing word" in result.output
    missing = runner.invoke(app, ["align", "import", str(tmp_path / "none.txt"), "-a", A, "-b", B, "-o", str(tmp_path / "x.jsonl")])
    assert missing.exit_code == 1
    export = runner.invoke(app, ["align", "export", str(tmp_path / "none.jsonl"), "-a", A, "-b", B])
    assert export.exit_code == 1 and "Error" in export.output
