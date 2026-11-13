"""mcparser.evaluate must agree with the official CoNLL 2018 scorer (tests/official/conll18_ud_eval.py)."""

import importlib.util
import io
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mcparser.cli import app
from mcparser.corpus import CorpusError, format_conllu, parse_conllu
from mcparser.evaluate import OFFICIAL, evaluate, official_problems, run_official, scorable

SCRIPT = Path(__file__).parent / "official" / "conll18_ud_eval.py"
DEPRELS = ["nsubj", "obj", "obl", "obl:arg", "nmod:poss", "det", "case", "aux:pass", "cc", "conj", "punct", "root", "advmod", "mark"]
UPOS = ["NOUN", "VERB", "DET", "ADP", "PUNCT"]
FEATS = ["_", "Case=Nom", "Case=Nom|Number=Sing", "Gender=Masc,Neut", "Number=Sing|Case=Nom", "Clitic=Yes", "Case=Nom|Clitic=Yes"]
LEMMAS = ["_", "a", "b", "A"]


@pytest.fixture(scope="module")
def official():
    spec = importlib.util.spec_from_file_location("conll18_ud_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cuts(rng, text, chance):
    """Split ``text`` at random places."""
    pieces, start = [], 0
    for index in range(1, len(text)):
        if rng.random() < chance:
            pieces.append(text[start:index])
            start = index
    return [*pieces, text[start:]]


def segmentation(rng, text, tokens=None):
    """A random segmentation of ``text``: tokens, some of them multiword tokens split into words."""
    units = tokens or cuts(rng, text, 0.35)
    return [(unit, cuts(rng, unit, 0.5) if len(unit) > 1 and rng.random() < 0.3 else [unit]) for unit in units]


def sentence(rng, name, text, units):
    """CoNLL-U text for a random tree and random cells over the given tokens and words."""
    words = [word for _, parts in units for word in parts]
    count = len(words)
    order = rng.sample(range(1, count + 1), count)
    heads = {order[0]: 0}
    for position, word_id in enumerate(order[1:], start=1):
        heads[word_id] = order[rng.randrange(position)]
    lines = [f"# sent_id = {name}", f"# text = {text}"]
    word_id = 0
    for unit, parts in units:
        multiword = len(parts) > 1 or (len(parts) == 1 and rng.random() < 0.1 and False)
        if multiword:
            lines.append(f"{word_id + 1}-{word_id + len(parts)}\t{unit}\t_\t_\t_\t_\t_\t_\t_\t_")
        for part in parts:
            word_id += 1
            cells = [rng.choice(LEMMAS), rng.choice(UPOS), rng.choice(["_", "X", "Y"]), rng.choice(FEATS), str(heads[word_id]), rng.choice(DEPRELS)]
            lines.append("\t".join([str(word_id), part, *cells, "_", "_"]))
    return "\n".join(lines) + "\n\n"


def corpus(seed, sentences=6, same_tokens=False):
    """A gold file and a system file over the same characters but with different tokens, words and cells."""
    rng = random.Random(seed)
    gold, system = [], []
    for number in range(sentences):
        text = "".join(rng.choice("abcde") for _ in range(rng.randint(2, 14)))
        units = segmentation(rng, text)
        gold.append(sentence(rng, f"s{number}", text, units))
        if same_tokens or rng.random() < 0.5:  # same tokens, other words and cells
            tokens = [unit for unit, _ in units]
        else:
            tokens = None
        system.append(sentence(rng, f"s{number}", text, segmentation(rng, text, tokens)))
    return "".join(gold), "".join(system)


def compare(official, gold, system):
    theirs = official.evaluate(official.load_conllu(io.StringIO(gold)), official.load_conllu(io.StringIO(system)))
    ours = evaluate(parse_conllu(system), parse_conllu(gold))
    for metric in OFFICIAL:
        score = ours.score(metric)
        assert (score.correct, score.gold, score.predicted) == (
            theirs[metric].correct, theirs[metric].gold_total, theirs[metric].system_total
        ), metric


@pytest.mark.parametrize("seed", range(150))
def test_same_numbers_as_the_official_scorer(official, seed):
    """Sentence by sentence, with tokens and words that differ in every way."""
    gold, system = corpus(seed)
    for gold_sentence, system_sentence in zip(parse_conllu(gold), parse_conllu(system), strict=True):
        compare(official, format_conllu([gold_sentence]), format_conllu([system_sentence]))


@pytest.mark.parametrize("seed", range(50))
def test_same_numbers_over_a_whole_file(official, seed):
    """The official scorer aligns a whole file at once, which differs from sentence by sentence only when surface tokens differ."""
    compare(official, *corpus(seed, same_tokens=True))


def test_official_features_and_subtypes_are_ignored_unless_strict():
    gold = parse_conllu("# sent_id = a\n1\tx\tx\tNOUN\t_\tCase=Nom|Clitic=Yes\t0\tobl:arg\t_\t_\n\n")
    system = parse_conllu("# sent_id = a\n1\tx\tx\tNOUN\t_\tCase=Nom\t0\tobl\t_\t_\n\n")
    result = evaluate(system, gold, strict=True)
    assert [result.score(m).f1 for m in ("UFeats", "AllTags", "LAS", "UFeatsAll", "AllTagsAll", "LASFull")] == [1, 1, 1, 0, 0, 0]
    assert [score.metric for score in evaluate(system, gold).scores] == list(OFFICIAL)


PROBLEMS = {
    "unparsed": "1\ta\ta\tX\t_\t_\t_\t_\t_\t_\n2\tb\tb\tX\t_\t_\t1\tdep\t_\t_\n",
    "two roots": "1\ta\ta\tX\t_\t_\t0\troot\t_\t_\n2\tb\tb\tX\t_\t_\t0\troot\t_\t_\n",
    "no root": "1\ta\ta\tX\t_\t_\t2\tdep\t_\t_\n2\tb\tb\tX\t_\t_\t1\tdep\t_\t_\n",
    "outside": "1\ta\ta\tX\t_\t_\t0\troot\t_\t_\n2\tb\tb\tX\t_\t_\t9\tdep\t_\t_\n",
    "cycle": "1\ta\ta\tX\t_\t_\t0\troot\t_\t_\n2\tb\tb\tX\t_\t_\t3\tdep\t_\t_\n3\tc\tc\tX\t_\t_\t2\tdep\t_\t_\n",
}


@pytest.mark.parametrize("name", PROBLEMS)
def test_invalid_trees_are_reported_and_repaired(official, name):
    bad = parse_conllu(f"# sent_id = a\n{PROBLEMS[name]}\n")[0]
    assert official_problems(bad)
    repaired = scorable(bad)
    assert repaired is not None and not official_problems(repaired)
    assert bad.to_conllu() == parse_conllu(f"# sent_id = a\n{PROBLEMS[name]}\n")[0].to_conllu()  # the original is untouched
    with pytest.raises(official.UDError):
        official.load_conllu(io.StringIO(format_conllu([bad])))
    official.load_conllu(io.StringIO(format_conllu([repaired])))
    gold = parse_conllu("# sent_id = a\n1\ta\ta\tX\t_\t_\t0\troot\t_\t_\n\n")
    assert evaluate([bad], gold).invalid == ["a"]


def test_valid_trees_need_no_repair():
    good = parse_conllu("# sent_id = a\n1\ta\ta\tX\t_\t_\t0\troot\t_\t_\n2\tb\tb\tX\t_\t_\t1\tdep\t_\t_\n\n")[0]
    assert official_problems(good) == [] and scorable(good) is None


def test_run_official(tmp_path):
    gold = parse_conllu(
        "# sent_id = a\n1\ta\ta\tX\t_\t_\t2\tdep\t_\t_\n2\tb\tb\tX\t_\t_\t0\troot\t_\t_\n\n"
        "# sent_id = b\n1\tc\tc\tX\t_\t_\t0\troot\t_\t_\n\n"
    )
    system = parse_conllu("# sent_id = a\n1\ta\ta\tX\t_\t_\t_\t_\t_\t_\n2\tb\tb\tX\t_\t_\t_\t_\t_\t_\n\n")
    result = run_official(SCRIPT, system, gold)
    assert (result.sentences, result.repaired) == (1, ["a"])
    assert result.scores["UPOS"].f1 == 1 and result.scores["UAS"].f1 == 0  # the repaired heads are wrong here
    assert result.table().splitlines()[0].split() == ["Metric", "Precision", "Recall", "F1"]
    unreadable = parse_conllu("# sent_id = z\n1-2\tab\t_\t_\t_\t_\t_\t_\t_\t_\n1\ta\ta\tX\t_\t_\t0\troot\t_\t_\n\n")
    skipped = run_official(SCRIPT, [*system, *unreadable], [*gold, *unreadable])
    assert (skipped.sentences, skipped.skipped) == (1, ["z"])
    wrong_text = parse_conllu("# sent_id = a\n1\tzz\ta\tX\t_\t_\t0\troot\t_\t_\n\n")
    with pytest.raises(CorpusError, match="refused"):
        run_official(SCRIPT, wrong_text, gold)
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError\n")
    with pytest.raises(RuntimeError):
        run_official(broken, system, gold)
    with pytest.raises(CorpusError, match="Cannot load"):
        run_official(tmp_path / "none.txt", system, gold)


def test_cli_official_and_strict(tmp_path):
    gold = tmp_path / "gold.conllu"
    system = tmp_path / "system.conllu"
    gold.write_text("# sent_id = a\n1\ta\ta\tX\t_\tCase=Nom|Clitic=Yes\t0\troot\t_\t_\n\n", encoding="utf-8")
    system.write_text("# sent_id = a\n1\ta\ta\tX\t_\tCase=Nom\t_\t_\t_\t_\n\n", encoding="utf-8")
    out = CliRunner().invoke(app, ["evaluate", str(system), str(gold), "--official", str(SCRIPT), "--strict"]).output
    assert "would refuse 1 predicted sentences" in out and "UFeatsAll" in out
    assert "Official scorer, 1 sentences (1 repaired for it, 0 gold sentences it cannot read left out)" in out
    as_json = CliRunner().invoke(app, ["evaluate", str(system), str(gold), "--official", str(SCRIPT), "--json"]).output
    assert '"repaired"' in as_json
