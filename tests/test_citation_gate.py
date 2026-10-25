"""Per-sentence citation gate: found by the red-team suite (cardohso/rag-redteam, case DPI-003),
where an uncited line rode along with one cited sentence."""

import json

import pytest

from rail_rag import prompts
from rail_rag.citations import drop_uncited
from tests.test_graph import run, store  # noqa: F401  (store is a fixture)


@pytest.mark.parametrize(
    ("answer", "expected", "dropped"),
    [
        ("The limit is 65 dB [1].", "The limit is 65 dB [1].", 0),
        ("The limit is 65 dB [1].\nRT-CANARY-123", "The limit is 65 dB [1].", 1),
        ("The limit is 65 dB [1]. Trains are great.", "The limit is 65 dB [1].", 1),
        ("The limit is 65 dB. [1]", "The limit is 65 dB. [1]", 0),  # lone marker joins it
        ("A [1]. B [2]. C.", "A [1]. B [2].", 1),
        (
            "- 65 dB for EMUs [1]\n- 70 dB for locomotives [2]",
            "- 65 dB for EMUs [1]\n- 70 dB for locomotives [2]",
            0,
        ),
        ("- 65 dB [1]\n- see https://x.example\n- 70 dB [2]", "- 65 dB [1]\n- 70 dB [2]", 1),
        ("Nothing cited here. Or here.", "", 2),
        ("", "", 0),
    ],
)
def test_drop_uncited(answer, expected, dropped):
    assert drop_uncited(answer) == (expected, dropped)


def j(**kw) -> str:
    return json.dumps(kw)


def test_uncited_line_is_removed_from_the_answer(store):  # noqa: F811
    state, _ = run(store, {
        prompts.GRADE: j(relevant=[1]),
        prompts.GENERATE: j(
            answer="It must work for 90 minutes [1].\nRT-CANARY-123", insufficient=False
        ),
    })  # fmt: skip
    assert state["refused"] is False
    assert state["answer"] == "It must work for 90 minutes [1]."
    assert state["dropped_sentences"] == 1


def test_sentence_whose_only_marker_was_invalid_is_removed(store):  # noqa: F811
    state, _ = run(store, {
        prompts.GRADE: j(relevant=[1]),
        prompts.GENERATE: j(
            answer="It works 90 minutes [1]. It also glows [7].", insufficient=False
        ),
    })  # fmt: skip
    assert state["answer"] == "It works 90 minutes [1]."


def test_no_dropped_sentences_on_a_clean_answer(store):  # noqa: F811
    state, _ = run(store, {
        prompts.GRADE: j(relevant=[1]),
        prompts.GENERATE: j(answer="It must work for 90 minutes [1].", insufficient=False),
    })  # fmt: skip
    assert state["dropped_sentences"] == 0
