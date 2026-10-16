import json

import pytest
import yaml

from rail_rag import paths, prompts
from rail_rag.evaluation import __main__ as cli
from rail_rag.evaluation.judge import RagasJudge, strip_markers
from rail_rag.evaluation.metrics import (
    aggregate,
    context_precision,
    context_recall,
    gate,
    grounded,
    per_document,
    score_item,
)
from rail_rag.graph import GraphConfig
from rail_rag.pipeline import Pipeline
from rail_rag.vectorstore import QdrantStore
from tests.fakes import FakeLLM, HashEmbedder, HashSparseEncoder
from tests.test_graph import PASSAGES

# --- judge-free metrics ---------------------------------------------------------------------


def test_context_precision_is_rank_weighted():
    gold = {"a", "b"}
    assert context_precision(["a", "b", "x"], gold) == 1.0
    assert context_precision(["x", "a"], gold) == pytest.approx(0.5)  # precision@2 = 1/2
    assert context_precision(["x", "a", "b"], gold) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert context_precision(["x", "y"], gold) == 0.0
    assert context_precision([], gold) == 0.0


def test_context_recall_counts_gold_passages_found():
    assert context_recall(["a", "x"], {"a", "b"}) == 0.5
    assert context_recall([], {"a"}) == 0.0


def test_grounding_requires_citations_all_from_expected_docs():
    assert grounded(["tsi-srt"], ["tsi-srt"])
    assert not grounded(["tsi-srt", "tsi-prm"], ["tsi-srt"])  # right answer, wrong document
    assert not grounded([], ["tsi-srt"])


def item(id_, type_="answerable", gold=("p1",), docs=("d1",)):
    return {
        "id": id_,
        "type": type_,
        "supporting_passages": list(gold),
        "expected_docs": list(docs),
    }


def record(id_, refused=False, ctx=("p1",), cited=("d1",)):
    return {
        "id": id_,
        "refused": refused,
        "contexts": [{"passage_id": p} for p in ctx],
        "hits": [{"passage_id": p} for p in ctx],
        "citations": [{"doc_id": d} for d in cited],
    }


def test_aggregate_separates_false_refusals_from_judge_metrics():
    scored = [
        score_item(item("q1"), record("q1")) | {"faithfulness": 1.0, "answer_relevancy": 0.8},
        score_item(item("q2"), record("q2", refused=True, ctx=(), cited=())),
        score_item(item("q3", "unanswerable", (), ()), record("q3", refused=True, ctx=())),
        score_item(item("q4", "unanswerable", (), ()), record("q4", refused=False)),
    ]
    m = aggregate(scored)
    assert m["faithfulness"] == 1.0  # refused q2 is not averaged in...
    assert m["answered_rate"] == 0.5  # ...it is counted here instead
    assert m["context_recall"] == 0.5  # and its empty context scores 0
    assert m["refusal_rate"] == 0.5
    assert m["grounding"] == 1.0


def test_per_document_reports_other_cited_documents():
    scored = [score_item(item("q1"), record("q1", cited=("d1", "d2")))]
    [row] = per_document(scored)
    assert row == {
        "doc_id": "d1", "questions": 1, "answered": 1, "grounding": 0.0,
        "cited_other_docs": ["d2"],
    }  # fmt: skip


def test_gate_fails_on_low_or_missing_metric():
    thresholds = {"faithfulness": {"min": 0.8}, "refusal_rate": {"min": 0.75}}
    assert gate({"faithfulness": 0.9, "refusal_rate": 1.0}, thresholds) == (True, [])
    ok, failures = gate({"faithfulness": 0.7, "refusal_rate": None}, thresholds)
    assert not ok and len(failures) == 2


def test_thresholds_file_covers_every_gated_metric():
    rules = yaml.safe_load((paths.REPO_ROOT / "eval" / "thresholds.yaml").read_text())["metrics"]
    assert set(rules) == {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "grounding", "answered_rate", "refusal_rate",
    }  # fmt: skip
    assert all(0 < r["min"] <= 1 for r in rules.values())


def test_strip_markers():
    assert strip_markers("It is 70 dB [1][2]. Also [3].") == "It is 70 dB. Also."


# --- RAGAS adapter, built offline -----------------------------------------------------------


def test_ragas_judge_builds_with_real_sdk_offline(tmp_path):
    judge = RagasJudge("groq", "dummy-key", HashEmbedder(), cache_dir=tmp_path)
    assert judge.name == "groq:openai/gpt-oss-120b"
    with pytest.raises(ValueError, match="unknown judge"):
        RagasJudge("openai", "k", HashEmbedder())


# --- end to end CLI with fakes --------------------------------------------------------------


class FakeJudge:
    def __init__(self, provider, api_key, embedder, model=None, cache_dir=None):
        self.name = f"{provider}:fake"
        self.faith = 0.95 if provider == "groq" else 1.0

    async def faithfulness(self, question, answer, contexts):
        return self.faith

    async def answer_relevancy(self, question, answer):
        return 0.9


def fake_pipeline():
    e = HashEmbedder()
    store = QdrantStore(":memory:", sparse_encoder=HashSparseEncoder())
    store.reset(e.dim)
    store.upsert(PASSAGES, e.embed_documents([p.embed_text for p in PASSAGES]))

    def grade(user):  # relevant only for the tunnel question; refuse everything else
        question = user.splitlines()[0]  # the prompt also contains passages that mention tunnels
        return json.dumps({"relevant": [1] if "tunnel" in question.lower() else []})

    llm = FakeLLM({
        prompts.REWRITE: lambda user: user,
        prompts.GRADE: grade,
        prompts.GENERATE: json.dumps({"answer": "90 minutes [1].", "insufficient": False}),
    })  # fmt: skip
    return Pipeline(llm, e, store, GraphConfig(k=3, min_score=0.0), store_name="qdrant")


@pytest.fixture
def golden(tmp_path):
    items = [
        item("q1", gold=[PASSAGES[0].id], docs=["tsi-srt"])
        | {"question": "tunnel emergency lighting duration"},
        item("q2", "unanswerable", (), ()) | {"question": "capital of France"},
    ]
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in items))
    return path


def run_cli(monkeypatch, tmp_path, golden, thresholds, *extra):
    monkeypatch.setattr(cli.Pipeline, "from_settings", classmethod(lambda cls, s: fake_pipeline()))
    monkeypatch.setattr(cli, "RagasJudge", FakeJudge)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda name: HashEmbedder())
    th = tmp_path / "th.yaml"
    th.write_text(yaml.safe_dump({"metrics": thresholds}))
    out, summary = tmp_path / "run.json", tmp_path / "summary.md"
    code = cli.main(
        ["--golden", str(golden), "--thresholds", str(th), "--out", str(out),
         "--summary", str(summary), *extra]
    )  # fmt: skip
    return code, json.loads(out.read_text()), summary.read_text()


def test_cli_passes_writes_results_and_summary(monkeypatch, tmp_path, golden):
    code, result, summary = run_cli(
        monkeypatch, tmp_path, golden, {"faithfulness": {"min": 0.9}, "refusal_rate": {"min": 1.0}}
    )
    assert code == 0 and result["gate"]["passed"]
    m = result["judges"]["groq:fake"]["metrics"]
    assert m["faithfulness"] == 0.95 and m["refusal_rate"] == 1.0 and m["context_recall"] == 1.0
    assert result["meta"]["generator"] == "fake:scripted"
    assert (
        "## Evaluation PASSED" in summary
        and "| Faithfulness (judge) | 0.95 | ≥ 0.90 | ✅ |" in summary
    )


def test_cli_fails_build_when_metric_below_threshold(monkeypatch, tmp_path, golden):
    code, result, summary = run_cli(monkeypatch, tmp_path, golden, {"faithfulness": {"min": 0.99}})
    assert code == 1
    assert result["gate"]["failures"] == ["faithfulness: 0.950 < 0.990"]
    assert "## Evaluation FAILED" in summary


def test_cli_compares_two_judges_on_the_same_answers(monkeypatch, tmp_path, golden):
    _, result, summary = run_cli(
        monkeypatch, tmp_path, golden, {"faithfulness": {"min": 0.5}}, "--judges", "groq,gemini"
    )
    assert list(result["judges"]) == ["groq:fake", "gemini:fake"]
    assert "Same answers, different judges" in summary and "+0.05" in summary


def test_cli_refuses_same_family_gating_judge(monkeypatch, tmp_path, golden):
    with pytest.raises(SystemExit, match="Self-preference"):
        run_cli(monkeypatch, tmp_path, golden, {}, "--judges", "gemini")


def test_retrieval_compare_reports_recall_rank_and_floor_changes():
    from rail_rag.evaluation.retrieval import compare
    from rail_rag.vectorstore import SearchHit

    items = [
        {"id": "a", "type": "answerable", "supporting_passages": [PASSAGES[0].id]},
        {"id": "u", "type": "unanswerable", "supporting_passages": []},
    ]
    canned = {  # (query, mode) -> [(passage, cosine)]
        ("qa", "dense"): [(PASSAGES[1], 0.7)],
        ("qa", "hybrid"): [(PASSAGES[1], 0.7), (PASSAGES[0], 0.5)],
        ("qu", "dense"): [(PASSAGES[1], 0.65)],
        ("qu", "hybrid"): [(PASSAGES[2], 0.55)],
    }

    def search(query, mode):
        return [SearchHit(p, s) for p, s in canned[(query, mode)]]

    out = compare(items, {"a": "qa", "u": "qu"}, search, k=2, floor=0.6)
    assert out["summary"]["dense"]["retrieval_recall_at_k"] == 0.0
    assert out["summary"]["hybrid"]["retrieval_recall_at_k"] == 1.0
    assert out["items"][0]["hybrid"]["gold_rank"] == 2
    assert out["summary"]["floor_decisions_changed"] == ["u"]
