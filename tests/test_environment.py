"""Tests for HotpotQA-MAS.

Run: make test  or  python -m pytest tests/ -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from hotpotqa_environment import BM25, HotpotDoc, HotpotQAEnvironment, HotpotTask


# ── BM25 ────────────────────────────────────────────────────────

def test_bm25_empty_corpus():
    bm = BM25([])
    assert bm.n_docs == 0
    assert bm.avgdl == 0.0
    assert bm.score("hello") == []


def test_bm25_basic_scoring():
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "never gonna give you up never gonna let you down",
        "hello world hello again",
    ]
    bm = BM25(corpus)
    scores = bm.score("quick fox")
    # First doc should score highest for "quick fox"
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_idf_rare_terms():
    corpus = ["aaa bbb ccc", "aaa bbb", "aaa"]
    bm = BM25(corpus)
    # "ccc" appears in only 1 doc -> higher IDF
    assert bm._idf("ccc") > bm._idf("aaa")


# ── Environment ──────────────────────────────────────────────────

def _make_task(question="Test?", answer="yes", docs=None, support_ids=None):
    if docs is None:
        docs = [
            HotpotDoc("D00", "Doc Zero", "The answer is yes.", []),
            HotpotDoc("D01", "Doc One", "Completely unrelated text here.", []),
        ]
    if support_ids is None:
        support_ids = ["D00"]
    return HotpotTask(
        task_id="test_1",
        question=question,
        answer=answer,
        support_doc_ids=support_ids,
        support_titles=[d.title for d in docs if d.doc_id in support_ids],
        docs=docs,
    )


def test_search_bm25_finds_relevant():
    task = _make_task(question="answer yes")
    result = json.loads(HotpotQAEnvironment.search(task, "answer yes"))
    doc_ids = [r["doc_id"] for r in result["results"]]
    assert "D00" in doc_ids


def test_search_keyword_fallback():
    task = _make_task()
    result = json.loads(HotpotQAEnvironment.search(task, "answer yes", use_bm25=False))
    doc_ids = [r["doc_id"] for r in result["results"]]
    assert "D00" in doc_ids


def test_read_valid_doc():
    task = _make_task()
    ok, result = HotpotQAEnvironment.read(task, "D00")
    assert ok
    data = json.loads(result)
    assert data["doc_id"] == "D00"


def test_read_invalid_doc_format():
    task = _make_task()
    ok, _ = HotpotQAEnvironment.read(task, "bad_id")
    assert not ok


def test_read_unknown_doc():
    task = _make_task()
    ok, _ = HotpotQAEnvironment.read(task, "D99")
    assert not ok


def test_execute_tool_search():
    task = _make_task()
    ok, result = HotpotQAEnvironment.execute_tool(task, "[tool_call]search(\"answer\")[/tool_call]")
    assert ok
    data = json.loads(result)
    assert len(data["results"]) > 0


def test_execute_tool_read():
    task = _make_task()
    ok, result = HotpotQAEnvironment.execute_tool(task, '[tool_call]read("D00")[/tool_call]')
    assert ok
    assert "answer is yes" in result


def test_execute_tool_invalid():
    task = _make_task()
    ok, result = HotpotQAEnvironment.execute_tool(task, "not a tool call")
    assert not ok


def test_execute_tool_exception_handling():
    ok, result = HotpotQAEnvironment.execute_tool(None, "[tool_call]search(\"test\")[/tool_call]")
    assert not ok


def test_parse_tool_call_search():
    parsed = HotpotQAEnvironment.parse_tool_call('[tool_call]search("hello world")[/tool_call]')
    assert parsed == ("search", "hello world")


def test_parse_tool_call_read():
    parsed = HotpotQAEnvironment.parse_tool_call('[tool_call]read("D05")[/tool_call]')
    assert parsed == ("read", "D05")


def test_parse_tool_call_invalid():
    assert HotpotQAEnvironment.parse_tool_call("no bracket") is None
    assert HotpotQAEnvironment.parse_tool_call("[tool_call]unknown()[/tool_call]") is None


def test_reward_perfect():
    task = _make_task(answer="yes")
    r = HotpotQAEnvironment.reward(task, "<result>yes | evidence: D00</result>")
    assert r["answer_f1"] == 1.0
    assert r["evidence"] == 1.0
    assert r["total"] >= 0.89  # 0.7*1.0 + 0.2*1.0 + 0.1*0 = 0.9 exactly... in a perfect world


def test_reward_wrong_answer():
    task = _make_task(answer="yes")
    r = HotpotQAEnvironment.reward(task, "<result>no | evidence: D00</result>")
    assert r["answer_f1"] < 1.0


def test_token_f1_exact():
    assert HotpotQAEnvironment.token_f1("hello", "hello") == 1.0


def test_token_f1_partial():
    score = HotpotQAEnvironment.token_f1("hello world", "hello there")
    assert 0.0 < score < 1.0


def test_token_f1_empty():
    assert HotpotQAEnvironment.token_f1("", "") == 1.0


def test_normalize():
    result = HotpotQAEnvironment.normalize("The Quick Brown Fox!")
    assert result == "quick brown fox"


def test_extract_result():
    text = "blah <result>answer | evidence: D00</result> extra"
    assert HotpotQAEnvironment.extract_result(text) == "answer | evidence: D00"


def test_extract_doc_ids():
    text = "evidence: D00, D05 and maybe D01"
    assert HotpotQAEnvironment.extract_doc_ids(text) == ["D00", "D01", "D05"]


# ── JSONL loading ───────────────────────────────────────────────

def test_from_jsonl(tmp_path):
    line = json.dumps({
        "task_id": "t1",
        "question": "Q?",
        "answer": "A",
        "support_doc_ids": ["D00"],
        "support_titles": ["Doc Zero"],
        "docs": [{"doc_id": "D00", "title": "Doc Zero", "text": "text", "sentences": []}],
        "level": "easy",
        "type": "comparison",
    })
    p = tmp_path / "test.jsonl"
    p.write_text(line + "\n", encoding="utf-8")
    env = HotpotQAEnvironment.from_jsonl(str(p))
    assert len(env.tasks) == 1
    assert env.tasks[0].task_id == "t1"


def test_from_jsonl_limit(tmp_path):
    lines = "\n".join(
        json.dumps({
            "task_id": f"t{i}",
            "question": "Q",
            "answer": "A",
            "support_doc_ids": [],
            "support_titles": [],
            "docs": [],
        })
        for i in range(10)
    )
    p = tmp_path / "test.jsonl"
    p.write_text(lines + "\n", encoding="utf-8")
    env = HotpotQAEnvironment.from_jsonl(str(p), limit=3)
    assert len(env.tasks) == 3


def test_from_jsonl_corrupt(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        HotpotQAEnvironment.from_jsonl(str(p))
