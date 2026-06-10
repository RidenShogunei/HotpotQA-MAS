"""Local HotpotQA environment with search/read tools over distractor context.

Features:
- BM25 search for better document retrieval
- Robust error handling with graceful degradation
- Keyword fallback when BM25 fails
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class HotpotDoc:
    doc_id: str
    title: str
    text: str
    sentences: List[str]


@dataclass
class HotpotTask:
    task_id: str
    question: str
    answer: str
    support_doc_ids: List[str]
    support_titles: List[str]
    docs: List[HotpotDoc]
    level: str = ""
    task_type: str = ""


# ── BM25 implementation ─────────────────────────────────────────

class BM25:
    """Simple BM25 scorer with default k1=1.5, b=0.75."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n_docs = len(corpus)
        self.avgdl = sum(len(d.split()) for d in corpus) / max(self.n_docs, 1)
        self._doc_lens = [len(d.split()) for d in corpus]
        self._idf_cache: Dict[str, float] = {}
        self._build_idf()

    def _build_idf(self):
        df: Dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc.split()):
                df[term] = df.get(term, 0) + 1
        for term, count in df.items():
            self._idf_cache[term] = math.log(
                (self.n_docs - count + 0.5) / (count + 0.5) + 1.0
            )

    def _idf(self, term: str) -> float:
        return self._idf_cache.get(term, 0.0)

    def score(self, query: str) -> List[float]:
        query_tokens = query.split()
        scores = []
        for i, doc in enumerate(self.corpus):
            doc_tokens = doc.split()
            doc_len = self._doc_lens[i]
            tf: Dict[str, int] = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                idf = self._idf(term)
                num = tf[term] * (self.k1 + 1)
                denom = tf[term] + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
                score += idf * num / max(denom, 1e-6)
            scores.append(score)
        return scores


# ── Environment ──────────────────────────────────────────────────

class HotpotQAEnvironment:
    """A real multi-hop QA environment using each HotpotQA row's local context."""

    def __init__(self, tasks: List[HotpotTask]):
        self.tasks = tasks

    @classmethod
    def from_jsonl(cls, path: str, limit: Optional[int] = None):
        tasks = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if limit is not None and len(tasks) >= limit:
                        break
                    raw = json.loads(line)
                    docs = [
                        HotpotDoc(
                            doc_id=doc["doc_id"],
                            title=doc["title"],
                            text=doc["text"],
                            sentences=doc.get("sentences", []),
                        )
                        for doc in raw["docs"]
                    ]
                    tasks.append(HotpotTask(
                        task_id=raw["task_id"],
                        question=raw["question"],
                        answer=raw["answer"],
                        support_doc_ids=raw.get("support_doc_ids", []),
                        support_titles=raw.get("support_titles", []),
                        docs=docs,
                        level=raw.get("level", ""),
                        task_type=raw.get("type", ""),
                    ))
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to load {path}: {e}")
        return cls(tasks)

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def token_f1(prediction: str, answer: str) -> float:
        pred = HotpotQAEnvironment.normalize(prediction).split()
        gold = HotpotQAEnvironment.normalize(answer).split()
        if not pred or not gold:
            return float(pred == gold)
        common = {}
        for tok in pred:
            common[tok] = common.get(tok, 0) + 1
        overlap = 0
        for tok in gold:
            if common.get(tok, 0) > 0:
                overlap += 1
                common[tok] -= 1
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def parse_tool_call(text: str) -> Optional[Tuple[str, str]]:
        m = re.search(r"\[tool_call\]\s*(.*?)\s*\[/tool_call\]", text, re.DOTALL)
        if not m:
            return None
        body = m.group(1).strip()
        call = re.match(r"(?is)(search|read)\s*\((.*)\)\s*$", body)
        if not call:
            return None
        return call.group(1).lower(), call.group(2).strip().strip('"').strip("'")

    @staticmethod
    def _keyword_search(task: HotpotTask, query: str, k: int = 5) -> List[Tuple[float, str, str]]:
        """Fallback: keyword overlap scoring."""
        terms = set(HotpotQAEnvironment.normalize(query).split())
        scored = []
        for doc in task.docs:
            hay = HotpotQAEnvironment.normalize(f"{doc.title} {doc.text}")
            score = sum(1 for term in terms if term and term in hay)
            if score:
                scored.append((float(score), doc.doc_id, doc.title))
        return scored

    @staticmethod
    def _bm25_search(task: HotpotTask, query: str, k: int = 5) -> List[Tuple[float, str, str]]:
        """Primary: BM25 search."""
        normalized = HotpotQAEnvironment.normalize(query)
        corpus = [HotpotQAEnvironment.normalize(f"{d.title} {d.text}") for d in task.docs]
        if not corpus:
            return []
        try:
            bm25 = BM25(corpus)
            scores = bm25.score(normalized)
            scored = [(scores[i], task.docs[i].doc_id, task.docs[i].title)
                      for i in range(len(scores)) if scores[i] > 0]
            scored.sort(key=lambda x: (-x[0], x[1]))
            return scored[:k]
        except Exception:
            # Fallback to keyword search on error
            return HotpotQAEnvironment._keyword_search(task, query, k)

    @staticmethod
    def search(task: HotpotTask, query: str, k: int = 5, use_bm25: bool = True) -> str:
        if use_bm25:
            scored = HotpotQAEnvironment._bm25_search(task, query, k)
            if not scored:
                scored = HotpotQAEnvironment._keyword_search(task, query, k)
        else:
            scored = HotpotQAEnvironment._keyword_search(task, query, k)

        scored.sort(key=lambda x: (-x[0], x[1]))
        rows = [
            {
                "doc_id": doc_id,
                "title": title,
                "hint": f'Use read("{doc_id}") to inspect this document.',
            }
            for _, doc_id, title in scored[:k]
        ]
        return json.dumps({"results": rows}, ensure_ascii=False)

    @staticmethod
    def read(task: HotpotTask, doc_id: str) -> Tuple[bool, str]:
        clean = doc_id.strip()
        if not re.fullmatch(r"D\d{2}", clean):
            return False, "Invalid doc_id"
        for doc in task.docs:
            if doc.doc_id == clean:
                return True, json.dumps(
                    {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text},
                    ensure_ascii=False,
                )
        return False, "Unknown doc_id"

    @staticmethod
    def execute_tool(task: HotpotTask, tool_call_text: str) -> Tuple[bool, str]:
        if not isinstance(task, HotpotTask):
            return False, "Invalid task object"
        try:
            parsed = HotpotQAEnvironment.parse_tool_call(tool_call_text)
            if parsed is None:
                return False, "No valid tool_call found"
            tool, arg = parsed
            if tool == "search":
                return True, HotpotQAEnvironment.search(task, arg)
            if tool == "read":
                return HotpotQAEnvironment.read(task, arg)
            return False, f"Unknown tool: {tool}"
        except Exception as e:
            return False, f"Tool execution error: {e}"

    @staticmethod
    def extract_result(text: str) -> str:
        matches = re.findall(r"<result>\s*(.*?)\s*</result>", text, re.DOTALL)
        return matches[-1].strip() if matches else ""

    @staticmethod
    def extract_doc_ids(text: str) -> List[str]:
        return sorted(set(re.findall(r"\bD\d{2}\b", text)))

    @staticmethod
    def reward(task: HotpotTask, response: str) -> Dict[str, float]:
        result = HotpotQAEnvironment.extract_result(response)
        answer_text = re.split(r"\|\s*evidence\s*:", result, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        answer_f1 = HotpotQAEnvironment.token_f1(answer_text, task.answer)
        pred_docs = set(HotpotQAEnvironment.extract_doc_ids(response))
        gold_docs = set(task.support_doc_ids)
        evidence = len(pred_docs & gold_docs) / max(len(gold_docs), 1)
        tool_valid = 1.0 if HotpotQAEnvironment.parse_tool_call(response) else 0.0
        total = 0.7 * answer_f1 + 0.2 * evidence + 0.1 * tool_valid
        return {"total": total, "answer_f1": answer_f1, "evidence": evidence, "tool_valid": tool_valid}


if __name__ == "__main__":
    default_path = Path("data/base") / "train.jsonl"
    if default_path.exists():
        env = HotpotQAEnvironment.from_jsonl(str(default_path), limit=2)
        for task in env.tasks:
            print("=" * 80)
            print(task.question)
            print("answer:", task.answer)
            print("support:", task.support_doc_ids, task.support_titles)
            print("BM25 search:", HotpotQAEnvironment.search(task, task.question))
            print("Keyword search:", HotpotQAEnvironment.search(task, task.question, use_bm25=False))
    else:
        print("Run prepare_hotpotqa_data.py first.")
