"""The results file every run writes (runs/<id>/results.json) and the report reads.

Shape:
{
  "id": "2026-11-14T18-02-11",
  "created": "2026-11-14T18:02:11",
  "config": {...Config as a dict...},
  "questions": [
    {"question": {...Question...},
     "trace": {...Trace...} | null,
     "judgement": {...Judgement...} | null,
     "verdict": {...Verdict...} | null,       # null when the answer was correct or diagnosis was skipped
     "error": "..." | null}                    # set when a Claude call failed for this question
  ],
  "summary": {"total": 100, "correct": 61, "partial": 9, "wrong": 30, "error": 0,
              "causes": {"chunking": 11, ...}},
  "calibration": {"n": 30, "agreement": 0.93,
                  "pairs": [{"question_id": "q1", "judge": "correct", "reference": "partial", "answer": "..."}]} | absent,
  "fix_simulation": {"fix": "heading-chunks", "description": "...",
                     "before": {"correct": 61, "partial": 9, "wrong": 30},
                     "after": {"correct": 74, "partial": 6, "wrong": 20},
                     "fixed_ids": [...], "broken_ids": [...]} | absent,
  "bench": {...see bench.py...} | absent
}
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def summarize(entries: list[dict]) -> dict:
    labels = Counter((e.get("judgement") or {}).get("label") for e in entries if not e.get("error"))
    causes = Counter(e["verdict"]["cause"] for e in entries if e.get("verdict"))
    return {"total": len(entries), "correct": labels["correct"], "partial": labels["partial"],
            "wrong": labels["wrong"], "error": sum(1 for e in entries if e.get("error")),
            "causes": dict(causes.most_common())}


def save(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
