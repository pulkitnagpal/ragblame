"""Render the synthetic sample results into runs/sample/report.html, to eyeball the report layout."""
from pathlib import Path

from ragblame.report import render
from ragblame.results import load

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    results = load(ROOT / "tests" / "fixtures" / "results_sample.json")
    out = ROOT / "runs" / "sample" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(results))
    print(out)
