"""Command-line entry point.

    ragblame ask "How many pulls do I get on a free account?"
    ragblame questions --n 60
    ragblame run                      # answer, judge and diagnose every question, then write the report
    ragblame calibrate runs/<id>      # check the judge against a stricter model
    ragblame fix heading-chunks runs/<id>
    ragblame report runs/<id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ragblame import __version__

DOCS = Path("data/docker")
QUESTIONS = Path("data/questions.jsonl")


def _bot(args):
    from ragblame.corpus import load_docs
    from ragblame.pipeline import Config, DemoBot
    return DemoBot(load_docs(Path(args.docs)), Config())


def _write_report(results: dict, run_dir: Path) -> Path:
    from ragblame.report import render
    out = run_dir / "report.html"
    out.write_text(render(results))
    return out


def cmd_ask(args):
    trace = _bot(args).run(args.question)
    print(trace.answer)
    print("\npacked:", ", ".join(c.id for c in trace.packed))


def cmd_questions(args):
    from ragblame.corpus import load_docs
    from ragblame.questions import save_questions, write_questions
    qs, rejected = write_questions(load_docs(Path(args.docs)), n=args.n, seed=args.seed)
    save_questions(qs, Path(args.out))
    print(f"wrote {len(qs)} questions to {args.out} ({rejected} rejected quotes)")


def cmd_run(args):
    from ragblame.questions import load_questions
    from ragblame.runner import run
    qs = load_questions(Path(args.questions))[: args.limit or None]
    results = run(_bot(args), qs, diagnose_failures=not args.no_diagnose)
    print("report:", _write_report(results, Path("runs") / results["id"]))


def cmd_calibrate(args):
    from ragblame.questions import load_questions
    from ragblame.results import load, save
    from ragblame.runner import calibrate
    run_dir = Path(args.run)
    results = calibrate(load(run_dir / "results.json"), load_questions(Path(args.questions)), n=args.n)
    save(results, run_dir / "results.json")
    print(f"judge agreement with reference labels: {results['calibration']['agreement']} over {results['calibration']['n']}")
    _write_report(results, run_dir)


def cmd_fix(args):
    from ragblame.questions import load_questions
    from ragblame.results import load, save
    from ragblame.runner import simulate_fix
    run_dir = Path(args.run)
    results = load(run_dir / "results.json")
    ids = {e["question"]["id"] for e in results["questions"]}
    qs = [q for q in load_questions(Path(args.questions)) if q.id in ids]
    results = simulate_fix(_bot(args), qs, results, args.name)
    save(results, run_dir / "results.json")
    print("report:", _write_report(results, run_dir))


def cmd_bench(args):
    from ragblame.bench import bench
    from ragblame.questions import load_questions
    from ragblame.results import load, save
    run_dir = Path(args.run)
    results = load(run_dir / "results.json")
    results["bench"] = bench(_bot(args), load_questions(Path(args.questions)), results, per_fault=args.per_fault)
    save(results, run_dir / "results.json")
    b = results["bench"]
    print(f"benchmark over {b['n']} injected failures: {b['arms']}")
    print("report:", _write_report(results, run_dir))


def cmd_report(args):
    from ragblame.results import load
    run_dir = Path(args.run)
    print(_write_report(load(run_dir / "results.json"), run_dir))


def main(argv=None):
    p = argparse.ArgumentParser(prog="ragblame", description="Find which RAG stage caused each wrong answer.")
    p.add_argument("--version", action="version", version=f"ragblame {__version__}")
    p.add_argument("--docs", default=str(DOCS))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ask", help="ask the demo bot one question")
    s.add_argument("question")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("questions", help="have Claude write a test set with gold quotes")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default=str(QUESTIONS))
    s.set_defaults(fn=cmd_questions)

    s = sub.add_parser("run", help="answer, judge and diagnose every question")
    s.add_argument("--questions", default=str(QUESTIONS))
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--no-diagnose", action="store_true")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("calibrate", help="check the judge against a stricter model")
    s.add_argument("run")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--questions", default=str(QUESTIONS))
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("fix", help="simulate one fix across every question")
    s.add_argument("name")
    s.add_argument("run")
    s.add_argument("--questions", default=str(QUESTIONS))
    s.set_defaults(fn=cmd_fix)

    s = sub.add_parser("bench", help="break the bot on purpose and score how well each method finds the stage")
    s.add_argument("run")
    s.add_argument("--per-fault", type=int, default=6)
    s.add_argument("--questions", default=str(QUESTIONS))
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("report", help="write the HTML report for a run")
    s.add_argument("run")
    s.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
