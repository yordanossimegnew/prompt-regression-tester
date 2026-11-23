"""Prompt Regression Tester: FastAPI app + a CLI for local runs and CI.

    uvicorn main:app --reload                                     # API (docs at /docs)
    python main.py run --suite support-triage                     # previous vs latest version
    python main.py run --suite support-triage --baseline v1 --candidate v2 --repeats 5
    python main.py validate                                       # lint every prompt and suite

Exit codes for `run`: 0 = pass/improvement, 1 = regression (or inconclusive with
--fail-on-inconclusive), 2 = error (bad spec, API failure).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from src.api import create_app, resolve_versions
from src.utils import load_settings, setup_logging

settings = load_settings()
setup_logging(settings.log_level, settings.log_dir)
app = create_app(settings)

GREEN, RED, YELLOW, BOLD, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[2m", "\033[0m"


def cmd_validate() -> int:
    from src.tools import SpecError, list_prompts, list_suites, load_prompt, load_suite, validate_suite

    problems = 0
    for name, versions in list_prompts(settings.prompts_dir).items():
        for v in versions:
            try:
                load_prompt(settings.prompts_dir, name, v)
                print(f"{GREEN}✓{RESET} prompts/{name}/{v}.yaml")
            except SpecError as exc:
                problems += 1
                print(f"{RED}✗ {exc}{RESET}")
    for s in list_suites(settings.suites_dir):
        try:
            suite = load_suite(settings.suites_dir / f"{s}.yaml")
            for v in list_prompts(settings.prompts_dir).get(suite.prompt, []):
                validate_suite(suite, load_prompt(settings.prompts_dir, suite.prompt, v))
            print(f"{GREEN}✓{RESET} suites/{s}.yaml ({len(suite.cases)} cases)")
        except SpecError as exc:
            problems += 1
            print(f"{RED}✗ {exc}{RESET}")
    return 2 if problems else 0


def cmd_run(args) -> int:
    from src.agent import default_components, render_markdown, run_suite
    from src.tools import SpecError, load_suite

    path = settings.suites_dir / f"{args.suite}.yaml"
    try:
        suite = load_suite(path)
        baseline, candidate = resolve_versions(settings.prompts_dir, suite.prompt, args.baseline, args.candidate)
    except SpecError as exc:
        print(f"{RED}{exc}{RESET}")
        return 2
    print(f"{DIM}{suite.prompt}: {baseline} (baseline) vs {candidate} (candidate) · "
          f"{len(suite.cases)} cases × {args.repeats or suite.repeats} repeats{RESET}")
    try:
        run = run_suite(default_components(settings), path, settings.prompts_dir, baseline, candidate, args.repeats,
                        {"source": "cli", "git_sha": os.getenv("GITHUB_SHA")}, settings.max_concurrency)
    except Exception as exc:
        print(f"{RED}Run failed: {type(exc).__name__}: {exc}{RESET}")
        return 2

    markdown = render_markdown(run)
    settings.reports_dir.mkdir(exist_ok=True)
    md_path = settings.reports_dir / f"{run['id']}.md"
    md_path.write_text(markdown, encoding="utf-8")
    (settings.reports_dir / f"{run['id']}.json").write_text(json.dumps(run["summary"], indent=2), encoding="utf-8")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(markdown)

    verdict = run["verdict"]
    color = {"pass": GREEN, "improvement": GREEN, "inconclusive": YELLOW, "regression": RED}[verdict]
    print(f"\n{color}{BOLD}{verdict.upper()}{RESET}")
    for reason in run["summary"]["reasons"]:
        print(f"  - {reason}")
    for case in run["summary"]["cases"]:
        if case["status"] != "same":
            print(f"  {case['id']:<22} {case['baseline']} → {case['candidate']}  {case['status']}")
    print(f"{DIM}report: {md_path}{RESET}")
    if verdict == "regression" or (verdict == "inconclusive" and args.fail_on_inconclusive):
        return 1
    return 0


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; we print ✓ ✗ →
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Prompt Regression Tester")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Compare two prompt versions on a suite.")
    run.add_argument("--suite", required=True)
    run.add_argument("--baseline", default="previous")
    run.add_argument("--candidate", default="latest")
    run.add_argument("--repeats", type=int)
    run.add_argument("--report", help="Also write the Markdown report here (e.g. for a PR comment).")
    run.add_argument("--fail-on-inconclusive", action="store_true")
    sub.add_parser("validate", help="Validate every prompt and suite file.")
    serve = sub.add_parser("serve", help="Run the API.")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "run":
        sys.exit(cmd_run(args))
    if args.command == "validate":
        sys.exit(cmd_validate())
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=getattr(args, "port", 8000))


if __name__ == "__main__":
    main()
