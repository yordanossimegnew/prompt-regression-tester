"""Run a suite: load specs from git-tracked YAML, execute the graph, return the finished run."""

from __future__ import annotations

from pathlib import Path

from ..tools import load_prompt, load_suite, validate_suite
from .agent import Components, build_run_graph


def run_suite(parts: Components, suite_path: Path, prompts_dir: Path, baseline: str, candidate: str,
              repeats: int | None = None, meta: dict | None = None, max_concurrency: int = 8) -> dict:
    suite = load_suite(suite_path)
    if repeats:
        suite = suite.model_copy(update={"repeats": repeats})
    base = load_prompt(prompts_dir, suite.prompt, baseline)
    cand = load_prompt(prompts_dir, suite.prompt, candidate)
    validate_suite(suite, base)
    validate_suite(suite, cand)

    run_id = parts.store.create_run(suite.name, suite.prompt, baseline, candidate, suite.repeats, meta or {})
    try:
        build_run_graph(parts).invoke(
            {"run_id": run_id, "suite": suite.model_dump(), "baseline": base.model_dump(), "candidate": cand.model_dump()},
            {"configurable": {"thread_id": f"run-{run_id}"}, "max_concurrency": max_concurrency})
    except Exception as exc:
        parts.store.update_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
        raise
    return parts.store.run(run_id)
