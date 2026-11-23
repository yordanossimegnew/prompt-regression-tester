"""A regression run as a LangGraph graph with four parallel fan-outs.

    START -> calibrate -> generate x (2 versions x cases x repeats) -> grade x outputs
          -> pairwise x (cases x repeats) -> report -> END

* calibrate - grade the suite's known-good/known-bad outputs first; if the judge disagrees
              with them too often, its scores are reported but not used for the verdict
* generate  - run each prompt version on each case N times (cached by prompt fingerprint)
* grade     - deterministic checks, rubric judge, and embedding similarity per output
* pairwise  - judge baseline vs candidate side by side, swapping positions on alternate
              repeats to cancel position bias
* report    - statistics decide pass / regression / improvement
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..models import Generator, Judges
from ..tools import PromptSpec, Store, SuiteSpec, cache_key, run_checks
from ..utils.logger import get_logger
from .report import build_summary

log = get_logger("runner")


@dataclass
class Components:
    store: Store
    generator: Generator
    judges: Judges | None
    embedder: Any | None = None
    checkpointer: Any = None


class RunState(TypedDict, total=False):
    run_id: str
    suite: dict
    baseline: dict
    candidate: dict
    calibration: dict
    generated: Annotated[list[dict], operator.add]
    graded: Annotated[list[dict], operator.add]
    pairs: Annotated[list[dict], operator.add]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_run_graph(parts: Components):
    store = parts.store

    def specs(state) -> tuple[SuiteSpec, PromptSpec, PromptSpec]:
        return SuiteSpec(**state["suite"]), PromptSpec(**state["baseline"]), PromptSpec(**state["candidate"])

    def judge_enabled(suite: SuiteSpec) -> bool:
        return suite.graders.judge and parts.judges is not None

    # ------------------------------------------------------------------ calibrate
    def calibrate(state: RunState) -> dict:
        suite, baseline, _ = specs(state)
        store.update_run(state["run_id"], progress={"stage": "calibrating judge"})
        if not judge_enabled(suite) or not suite.calibration:
            return {"calibration": {"items": 0, "agreement": None, "trusted": judge_enabled(suite)}}
        cases = {c.id: c for c in suite.cases}
        hits, details = 0, []
        for item in suite.calibration:
            case = cases[item.case]
            _, user = baseline.render(case.vars)
            verdict = parts.judges.judge(case.rubric or "", user, item.output)
            predicted = verdict.score >= suite.graders.judge_min_score
            hits += predicted == item.expected_pass
            details.append({"case": item.case, "expected_pass": item.expected_pass, "score": verdict.score,
                            "agrees": predicted == item.expected_pass, "reason": verdict.reason})
        agreement = hits / len(suite.calibration)
        return {"calibration": {"items": len(suite.calibration), "agreement": round(agreement, 3),
                                "trusted": agreement >= suite.thresholds.min_judge_agreement, "details": details}}

    # ------------------------------------------------------------------ generate
    def plan_generation(state: RunState):
        suite, _, _ = specs(state)
        store.update_run(state["run_id"], progress={"stage": "generating"})
        return [Send("generate", {"run_id": state["run_id"], "side": side, "spec": state[side], "case": case.model_dump(),
                                  "repeat": r})
                for side in ("baseline", "candidate") for case in suite.cases for r in range(suite.repeats)]

    def generate(task: dict) -> dict:
        spec = PromptSpec(**task["spec"])
        system, user = spec.render(task["case"]["vars"])
        key = cache_key(spec.fingerprint(), system, user, task["repeat"])
        hit = store.cached(key)
        error = None
        if hit:
            text, tin, tout, latency, cached = hit["text"], hit["input_tokens"], hit["output_tokens"], 0.0, True
        else:
            try:
                gen = parts.generator.generate(spec, system, user)
                text, tin, tout, latency, cached = gen.text, gen.input_tokens, gen.output_tokens, gen.latency_ms, False
                store.put_cache(key, text, tin, tout)
            except anthropic.AuthenticationError:
                raise  # a config problem, not a prompt regression: fail the run
            except Exception as exc:  # record it; the output simply fails its checks
                log.warning("generation failed: %s", exc)
                text, tin, tout, latency, cached, error = "", 0, 0, 0.0, False, f"{type(exc).__name__}: {exc}"[:500]
        oid = store.add_output(run_id=task["run_id"], side=task["side"], version=spec.version,
                               case_id=task["case"]["id"], repeat=task["repeat"], text=text, error=error,
                               latency_ms=round(latency, 1), input_tokens=tin, output_tokens=tout, cached=cached)
        return {"generated": [{"output_id": oid, "side": task["side"], "case_id": task["case"]["id"],
                               "repeat": task["repeat"], "text": text, "user": user}]}

    # ------------------------------------------------------------------ grade
    def plan_grading(state: RunState):
        store.update_run(state["run_id"], progress={"stage": "grading"})
        return [Send("grade", {"suite": state["suite"], "spec": state[g["side"]], "calibration": state["calibration"],
                               "item": g}) for g in state["generated"]]

    def grade(task: dict) -> dict:
        suite = SuiteSpec(**task["suite"])
        spec = PromptSpec(**task["spec"])
        item = task["item"]
        case = next(c for c in suite.cases if c.id == item["case_id"])
        checks = run_checks(case.checks, item["text"], spec.json_schema)
        judge, similarity = None, None
        if judge_enabled(suite) and case.rubric and item["text"]:
            v = parts.judges.judge(case.rubric, item["user"], item["text"])
            judge = {"score": v.score, "passed": v.score >= suite.graders.judge_min_score, "reason": v.reason}
        if suite.graders.similarity and case.reference and parts.embedder is not None and item["text"]:
            ref, out = parts.embedder.embed_documents([case.reference, item["text"]])
            similarity = round(_cosine(ref, out), 4)
        use_judge = judge is not None and task["calibration"].get("trusted", True)
        passed = (bool(item["text"]) and all(c["passed"] for c in checks)
                  and (not use_judge or judge["passed"])
                  and (similarity is None or similarity >= suite.graders.similarity_min))
        store.grade_output(item["output_id"], checks, judge, similarity, passed)
        return {"graded": [{**item, "passed": passed}]}

    # ------------------------------------------------------------------ pairwise
    def plan_pairwise(state: RunState):
        suite, _, _ = specs(state)
        if not (suite.graders.pairwise and parts.judges is not None):
            return ["report"]
        store.update_run(state["run_id"], progress={"stage": "pairwise"})
        by = {(g["side"], g["case_id"], g["repeat"]): g for g in state["generated"]}
        tasks = []
        for case in suite.cases:
            if not case.rubric:
                continue
            for r in range(suite.repeats):
                b, c = by.get(("baseline", case.id, r)), by.get(("candidate", case.id, r))
                if b and c and b["text"] and c["text"]:
                    tasks.append(Send("pairwise", {"run_id": state["run_id"], "rubric": case.rubric, "case_id": case.id,
                                                   "repeat": r, "user": b["user"], "baseline": b["text"],
                                                   "candidate": c["text"], "candidate_first": r % 2 == 1}))
        return tasks or ["report"]

    def pairwise(task: dict) -> dict:
        first, second = ("candidate", "baseline") if task["candidate_first"] else ("baseline", "candidate")
        v = parts.judges.compare(task["rubric"], task["user"], task[first], task[second])
        winner = "tie" if v.winner == "tie" else (first if v.winner == "A" else second)
        store.add_pairwise(run_id=task["run_id"], case_id=task["case_id"], repeat=task["repeat"], winner=winner,
                           candidate_first=task["candidate_first"], reason=v.reason)
        return {"pairs": [{"case_id": task["case_id"], "winner": winner}]}

    # ------------------------------------------------------------------ report
    def report(state: RunState) -> dict:
        suite, baseline, candidate = specs(state)
        summary = build_summary(suite, store.outputs_of(state["run_id"]), store.pairwise_of(state["run_id"]),
                                state.get("calibration") or {})
        store.update_run(state["run_id"], status="completed", verdict=summary["verdict"], summary=summary,
                         progress={"stage": "done"})
        return {}

    g = StateGraph(RunState)
    for name, fn in [("calibrate", calibrate), ("generate", generate), ("grade", grade), ("pairwise", pairwise),
                     ("report", report)]:
        g.add_node(name, fn)
    g.add_node("collect_outputs", lambda s: {})
    g.add_node("collect_grades", lambda s: {})
    g.add_edge(START, "calibrate")
    g.add_conditional_edges("calibrate", plan_generation, ["generate"])
    g.add_edge("generate", "collect_outputs")
    g.add_conditional_edges("collect_outputs", plan_grading, ["grade"])
    g.add_edge("grade", "collect_grades")
    g.add_conditional_edges("collect_grades", plan_pairwise, ["pairwise", "report"])
    g.add_edge("pairwise", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=parts.checkpointer)
