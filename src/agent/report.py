"""Turn graded outputs into a verdict with statistics, and render it for humans and CI."""

from __future__ import annotations

from collections import defaultdict

from ..tools import SuiteSpec, bootstrap_diff, fisher_greater, fisher_less, sign_test_less

ICON = {"pass": "✅", "regression": "❌", "improvement": "🚀", "inconclusive": "⚠️"}


def _rate(k: int, n: int) -> float:
    return round(k / n, 4) if n else 0.0


def build_summary(suite: SuiteSpec, outputs: list[dict], pairs: list[dict], calibration: dict) -> dict:
    t = suite.thresholds
    by_case: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"baseline": [], "candidate": []})
    for o in outputs:
        by_case[o["case_id"]][o["side"]].append(o)

    # ---- overall pass rate (all graders combined), one-sided Fisher exact tests
    def tally(side):
        rows = [o for o in outputs if o["side"] == side]
        return sum(bool(o["passed"]) for o in rows), len(rows)

    bp, bn = tally("baseline")
    cp, cn = tally("candidate")
    pass_rate = {"baseline": _rate(bp, bn), "candidate": _rate(cp, cn),
                 "delta": round(_rate(cp, cn) - _rate(bp, bn), 4),
                 "p_worse": round(fisher_less(bp, bn, cp, cn), 4), "p_better": round(fisher_greater(bp, bn, cp, cn), 4)}

    # ---- per-grader views
    def grader_rate(side, key):
        rows = [o for o in outputs if o["side"] == side]
        if key == "checks":
            vals = [all(c["passed"] for c in o["checks"]) and bool(o["text"]) for o in rows]
        else:
            vals = [o["judge"]["passed"] for o in rows if o["judge"]]
        return _rate(sum(vals), len(vals)) if vals else None

    judge_means = {}
    for case_id, sides in by_case.items():
        scores = {s: [o["judge"]["score"] for o in sides[s] if o["judge"]] for s in ("baseline", "candidate")}
        if scores["baseline"] and scores["candidate"]:
            judge_means[case_id] = (sum(scores["baseline"]) / len(scores["baseline"]),
                                    sum(scores["candidate"]) / len(scores["candidate"]))
    judge_diff = bootstrap_diff([c - b for b, c in judge_means.values()])
    sims = {s: [o["similarity"] for o in outputs if o["side"] == s and o["similarity"] is not None]
            for s in ("baseline", "candidate")}
    wins = sum(p["winner"] == "candidate" for p in pairs)
    losses = sum(p["winner"] == "baseline" for p in pairs)
    ties = sum(p["winner"] == "tie" for p in pairs)
    graders = {
        "checks": {"baseline": grader_rate("baseline", "checks"), "candidate": grader_rate("candidate", "checks")},
        "judge": {"baseline": grader_rate("baseline", "judge"), "candidate": grader_rate("candidate", "judge"),
                  "mean_score_diff": judge_diff, "trusted": calibration.get("trusted", True)},
        "pairwise": {"candidate_wins": wins, "baseline_wins": losses, "ties": ties,
                     "p_worse": round(sign_test_less(wins, losses), 4), "p_better": round(sign_test_less(losses, wins), 4)},
        "similarity": {s: (round(sum(v) / len(v), 4) if v else None) for s, v in sims.items()},
    }

    # ---- per case
    cases = []
    for case in suite.cases:
        sides = by_case.get(case.id, {"baseline": [], "candidate": []})
        b, c = sides["baseline"], sides["candidate"]
        bk, ck = sum(bool(o["passed"]) for o in b), sum(bool(o["passed"]) for o in c)
        p_worse, p_better = fisher_less(bk, len(b), ck, len(c)), fisher_greater(bk, len(b), ck, len(c))
        if ck < bk:
            status = "regressed" if p_worse < t.alpha else "worse"
        elif ck > bk:
            status = "fixed" if p_better < t.alpha else "better"
        else:
            status = "same"
        failing = [d for o in c for d in o["checks"] if not d["passed"]]
        cases.append({"id": case.id, "tags": case.tags, "baseline": f"{bk}/{len(b)}", "candidate": f"{ck}/{len(c)}",
                      "p_worse": round(p_worse, 4), "status": status,
                      "candidate_failures": sorted({f"{d['check']}: {d['detail']}" for d in failing})[:5],
                      "judge": {"baseline": [o["judge"]["score"] for o in b if o["judge"]],
                                "candidate": [o["judge"]["score"] for o in c if o["judge"]]}})

    # ---- verdict
    reasons_bad, reasons_good = [], []
    if pass_rate["p_worse"] < t.alpha and -pass_rate["delta"] >= t.min_drop:
        reasons_bad.append(f"pass rate fell {pass_rate['baseline']:.0%} → {pass_rate['candidate']:.0%} "
                           f"(Fisher p={pass_rate['p_worse']:.3f})")
    if pass_rate["p_better"] < t.alpha and pass_rate["delta"] >= t.min_drop:
        reasons_good.append(f"pass rate rose {pass_rate['baseline']:.0%} → {pass_rate['candidate']:.0%} "
                            f"(Fisher p={pass_rate['p_better']:.3f})")
    if graders["judge"]["trusted"] and judge_diff["n"] and judge_diff["high"] < 0:
        reasons_bad.append(f"judge score dropped {judge_diff['mean']:+.2f} (95% CI {judge_diff['low']:+.2f}…{judge_diff['high']:+.2f})")
    if graders["judge"]["trusted"] and judge_diff["n"] and judge_diff["low"] > 0:
        reasons_good.append(f"judge score rose {judge_diff['mean']:+.2f} (95% CI {judge_diff['low']:+.2f}…{judge_diff['high']:+.2f})")
    pw = graders["pairwise"]
    if pw["p_worse"] < t.alpha and pw["baseline_wins"] > pw["candidate_wins"]:
        reasons_bad.append(f"baseline preferred in pairwise review {pw['baseline_wins']}–{pw['candidate_wins']} "
                           f"(sign test p={pw['p_worse']:.3f})")
    if pw["p_better"] < t.alpha and pw["candidate_wins"] > pw["baseline_wins"]:
        reasons_good.append(f"candidate preferred in pairwise review {pw['candidate_wins']}–{pw['baseline_wins']} "
                            f"(sign test p={pw['p_better']:.3f})")

    if reasons_bad:
        verdict = "regression"
    elif reasons_good:
        verdict = "improvement"
    elif any(c["status"] in ("worse", "regressed") for c in cases):
        verdict = "inconclusive"   # some cases dropped, but not enough evidence overall
    else:
        verdict = "pass"

    gen = [o for o in outputs if not o["cached"]]
    return {
        "verdict": verdict, "reasons": reasons_bad or reasons_good, "pass_rate": pass_rate, "graders": graders,
        "cases": cases, "calibration": calibration, "repeats": suite.repeats, "alpha": t.alpha,
        "generation": {"outputs": len(outputs), "cached": len(outputs) - len(gen),
                       "input_tokens": sum(o["input_tokens"] for o in gen),
                       "output_tokens": sum(o["output_tokens"] for o in gen),
                       "errors": sum(bool(o["error"]) for o in outputs)},
    }


def render_markdown(run: dict) -> str:
    s = run["summary"]
    pr, g = s["pass_rate"], s["graders"]
    lines = [f"## {ICON[s['verdict']]} Prompt regression: **{s['verdict'].upper()}**",
             f"`{run['prompt']}` · baseline **{run['baseline']}** vs candidate **{run['candidate']}** · "
             f"suite `{run['suite']}` · {s['repeats']} repeats per case", ""]
    lines += [f"- {r}" for r in s["reasons"]] + ([""] if s["reasons"] else [])
    lines += ["| Metric | Baseline | Candidate |", "|---|---|---|",
              f"| Pass rate (all graders) | {pr['baseline']:.0%} | {pr['candidate']:.0%} |"]
    if g["checks"]["baseline"] is not None:
        lines.append(f"| Deterministic checks | {g['checks']['baseline']:.0%} | {g['checks']['candidate']:.0%} |")
    if g["judge"]["baseline"] is not None:
        trust = "" if g["judge"]["trusted"] else " (judge untrusted)"
        lines.append(f"| Judge pass rate{trust} | {g['judge']['baseline']:.0%} | {g['judge']['candidate']:.0%} |")
    pw = g["pairwise"]
    if pw["candidate_wins"] + pw["baseline_wins"] + pw["ties"]:
        lines.append(f"| Pairwise wins (ties {pw['ties']}) | {pw['baseline_wins']} | {pw['candidate_wins']} |")
    if g["similarity"]["baseline"] is not None:
        lines.append(f"| Similarity to reference | {g['similarity']['baseline']:.2f} | {g['similarity']['candidate']:.2f} |")
    changed = [c for c in s["cases"] if c["status"] != "same"]
    if changed:
        lines += ["", "| Case | Baseline | Candidate | Status | Why |", "|---|---|---|---|---|"]
        for c in changed:
            why = "; ".join(c["candidate_failures"][:2]) or "–"
            lines.append(f"| `{c['id']}` | {c['baseline']} | {c['candidate']} | {c['status']} | {why} |")
    cal = s["calibration"]
    if cal.get("items"):
        lines += ["", f"Judge calibration: {cal['agreement']:.0%} agreement on {cal['items']} known examples "
                      f"({'trusted' if cal['trusted'] else 'NOT trusted: judge scores excluded from the verdict'})."]
    gen = s["generation"]
    lines += ["", f"<sub>{gen['outputs']} outputs ({gen['cached']} from cache), {gen['input_tokens']:,} input / "
                  f"{gen['output_tokens']:,} output tokens generated · run `{run['id']}`</sub>"]
    return "\n".join(lines)
