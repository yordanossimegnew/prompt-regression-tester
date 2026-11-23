"""Full runs through the LangGraph pipeline on PostgreSQL, plus the API and CLI."""

from __future__ import annotations

import yaml
from fastapi.testclient import TestClient

from src.agent import Components, render_markdown, run_suite
from src.api import create_app
from src.utils.config import PROJECT_ROOT
from tests.fakes import FakeJudges

PROMPTS = PROJECT_ROOT / "prompts"
SUITE = PROJECT_ROOT / "suites" / "support-triage.yaml"


def test_v2_cleanup_is_caught_as_a_regression(parts):
    run = run_suite(parts, SUITE, PROMPTS, "v1", "v2")
    s = run["summary"]
    assert run["status"] == "completed" and run["verdict"] == "regression"
    assert s["pass_rate"]["baseline"] == 1.0 and s["pass_rate"]["candidate"] < 0.5
    assert s["pass_rate"]["p_worse"] < 0.05
    status = {c["id"]: c["status"] for c in s["cases"]}
    assert {status[c] for c in ("outage-pro", "outage-free", "double-charge", "refund-request")} == {"regressed"}
    assert status["feature-dark-mode"] == status["spanish-bug"] == "same"
    assert any("priority" in f for f in next(c for c in s["cases"] if c["id"] == "outage-pro")["candidate_failures"])
    pw = s["graders"]["pairwise"]
    assert pw["baseline_wins"] == 16 and pw["candidate_wins"] == 0 and pw["p_worse"] < 0.001
    assert s["calibration"] == {**s["calibration"], "items": 4, "agreement": 1.0, "trusted": True}
    assert s["graders"]["judge"]["mean_score_diff"]["high"] < 0
    assert len(parts.store.outputs_of(run["id"])) == 2 * 7 * 4

    md = render_markdown(run)
    assert "REGRESSION" in md and "`outage-pro`" in md and "regressed" in md


def test_unchanged_baseline_comes_from_cache(parts):
    run_suite(parts, SUITE, PROMPTS, "v1", "v2")
    calls = parts.generator.calls
    again = run_suite(parts, SUITE, PROMPTS, "v1", "v2")
    assert parts.generator.calls == calls  # nothing regenerated
    assert again["summary"]["generation"]["cached"] == 56 and again["verdict"] == "regression"


def test_reversed_comparison_is_an_improvement(parts):
    assert run_suite(parts, SUITE, PROMPTS, "v2", "v1")["verdict"] == "improvement"


def test_noise_alone_is_not_a_regression(parts, tmp_path):
    suite = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    suite["cases"] = [c for c in suite["cases"] if c["id"] in ("feature-dark-mode", "spanish-bug", "password-reset")]
    suite["calibration"], suite["repeats"] = [], 5
    path = tmp_path / "noisy.yaml"
    path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    parts.generator.flaky = {"feature-dark-mode", "spanish-bug"}  # both versions randomly misclassify
    run = run_suite(parts, path, PROMPTS, "v1", "v2")
    assert run["verdict"] in ("pass", "inconclusive")  # differences are noise, not a regression


def test_untrusted_judge_is_excluded_from_the_verdict(parts, tmp_path):
    inverted = Components(parts.store, parts.generator, FakeJudges(invert=True), None, parts.checkpointer)
    suite = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    suite["cases"] = [c for c in suite["cases"] if c["id"] in ("feature-dark-mode", "double-charge", "outage-pro")]
    suite["graders"]["pairwise"] = False
    path = tmp_path / "judge.yaml"
    path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    s = run_suite(inverted, path, PROMPTS, "v1", "v2")["summary"]
    assert s["calibration"]["agreement"] == 0.0 and s["calibration"]["trusted"] is False
    assert not s["graders"]["judge"]["trusted"]
    # checks alone still catch it
    assert s["verdict"] in ("regression", "inconclusive")


def test_api_and_cli(parts, settings, monkeypatch, tmp_path, capsys):
    client = TestClient(create_app(settings, components=lambda: parts, run_inline=True))
    assert client.get("/prompts").json() == {"support-triage": ["v1", "v2"]}
    assert client.get("/suites").json()[0]["cases"] == 7
    diff = client.get("/prompts/support-triage/diff/v1/v2").text
    assert "-  Rules for the reply:" in diff and "+version: v2" in diff
    run = client.post("/runs", json={"suite": "support-triage"}).json()  # previous vs latest
    assert run["verdict"] == "regression" and (run["baseline"], run["candidate"]) == ("v1", "v2")
    assert "REGRESSION" in client.get(f"/runs/{run['id']}/report.md").text
    assert len(client.get(f"/runs/{run['id']}").json()["outputs"]) == 56
    assert client.post("/runs", json={"suite": "support-triage", "baseline": "v1", "candidate": "v1"}).status_code == 422

    import main
    import src.agent
    monkeypatch.setattr(src.agent, "default_components", lambda *_a, **_k: parts)
    monkeypatch.setattr(main, "settings", settings)
    report = tmp_path / "pr-comment.md"
    args = type("A", (), {"suite": "support-triage", "baseline": "v1", "candidate": "v2", "repeats": None,
                          "report": str(report), "fail_on_inconclusive": False})()
    assert main.cmd_run(args) == 1  # CI fails on regression
    assert "REGRESSION" in report.read_text(encoding="utf-8")
    args.baseline, args.candidate = "v2", "v1"
    assert main.cmd_run(args) == 0
