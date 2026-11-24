"""Specs, deterministic checks and statistics (no database)."""

from __future__ import annotations

import json

import pytest

from src.api import resolve_versions
from src.tools import (
    Check,
    PromptSpec,
    SpecError,
    bootstrap_diff,
    fisher_greater,
    fisher_less,
    load_prompt,
    run_check,
    sign_test_less,
)
from src.utils.config import PROJECT_ROOT

PROMPTS = PROJECT_ROOT / "prompts"


def test_prompt_rendering_and_fingerprint():
    spec = PromptSpec(name="p", version="v1", system="You help {{who}}.", template="Q: {{question}}")
    assert spec.variables == {"who", "question"}
    assert spec.render({"who": "devs", "question": "hi"}) == ("You help devs.", "Q: hi")
    with pytest.raises(SpecError, match="Missing variables"):
        spec.render({"who": "devs"})
    tweaked = spec.model_copy(update={"system": "You help {{who}}!"})
    assert spec.fingerprint() != tweaked.fingerprint()  # any change invalidates the cache
    assert spec.fingerprint() == spec.model_copy(update={"description": "docs only"}).fingerprint()
    with pytest.raises(ValueError, match="json_schema"):
        PromptSpec(name="p", version="v1", template="x", output="json")


def test_example_files_load_and_versions_resolve():
    assert load_prompt(PROMPTS, "support-triage", "v2").output == "json"
    assert resolve_versions(PROMPTS, "support-triage", "previous", "latest") == ("v1", "v2")
    with pytest.raises(SpecError):
        resolve_versions(PROMPTS, "support-triage", "v1", "v1")
    with pytest.raises(SpecError):
        load_prompt(PROMPTS, "support-triage", "v9")


@pytest.mark.parametrize("check, output, expected", [
    (Check(type="contains", value="Refund"), "we will refund you", True),
    (Check(type="contains", value="Refund", case_sensitive=True), "we will refund you", False),
    (Check(type="not_contains", value="password"), "Your password is...", False),
    (Check(type="regex", value=r"\d{3}-\d{4}"), "call 555-1234", True),
    (Check(type="not_regex", value="refunded"), "we have refunded it", False),
    (Check(type="max_chars", value=10), "short", True),
    (Check(type="json_valid"), "{not json", False),
    (Check(type="json_field_equals", field="a.b", value="x"), json.dumps({"a": {"b": "X"}}), True),
    (Check(type="json_field_in", field="p", value=["low", "high"]), json.dumps({"p": "urgent"}), False),
    (Check(type="json_field_equals", field="missing", value=1), "{}", False),
])
def test_deterministic_checks(check, output, expected):
    assert run_check(check, output)[0] is expected


def test_json_schema_check_uses_prompt_schema():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    assert run_check(Check(type="json_schema"), '{"n": 3}', schema)[0]
    ok, detail = run_check(Check(type="json_schema"), '{"n": "3"}', schema)
    assert not ok and "schema" in detail


def test_statistics_against_known_values():
    assert fisher_less(3, 3, 0, 3) == pytest.approx(0.05)          # classic 2x2 table
    assert fisher_less(5, 10, 5, 10) == pytest.approx(0.6719, abs=1e-3)
    assert fisher_less(0, 3, 3, 3) == 1.0
    assert fisher_greater(0, 3, 3, 3) == pytest.approx(0.05)
    assert sign_test_less(0, 6) == pytest.approx(1 / 64)
    assert sign_test_less(3, 3) > 0.5
    ci = bootstrap_diff([-1.0] * 8)
    assert ci["mean"] == ci["low"] == ci["high"] == -1.0
    noisy = bootstrap_diff([1, -1, 0.5, -0.5, 0, 0.2])
    assert noisy["low"] < 0 < noisy["high"]  # no clear direction
