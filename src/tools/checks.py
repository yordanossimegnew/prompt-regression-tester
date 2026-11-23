"""Deterministic graders: fast, free, and exact."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator

from .specs import Check


def _field(data: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(data, dict) and part in data:
            data = data[part]
        elif isinstance(data, list) and part.isdigit() and int(part) < len(data):
            data = data[int(part)]
        else:
            raise KeyError(path)
    return data


def _norm(s: str, case_sensitive: bool) -> str:
    return s if case_sensitive else s.lower()


def run_check(check: Check, output: str, json_schema: dict | None = None) -> tuple[bool, str]:
    """Return (passed, explanation)."""
    t, v = check.type, check.value
    try:
        if t == "contains":
            ok = _norm(str(v), check.case_sensitive) in _norm(output, check.case_sensitive)
            return ok, f"contains {v!r}" if ok else f"missing {v!r}"
        if t == "not_contains":
            ok = _norm(str(v), check.case_sensitive) not in _norm(output, check.case_sensitive)
            return ok, f"does not contain {v!r}" if ok else f"must not contain {v!r}"
        if t in ("regex", "not_regex"):
            found = re.search(str(v), output, 0 if check.case_sensitive else re.IGNORECASE) is not None
            ok = found if t == "regex" else not found
            return ok, f"{'matches' if found else 'no match for'} /{v}/"
        if t == "exact":
            ok = output.strip() == str(v).strip()
            return ok, "exact match" if ok else "differs from expected"
        if t == "max_chars":
            return len(output) <= int(v), f"{len(output)} chars (max {v})"
        if t == "min_chars":
            return len(output) >= int(v), f"{len(output)} chars (min {v})"

        data = json.loads(output)  # every remaining check needs valid JSON
        if t == "json_valid":
            return True, "valid JSON"
        if t == "json_schema":
            schema = v or json_schema
            errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: e.path)
            return (not errors), "matches schema" if not errors else f"schema: {errors[0].message}"
        if t == "json_field_equals":
            got = _field(data, check.field)
            ok = _norm(str(got), check.case_sensitive) == _norm(str(v), check.case_sensitive)
            return ok, f"{check.field} = {got!r}" + ("" if ok else f" (expected {v!r})")
        if t == "json_field_in":
            got = _field(data, check.field)
            allowed = [_norm(str(x), check.case_sensitive) for x in v]
            ok = _norm(str(got), check.case_sensitive) in allowed
            return ok, f"{check.field} = {got!r}" + ("" if ok else f" (expected one of {v})")
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc.msg}"
    except KeyError as exc:
        return False, f"field {exc.args[0]!r} not found"
    except re.error as exc:
        return False, f"bad regex: {exc}"
    return False, f"unknown check {t}"


def run_checks(checks: list[Check], output: str, json_schema: dict | None = None) -> list[dict]:
    return [{"check": c.type + (f":{c.field}" if c.field else ""), "passed": ok, "detail": detail}
            for c in checks for ok, detail in [run_check(c, output, json_schema)]]
