"""Prompt and suite specs: YAML files in git, validated with Pydantic.

    prompts/<name>/<version>.yaml     one file per prompt version
    suites/<name>.yaml                test cases + grading settings for one prompt
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

_VAR = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class SpecError(ValueError):
    pass


class PromptSpec(BaseModel):
    name: str
    version: str
    description: str = ""
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(2000, ge=1, le=128000)
    system: str = ""
    template: str = Field(..., description="User message; {{variable}} placeholders are filled from each case.")
    output: Literal["text", "json"] = "text"
    json_schema: dict | None = Field(None, description="For output: json - enforced with structured outputs.")

    @model_validator(mode="after")
    def _json(self) -> "PromptSpec":
        if self.output == "json" and not self.json_schema:
            raise ValueError("output: json requires json_schema")
        return self

    @property
    def variables(self) -> set[str]:
        return set(_VAR.findall(self.system)) | set(_VAR.findall(self.template))

    def fingerprint(self) -> str:
        """Hash of everything that affects the output (used as the cache key)."""
        payload = self.model_dump(exclude={"description"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def render(self, variables: dict[str, Any]) -> tuple[str, str]:
        missing = sorted(v for v in self.variables if v not in variables)
        if missing:
            raise SpecError(f"Missing variables {missing} for prompt {self.name}@{self.version}")

        def fill(text: str) -> str:
            return _VAR.sub(lambda m: str(variables[m.group(1)]), text)

        return fill(self.system), fill(self.template)


class Check(BaseModel):
    """A deterministic check on the output."""

    type: Literal["contains", "not_contains", "regex", "not_regex", "exact", "max_chars", "min_chars",
                  "json_valid", "json_schema", "json_field_equals", "json_field_in"]
    value: Any = None
    field: str | None = None        # for json_field_* (dot path, e.g. "ticket.priority")
    case_sensitive: bool = False


class Case(BaseModel):
    id: str
    vars: dict[str, Any]
    checks: list[Check] = []
    rubric: str | None = Field(None, description="What a good answer looks like; graded by the LLM judge.")
    reference: str | None = Field(None, description="A reference answer for semantic similarity.")
    tags: list[str] = []


class CalibrationItem(BaseModel):
    """A known-good or known-bad output used to check the judge before trusting it."""

    case: str
    output: str
    expected_pass: bool


class Graders(BaseModel):
    judge: bool = True
    pairwise: bool = True
    similarity: bool = False
    similarity_min: float = Field(0.80, ge=0, le=1)
    judge_min_score: int = Field(4, ge=1, le=5, description="Judge score (1-5) needed to pass.")


class Thresholds(BaseModel):
    alpha: float = Field(0.05, gt=0, lt=1, description="Significance level for regression tests.")
    min_drop: float = Field(0.05, ge=0, le=1, description="Ignore pass-rate drops smaller than this.")
    min_judge_agreement: float = Field(0.8, ge=0, le=1, description="Judge must match calibration this often.")


class SuiteSpec(BaseModel):
    name: str
    prompt: str
    description: str = ""
    repeats: int = Field(4, ge=1, le=20, description="Runs per case; 4+ lets a single case reach p < 0.05.")
    graders: Graders = Graders()
    thresholds: Thresholds = Thresholds()
    cases: list[Case]
    calibration: list[CalibrationItem] = []

    @model_validator(mode="after")
    def _ids(self) -> "SuiteSpec":
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        unknown = {c.case for c in self.calibration} - set(ids)
        if unknown:
            raise ValueError(f"calibration refers to unknown cases {sorted(unknown)}")
        return self


def _load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"{path}: expected a mapping at the top level")
    return data


def load_prompt(prompts_dir: Path, name: str, version: str) -> PromptSpec:
    path = Path(prompts_dir) / name / f"{version}.yaml"
    if not path.exists():
        raise SpecError(f"Prompt {name}@{version} not found ({path})")
    try:
        spec = PromptSpec(**_load_yaml(path))
    except ValidationError as exc:
        raise SpecError(f"{path}: {exc}") from exc
    if spec.name != name or spec.version != version:
        raise SpecError(f"{path}: name/version in the file must be {name}/{version}")
    return spec


def list_prompts(prompts_dir: Path) -> dict[str, list[str]]:
    out = {}
    for folder in sorted(p for p in Path(prompts_dir).iterdir() if p.is_dir()):
        versions = sorted((f.stem for f in folder.glob("*.yaml")), key=_version_key)
        if versions:
            out[folder.name] = versions
    return out


def _version_key(v: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", v)]


def load_suite(path: Path) -> SuiteSpec:
    try:
        return SuiteSpec(**_load_yaml(Path(path)))
    except ValidationError as exc:
        raise SpecError(f"{path}: {exc}") from exc


def list_suites(suites_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(suites_dir).glob("*.yaml"))


def validate_suite(suite: SuiteSpec, prompt: PromptSpec) -> None:
    for case in suite.cases:
        missing = sorted(prompt.variables - set(case.vars))
        if missing:
            raise SpecError(f"case '{case.id}' is missing variables {missing} needed by {prompt.name}@{prompt.version}")
