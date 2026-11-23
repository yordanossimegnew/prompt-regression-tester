"""Resolve 'latest' / 'previous' to concrete prompt versions (handy in CI)."""

from __future__ import annotations

from pathlib import Path

from ..tools import SpecError, list_prompts


def resolve_versions(prompts_dir: Path, prompt: str, baseline: str, candidate: str) -> tuple[str, str]:
    versions = list_prompts(prompts_dir).get(prompt, [])
    if not versions:
        raise SpecError(f"No versions found for prompt '{prompt}'")

    def pick(v: str) -> str:
        if v == "latest":
            return versions[-1]
        if v == "previous":
            if len(versions) < 2:
                raise SpecError(f"'{prompt}' has only one version; nothing to compare against")
            return versions[-2]
        if v not in versions:
            raise SpecError(f"Unknown version '{v}' for '{prompt}' (have {versions})")
        return v

    b, c = pick(baseline), pick(candidate)
    if b == c:
        raise SpecError("Baseline and candidate are the same version")
    return b, c
