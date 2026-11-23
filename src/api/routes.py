"""FastAPI app: browse prompts and suites, start runs, read reports."""

from __future__ import annotations

import difflib
import threading
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..agent import Components, default_components, render_markdown, run_suite
from ..tools import SpecError, Store, list_prompts, list_suites, load_prompt, load_suite
from ..utils.config import Settings, load_settings
from ..utils.logger import get_logger
from .versions import resolve_versions

log = get_logger("api")


class RunRequest(BaseModel):
    suite: str
    baseline: str = Field("previous", description="A version, or 'previous'")
    candidate: str = Field("latest", description="A version, or 'latest'")
    repeats: int | None = Field(None, ge=1, le=20)


def create_app(settings: Settings | None = None, components: Callable[[], Components] | None = None,
               run_inline: bool = False) -> FastAPI:
    settings = settings or load_settings()
    state: dict = {}
    if components:
        state["parts"] = components()
        store = state["parts"].store
    else:
        store = Store(settings.database_url)  # reading runs needs no API keys...
    store.init()

    def parts() -> Components:
        if "parts" not in state:  # ...models are only built when a run starts
            state["parts"] = default_components(settings, store)
        return state["parts"]

    app = FastAPI(title="Prompt Regression Tester", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "judge_model": settings.judge_model, "prompts": len(list_prompts(settings.prompts_dir)),
                "suites": len(list_suites(settings.suites_dir))}

    @app.get("/prompts")
    def prompts() -> dict:
        return list_prompts(settings.prompts_dir)

    @app.get("/prompts/{name}/{version}")
    def prompt(name: str, version: str) -> dict:
        try:
            spec = load_prompt(settings.prompts_dir, name, version)
        except SpecError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {**spec.model_dump(), "fingerprint": spec.fingerprint()}

    @app.get("/prompts/{name}/diff/{a}/{b}", response_class=PlainTextResponse)
    def prompt_diff(name: str, a: str, b: str) -> str:
        files = [settings.prompts_dir / name / f"{v}.yaml" for v in (a, b)]
        if not all(f.exists() for f in files):
            raise HTTPException(status_code=404, detail="Version not found")
        return "".join(difflib.unified_diff(files[0].read_text(encoding="utf-8").splitlines(True),
                                            files[1].read_text(encoding="utf-8").splitlines(True),
                                            fromfile=f"{name}/{a}", tofile=f"{name}/{b}"))

    @app.get("/suites")
    def suites() -> list[dict]:
        out = []
        for name in list_suites(settings.suites_dir):
            try:
                s = load_suite(settings.suites_dir / f"{name}.yaml")
                out.append({"name": name, "prompt": s.prompt, "cases": len(s.cases), "repeats": s.repeats,
                            "graders": s.graders.model_dump()})
            except SpecError as exc:
                out.append({"name": name, "error": str(exc)})
        return out

    @app.post("/runs", status_code=202)
    def start_run(request: RunRequest) -> dict:
        path = settings.suites_dir / f"{request.suite}.yaml"
        try:
            suite = load_suite(path)
            baseline, candidate = resolve_versions(settings.prompts_dir, suite.prompt, request.baseline, request.candidate)
        except SpecError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result: dict = {}

        def job():
            try:
                result["run"] = run_suite(parts(), path, settings.prompts_dir, baseline, candidate, request.repeats,
                                          {"source": "api"}, settings.max_concurrency)
            except Exception as exc:
                log.exception("run failed: %s", exc)

        if run_inline:
            job()
            return result.get("run") or {"status": "failed"}
        threading.Thread(target=job, daemon=True).start()
        return {"status": "started", "suite": request.suite, "baseline": baseline, "candidate": candidate}

    @app.get("/runs")
    def runs() -> list[dict]:
        return [{k: r[k] for k in ("id", "suite", "prompt", "baseline", "candidate", "status", "verdict", "created_at",
                                   "progress")} for r in store.list_runs()]

    @app.get("/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        run = store.run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {**run, "outputs": store.outputs_of(run_id), "pairwise": store.pairwise_of(run_id)}

    @app.get("/runs/{run_id}/report.md", response_class=PlainTextResponse)
    def report(run_id: str) -> str:
        run = store.run(run_id)
        if run is None or not run.get("summary"):
            raise HTTPException(status_code=404, detail="No finished run with that id")
        return render_markdown(run)

    return app
