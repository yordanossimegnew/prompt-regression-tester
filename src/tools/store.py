"""PostgreSQL storage: runs, every generated output with its grades, pairwise verdicts, and a cache.

The cache is keyed by (prompt fingerprint, rendered input, repeat index), so re-running an
unchanged baseline costs nothing; only the new candidate version calls the model.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, Text,
                        create_engine, insert, select, update)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.engine import Engine

metadata = MetaData()

runs = Table(
    "runs", metadata,
    Column("id", String(32), primary_key=True),
    Column("suite", Text, nullable=False),
    Column("prompt", Text, nullable=False),
    Column("baseline", Text, nullable=False),
    Column("candidate", Text, nullable=False),
    Column("repeats", Integer, nullable=False),
    Column("status", String(20), nullable=False),     # running | completed | failed
    Column("verdict", String(20)),                    # pass | regression | improvement | inconclusive
    Column("summary", JSONB),
    Column("progress", JSONB, nullable=False, default=dict),
    Column("meta", JSONB, nullable=False, default=dict),   # e.g. git sha, triggered by
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
)
outputs = Table(
    "outputs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("side", String(10), nullable=False),       # baseline | candidate
    Column("version", Text, nullable=False),
    Column("case_id", Text, nullable=False),
    Column("repeat", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("error", Text),
    Column("latency_ms", Float, nullable=False),
    Column("input_tokens", Integer, nullable=False),
    Column("output_tokens", Integer, nullable=False),
    Column("cached", Boolean, nullable=False),
    Column("checks", JSONB, nullable=False, default=list),
    Column("judge", JSONB),                            # {"score", "passed", "reason"}
    Column("similarity", Float),
    Column("passed", Boolean),
)
pairwise = Table(
    "pairwise", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("case_id", Text, nullable=False),
    Column("repeat", Integer, nullable=False),
    Column("winner", String(10), nullable=False),     # baseline | candidate | tie
    Column("candidate_first", Boolean, nullable=False),
    Column("reason", Text, nullable=False),
)
cache = Table(
    "output_cache", metadata,
    Column("key", String(64), primary_key=True),
    Column("text", Text, nullable=False),
    Column("input_tokens", Integer, nullable=False),
    Column("output_tokens", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def cache_key(fingerprint: str, system: str, user: str, repeat: int) -> str:
    return hashlib.sha256(json.dumps([fingerprint, system, user, repeat]).encode()).hexdigest()


def _row(row) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row._mapping.items()}


class Store:
    def __init__(self, database_url: str):
        url = database_url
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                url = "postgresql+psycopg://" + url[len(prefix):]
        self.engine: Engine = create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10)

    def init(self) -> None:
        metadata.create_all(self.engine)

    # ------------------------------------------------------------------ runs
    def create_run(self, suite: str, prompt: str, baseline: str, candidate: str, repeats: int, meta: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        with self.engine.begin() as c:
            c.execute(insert(runs).values(id=rid, suite=suite, prompt=prompt, baseline=baseline, candidate=candidate,
                                          repeats=repeats, status="running", progress={"stage": "starting"},
                                          meta=meta, created_at=now()))
        return rid

    def update_run(self, rid: str, **values) -> None:
        if values.get("status") in ("completed", "failed"):
            values["finished_at"] = now()
        with self.engine.begin() as c:
            c.execute(update(runs).where(runs.c.id == rid).values(**values))

    def run(self, rid: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(runs).where(runs.c.id == rid)).first()
        return _row(row) if row else None

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(runs).order_by(runs.c.created_at.desc()).limit(limit)).all()
        return [_row(r) for r in rows]

    # ------------------------------------------------------------------ outputs
    def add_output(self, **values) -> int:
        with self.engine.begin() as c:
            return c.execute(insert(outputs).values(**values).returning(outputs.c.id)).scalar()

    def grade_output(self, output_id: int, checks: list, judge: dict | None, similarity: float | None, passed: bool) -> None:
        with self.engine.begin() as c:
            c.execute(update(outputs).where(outputs.c.id == output_id).values(
                checks=checks, judge=judge, similarity=similarity, passed=passed))

    def outputs_of(self, rid: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(outputs).where(outputs.c.run_id == rid)
                             .order_by(outputs.c.case_id, outputs.c.side, outputs.c.repeat)).all()
        return [_row(r) for r in rows]

    def add_pairwise(self, **values) -> None:
        with self.engine.begin() as c:
            c.execute(insert(pairwise).values(**values))

    def pairwise_of(self, rid: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(pairwise).where(pairwise.c.run_id == rid).order_by(pairwise.c.case_id, pairwise.c.repeat)).all()
        return [_row(r) for r in rows]

    # ------------------------------------------------------------------ cache
    def cached(self, key: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(cache).where(cache.c.key == key)).first()
        return _row(row) if row else None

    def put_cache(self, key: str, text: str, input_tokens: int, output_tokens: int) -> None:
        with self.engine.begin() as c:
            c.execute(pg_insert(cache).values(key=key, text=text, input_tokens=input_tokens,
                                              output_tokens=output_tokens, created_at=now())
                      .on_conflict_do_nothing(index_elements=["key"]))
