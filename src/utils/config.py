"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://postgres:postgres@localhost:5432/prompt_regression"
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    suites_dir: Path = PROJECT_ROOT / "suites"
    reports_dir: Path = PROJECT_ROOT / "reports"
    judge_model: str = "claude-opus-5-5"
    judge_effort: str = "medium"
    voyage_model: str = "voyage-3.5"
    max_concurrency: int = 8
    # USD per million tokens (Claude Opus 5.5). Used for the cost line in reports.
    input_price: float = 4.0
    output_price: float = 20.0
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"

    def __post_init__(self) -> None:
        if not self.database_url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL.")


def load_settings(**overrides) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings(
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        judge_model=os.getenv("JUDGE_MODEL", Settings.judge_model),
        voyage_model=os.getenv("VOYAGE_MODEL", Settings.voyage_model),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", Settings.max_concurrency)),
        log_level=os.getenv("LOG_LEVEL", Settings.log_level),
    )
    return replace(settings, **{k: v for k, v in overrides.items() if v is not None})
