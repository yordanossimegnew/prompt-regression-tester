"""Fixtures: a dedicated PostgreSQL test database and the example prompts/suite."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

from src.agent import Components, open_checkpointer
from src.tools import Store
from src.utils.config import PROJECT_ROOT, Settings
from tests.fakes import FakeGenerator, FakeJudges, HashEmbedder

load_dotenv(PROJECT_ROOT / ".env")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def pytest_collection_modifyitems(config, items):
    if not TEST_DATABASE_URL:
        skip = pytest.mark.skip(reason="TEST_DATABASE_URL is not set")
        for item in items:
            if "store" in item.fixturenames or "parts" in item.fixturenames:
                item.add_marker(skip)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(database_url=TEST_DATABASE_URL or "postgresql://x@localhost/x", reports_dir=tmp_path / "reports",
                    log_dir=tmp_path / "logs")


@pytest.fixture
def store(settings):
    s = Store(settings.database_url)
    s.init()
    with s.engine.begin() as c:
        c.execute(text("TRUNCATE runs, outputs, pairwise, output_cache CASCADE"))
    yield s
    s.engine.dispose()


@pytest.fixture
def parts(settings, store):
    saver = open_checkpointer(settings.database_url)
    yield Components(store, FakeGenerator(), FakeJudges(), HashEmbedder(), saver)
    saver.conn.close()
