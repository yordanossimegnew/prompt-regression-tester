"""Build the real components (Claude generator + judges, Voyage embeddings, Postgres)."""

from __future__ import annotations

from ..tools import Store
from ..utils.config import Settings
from .agent import Components
from .memory import open_checkpointer


def default_components(settings: Settings, store: Store | None = None) -> Components:
    from ..models import AnthropicGenerator, AnthropicJudges, make_embedder

    store = store or Store(settings.database_url)
    store.init()
    try:
        embedder = make_embedder(settings)
    except Exception:
        embedder = None  # similarity grading needs VOYAGE_API_KEY; other graders still work
    return Components(store, AnthropicGenerator(), AnthropicJudges(settings), embedder,
                      open_checkpointer(settings.database_url))
