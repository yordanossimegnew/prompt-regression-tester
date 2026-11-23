"""Models via LangChain: the prompt under test, the LLM judges, and Voyage embeddings."""

from __future__ import annotations

import json
import time
from typing import Literal, Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import BaseModel, Field

from ..prompts import JUDGE_PROMPT, PAIRWISE_PROMPT
from ..tools.specs import PromptSpec
from ..utils.config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Generation(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class Verdict(BaseModel):
    score: int = Field(description="1 (bad) to 5 (fully meets the rubric).")
    reason: str = Field(description="One sentence naming the deciding issue.")


class PairwiseVerdict(BaseModel):
    winner: Literal["A", "B", "tie"]
    reason: str = Field(description="One sentence.")


class Generator(Protocol):
    def generate(self, spec: PromptSpec, system: str, user: str) -> Generation: ...


class Judges(Protocol):
    def judge(self, rubric: str, user_input: str, output: str) -> Verdict: ...
    def compare(self, rubric: str, user_input: str, output_a: str, output_b: str) -> PairwiseVerdict: ...


def _usage(msg) -> tuple[int, int]:
    meta = getattr(msg, "usage_metadata", None) or {}
    return meta.get("input_tokens", 0), meta.get("output_tokens", 0)


def _text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")


class AnthropicGenerator:
    """Runs a prompt version exactly as its YAML specifies (model, effort, max_tokens, output format)."""

    def generate(self, spec: PromptSpec, system: str, user: str) -> Generation:
        chat = ChatAnthropic(model=spec.model, max_tokens=spec.max_tokens, output_config={"effort": spec.effort},
                             betas=[FALLBACK_BETA], model_kwargs={"fallbacks": "default"}, max_retries=3)
        messages = ([SystemMessage(system)] if system else []) + [HumanMessage(user)]
        start = time.perf_counter()
        if spec.output == "json":
            out = chat.with_structured_output(spec.json_schema, method="json_schema", include_raw=True).invoke(messages)
            raw, parsed = out["raw"], out["parsed"]
            text = json.dumps(parsed, ensure_ascii=False) if parsed is not None else _text(raw)
        else:
            raw = chat.invoke(messages)
            text = _text(raw)
        tokens_in, tokens_out = _usage(raw)
        return Generation(text=text, input_tokens=tokens_in, output_tokens=tokens_out,
                          latency_ms=(time.perf_counter() - start) * 1000)


class AnthropicJudges:
    def __init__(self, settings: Settings):
        self.chat = ChatAnthropic(model=settings.judge_model, max_tokens=2000,
                                  output_config={"effort": settings.judge_effort}, betas=[FALLBACK_BETA],
                                  model_kwargs={"fallbacks": "default"}, max_retries=3)

    def judge(self, rubric: str, user_input: str, output: str) -> Verdict:
        model = self.chat.with_structured_output(Verdict, method="json_schema")
        return model.invoke([SystemMessage(JUDGE_PROMPT), HumanMessage(
            f"<rubric>\n{rubric}\n</rubric>\n<input>\n{user_input}\n</input>\n<output>\n{output}\n</output>")])

    def compare(self, rubric: str, user_input: str, output_a: str, output_b: str) -> PairwiseVerdict:
        model = self.chat.with_structured_output(PairwiseVerdict, method="json_schema")
        return model.invoke([SystemMessage(PAIRWISE_PROMPT), HumanMessage(
            f"<rubric>\n{rubric}\n</rubric>\n<input>\n{user_input}\n</input>\n"
            f"<output_a>\n{output_a}\n</output_a>\n<output_b>\n{output_b}\n</output_b>")])


def make_embedder(settings: Settings):
    return VoyageAIEmbeddings(model=settings.voyage_model)
