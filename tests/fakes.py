"""Scripted stand-ins for the prompt under test, the judges and the embedder (no API keys)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading

from src.models import Generation, PairwiseVerdict, Verdict

GOOD = {
    "outage-pro": ("bug", "urgent", "Thanks Dana, we're treating this as urgent. Engineering is investigating the 502 errors now and will update you shortly."),
    "outage-free": ("bug", "urgent", "Hi Sam, we're sorry. Our engineers are investigating the missing data as a priority and will follow up soon."),
    "double-charge": ("billing", "high", "Thanks Priya, sorry about the duplicate charge. Our billing team will review it and reply within one business day."),
    "refund-request": ("billing", "medium", "Hi Tom, I've passed your cancellation to billing. They'll review the refund request and confirm next steps."),
    "feature-dark-mode": ("feature_request", "low", "Thanks Lena! Dark mode is a popular idea and I've shared it with our product team."),
    "password-reset": ("account", "high", "Hi Ahmed, for security we can't change passwords by email. Please verify your identity at acme.example/verify."),
    "spanish-bug": ("bug", "medium", "Hola Lucía, gracias por avisarnos. Nuestro equipo está revisando el error 413."),
}


class FakeGenerator:
    """v1 behaves; v2 (the 'cleanup' edit) loses the urgency rule and starts promising refunds."""

    def __init__(self, flaky: set[str] | None = None):
        self.calls = 0
        self.flaky = flaky or set()
        self._lock = threading.Lock()

    def generate(self, spec, system, user):
        with self._lock:
            self.calls += 1
        case = next(cid for cid, (_, _, reply) in GOOD.items() if _case_marker(cid) in user)
        category, priority, reply = GOOD[case]
        if spec.version == "v2":
            if case.startswith("outage"):
                priority = "high"
            if case in ("double-charge", "refund-request"):
                reply = "Sorry about that! We have refunded the charge; you'll see it in 3-5 days."
        if case in self.flaky and "uptime" not in system:  # identical noise for both versions
            seed = int(hashlib.md5(f"{case}{self.calls}".encode()).hexdigest(), 16)
            if seed % 3 == 0:
                category = "other"
        return Generation(text=json.dumps({"category": category, "priority": priority,
                                           "summary": f"Ticket about {case}.", "reply": reply}),
                          input_tokens=120, output_tokens=60, latency_ms=5.0)


def _case_marker(case_id: str) -> str:
    return {"outage-pro": "Site down", "outage-free": "database gone", "double-charge": "Charged twice",
            "refund-request": "Cancel and refund", "feature-dark-mode": "Dark mode", "password-reset": "Can't log in",
            "spanish-bug": "Error al subir"}[case_id]


class FakeJudges:
    """Grades with the same rules the rubrics describe (urgency, no refund promises)."""

    def __init__(self, invert: bool = False):
        self.invert = invert

    def _score(self, rubric: str, output: str) -> int:
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return 1
        score = 5
        if "urgent" in rubric.lower() and data.get("priority") != "urgent":
            score = 2
        if re.search(r"(have|has been) refunded|we('| wi)ll refund", data.get("reply", ""), re.I):
            score = 1
        return 6 - score if self.invert else score

    def judge(self, rubric, user_input, output):
        s = self._score(rubric, output)
        return Verdict(score=s, reason="meets the rubric" if s >= 4 else "breaks a rubric rule")

    def compare(self, rubric, user_input, output_a, output_b):
        a, b = self._score(rubric, output_a), self._score(rubric, output_b)
        return PairwiseVerdict(winner="tie" if a == b else ("A" if a > b else "B"), reason="scored by rubric")


class HashEmbedder:
    def embed_documents(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in re.findall(r"[a-z]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out
