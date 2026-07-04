"""LLM-judge answer scoring.

Term-overlap scoring (does an expected keyword appear verbatim in the answer)
is brittle: it misses paraphrases and transcription spellings (e.g. Whisper's
"Miguel Angelo" vs "Michelangelo"). This module scores an answer *semantically*:
a local LLM decides whether the candidate answer conveys the expected facts and
actually answers the question, returning a structured verdict.
"""

from __future__ import annotations

import json
import re
from urllib import error, request

from pydantic import BaseModel

DEFAULT_JUDGE_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class JudgeError(RuntimeError):
    """Raised when the judge model cannot be reached."""


class JudgeVerdict(BaseModel):
    correct: bool
    score: float
    reason: str


def build_judge_prompt(query: str, expected_terms: list[str], answer: str) -> str:
    facts = ", ".join(t for t in expected_terms if t.strip()) or "(none specified)"
    return (
        "You are grading whether an ANSWER correctly responds to a QUESTION about a video.\n"
        "The answer is CORRECT if it conveys the expected key facts and answers the question, "
        "even if worded differently, paraphrased, or containing minor spelling/transcription "
        "errors of names. It is INCORRECT if it omits the key facts, is off-topic, contradicts "
        "them, or says it cannot answer.\n"
        f"QUESTION: {query}\n"
        f"EXPECTED KEY FACTS the answer should convey: {facts}\n"
        f"ANSWER: {answer or '(empty)'}\n"
        'Respond with ONLY a compact JSON object of the form '
        '{"correct": true, "score": 0.0, "reason": "<one short sentence>"} '
        "where score is your confidence from 0.0 to 1.0."
    )


def _extract_verdict(text: str) -> JudgeVerdict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return JudgeVerdict(correct=False, score=0.0, reason="no JSON in judge output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return JudgeVerdict(correct=False, score=0.0, reason="unparseable judge JSON")
    correct = bool(data.get("correct", False))
    try:
        score = float(data.get("score", 1.0 if correct else 0.0))
    except (TypeError, ValueError):
        score = 1.0 if correct else 0.0
    score = max(0.0, min(1.0, score))
    reason = str(data.get("reason", ""))[:200]
    return JudgeVerdict(correct=correct, score=score, reason=reason)


class LlmJudge:
    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def judge(self, query: str, expected_terms: list[str], answer: str) -> JudgeVerdict:
        prompt = build_judge_prompt(query, expected_terms, answer)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(  # noqa: S310 - localhost model endpoint.
                http_request, timeout=self.timeout_seconds
            ) as response:
                decoded = response.read().decode("utf-8")
        except error.URLError as exc:
            raise JudgeError(f"Could not reach Ollama judge at {self.base_url}") from exc
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise JudgeError("Ollama judge returned invalid JSON envelope") from exc
        return _extract_verdict(str(parsed.get("response", "")))
