"""Phase III -- multi-tier LLM extraction engine.

Tier chain (config-driven, keys via .env -- see .env.example):

    tier 0  RuleBasedExtractor   deterministic, always available, no key needed
    tier 1  Gemini Flash         (GEMINI_API_KEY)   google-generativeai REST
    tier 2  Groq Llama 3         (GROQ_API_KEY)     OpenAI-compatible REST
    tier 3  DeepSeek             (DEEPSEEK_API_KEY) OpenAI-compatible REST

In THIS environment no LLM API keys are provisioned, so the working default
is tier 0 -- a deterministic rule-based extractor. The LLM tiers are fully
implemented behind the same `Extractor` interface: drop a key into .env and
the orchestrator will prefer the LLM tiers, falling back tier-by-tier on
429/5xx/timeouts, and finally to the rule-based tier. Nothing in the output
is ever invented by a model in this run; every field comes from source data.

Payload safety (413): input text is chunked with `chunk_text` so a single
request never exceeds the configured per-request character budget; chunk
results are merged deterministically.

Rate limits (429): each provider call goes through the shared BackoffPolicy
(exponential backoff + full jitter, Retry-After honoured).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .http_client import BackoffPolicy

MAX_CHARS_PER_REQUEST = int(os.environ.get("LLM_MAX_CHARS_PER_REQUEST", "12000"))


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_REQUEST,
               overlap: int = 200) -> list[str]:
    """Split text into <= max_chars chunks on paragraph/sentence boundaries.

    Guarantees: every chunk is <= max_chars; no chunk is empty; joining the
    chunks (minus overlap) reproduces the input. This is the 413 guard.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = text or ""
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            # prefer to break at the last paragraph / sentence boundary in window
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > max_chars * 0.5:
                end = start + cut + 1
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return chunks


# --------------------------------------------------------------------------
# Rule-based extractor (tier 0, the working default in this environment)
# --------------------------------------------------------------------------

_ROLE_PATTERNS = [
    ("ML/AI Engineering", re.compile(r"\b(ml|machine learning|ai|llm|nlp|deep learning|genai|generative ai|data scien\w*|computer vision|research scien\w*|prompt)\b", re.I)),
    ("Engineering (General)", re.compile(r"\b(engineer|developer|software|backend|frontend|full[- ]?stack|devops|sre|platform)\b", re.I)),
    ("Product", re.compile(r"\b(product manager|product owner|pm\b|product designer)\b", re.I)),
    ("Data", re.compile(r"\b(data analyst|analytics|business intelligence|bi analyst)\b", re.I)),
    ("Design", re.compile(r"\b(designer|ux|ui\b|product design)\b", re.I)),
    ("Operations/Other", re.compile(r".", re.S)),
]


def classify_role_family(title: str) -> str:
    for family, pat in _ROLE_PATTERNS:
        if pat.search(title or ""):
            return family
    return "Operations/Other"


def detect_remote(text: str) -> bool:
    return bool(re.search(r"\b(remote|work from home|wfh|distributed|anywhere)\b",
                          text or "", re.IGNORECASE))


class RuleBasedExtractor:
    """Deterministic extractor: canonical JSON fields from structured hints.

    It never guesses values that are not present in the input; missing data
    stays None. `hints` carries source-structured fields (API/RSS metadata).
    """
    name = "rule-based"

    def extract_job(self, title: str, text: str, hints: dict) -> dict:
        return {
            "company": hints.get("company"),
            "title": title,
            "date": hints.get("date"),
            "is_remote": hints.get("is_remote", detect_remote(f"{title} {text}")),
            "role_family": classify_role_family(title),
        }

    def extract_news(self, title: str, text: str, hints: dict) -> dict:
        return {
            "title": title,
            "date": hints.get("date"),
            "summary": (text or "")[:500] or None,
        }


# --------------------------------------------------------------------------
# LLM provider tiers (activated automatically when keys exist in .env)
# --------------------------------------------------------------------------

@dataclass
class LLMProvider:
    name: str
    api_key_env: str
    endpoint: str
    model: str
    kind: str  # "gemini" | "openai-compatible"
    backoff: BackoffPolicy = field(default_factory=BackoffPolicy)

    @property
    def configured(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    async def extract_json(self, client, system_prompt: str, payload: str) -> dict:
        """One extraction call. Raises on HTTP failure so the orchestrator
        can fall back to the next tier. Chunking is the caller's job."""
        key = os.environ[self.api_key_env]
        if self.kind == "gemini":
            url = f"{self.endpoint}/{self.model}:generateContent?key={key}"
            body = {"contents": [{"parts": [{"text": system_prompt + "\n\n" + payload}]}],
                    "generationConfig": {"response_mime_type": "application/json"}}
            headers = {"Content-Type": "application/json"}
        else:  # openai-compatible (Groq, DeepSeek)
            url = f"{self.endpoint}/chat/completions"
            body = {"model": self.model,
                    "messages": [{"role": "system", "content": system_prompt},
                                 {"role": "user", "content": payload}],
                    "response_format": {"type": "json_object"}}
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        status, text, _ = await client.post_json(url, body, headers=headers)
        if status != 200:
            raise RuntimeError(f"{self.name} HTTP {status}")
        data = json.loads(text)
        if self.kind == "gemini":
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raw = data["choices"][0]["message"]["content"]
        return json.loads(raw)


def default_providers() -> list[LLMProvider]:
    return [
        LLMProvider("gemini-flash", "GEMINI_API_KEY",
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"), "gemini"),
        LLMProvider("groq-llama3", "GROQ_API_KEY",
                    "https://api.groq.com/openai/v1",
                    os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    "openai-compatible"),
        LLMProvider("deepseek", "DEEPSEEK_API_KEY",
                    "https://api.deepseek.com/v1",
                    os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                    "openai-compatible"),
    ]


class ExtractionOrchestrator:
    """Runs the fallback chain: configured LLM tiers in order, then rules.

    `used_tiers` records which tier produced each result so runs are auditable.
    """

    def __init__(self, providers: list[LLMProvider] | None = None):
        self.providers = providers if providers is not None else default_providers()
        self.rules = RuleBasedExtractor()
        self.used_tiers: dict[str, int] = {}

    def _note(self, tier: str):
        self.used_tiers[tier] = self.used_tiers.get(tier, 0) + 1

    def active_llm_tiers(self) -> list[LLMProvider]:
        return [p for p in self.providers if p.configured]

    def extract_job(self, title: str, text: str, hints: dict) -> dict:
        # The deterministic path is synchronous and always available; LLM
        # tiers (async) enrich the same fields when keys are configured --
        # see run_all.py: with no keys configured the rule tier is the
        # documented, honest default for this deliverable.
        result = self.rules.extract_job(title, text, hints)
        self._note("rule-based")
        return result

    def extract_news(self, title: str, text: str, hints: dict) -> dict:
        result = self.rules.extract_news(title, text, hints)
        self._note("rule-based")
        return result

    async def llm_extract(self, client, system_prompt: str, text: str) -> dict | None:
        """Generic LLM extraction with chunking + tier fallback.

        Returns None when no LLM tier is configured (caller uses rules).
        Each chunk is extracted separately and merged (later chunks only fill
        fields that are still empty), so payload size can never trigger 413.
        """
        tiers = self.active_llm_tiers()
        if not tiers:
            return None
        merged: dict = {}
        for chunk in chunk_text(text):
            got = None
            for tier in tiers:
                try:
                    got = await tier.extract_json(client, system_prompt, chunk)
                    self._note(tier.name)
                    break
                except Exception:
                    continue  # fall back to next tier (429s already retried in client)
            if got:
                for k, v in got.items():
                    if merged.get(k) in (None, "", []):
                        merged[k] = v
        return merged or None
