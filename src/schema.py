"""Canonical record schema (schemaVersion 1.0) for the Intelligence Graph.

Every record produced by the pipeline has the shape:

    {
      "schemaVersion": "1.0",
      "recordType": "STARTUP" | "PRODUCT" | "RESEARCH_PAPER" | "JOB" | "NEWS",
      "source": {"name": <str>, "url": <valid source URL>},
      "collectedAt": <ISO-8601 UTC timestamp>,
      "content": { ... type-specific fields ... }
    }

No record is emitted without a legitimate source URL -- this is the
anti-hallucination contract of the whole pipeline.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

SCHEMA_VERSION = "1.0"
RECORD_TYPES = {"STARTUP", "PRODUCT", "RESEARCH_PAPER", "JOB", "NEWS"}
PRICING_MODELS = {"FREE", "FREEMIUM", "PAID", "ENTERPRISE"}

_URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def content_hash(text: str) -> str:
    """Stable SHA-256 of normalised text -- used for dedup / first-seen tracking."""
    return hashlib.sha256(" ".join((text or "").split()).lower().encode("utf-8")).hexdigest()


def make_record(record_type: str, source_name: str, source_url: str,
                content: dict, collected_at: str | None = None) -> dict:
    if record_type not in RECORD_TYPES:
        raise ValueError(f"unknown recordType: {record_type}")
    if not source_url or not _URL_RE.match(source_url):
        raise ValueError(f"record without a valid source URL refused: {source_url!r}")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "recordType": record_type,
        "source": {"name": source_name, "url": source_url},
        "collectedAt": collected_at or now_iso(),
        "content": content,
    }


def validate_record(rec: dict) -> list[str]:
    """Return a list of schema violations (empty list = valid)."""
    problems: list[str] = []
    if rec.get("schemaVersion") != SCHEMA_VERSION:
        problems.append("schemaVersion != 1.0")
    rt = rec.get("recordType")
    if rt not in RECORD_TYPES:
        problems.append(f"bad recordType {rt!r}")
        return problems
    src = rec.get("source") or {}
    if not src.get("name"):
        problems.append("source.name missing")
    if not src.get("url") or not _URL_RE.match(str(src["url"])):
        problems.append("source.url missing/invalid")
    c = rec.get("content") or {}
    if rt == "STARTUP":
        if not c.get("entityName"):
            problems.append("STARTUP.content.entityName missing")
        ec = c.get("employeeCount")
        if ec is not None and not isinstance(ec, int):
            problems.append("STARTUP.content.employeeCount must be int or null")
    elif rt == "PRODUCT":
        if not c.get("startupName"):
            problems.append("PRODUCT.content.startupName missing")
        if c.get("pricingModel") not in PRICING_MODELS:
            problems.append("PRODUCT.content.pricingModel invalid")
    elif rt == "RESEARCH_PAPER":
        if not c.get("title"):
            problems.append("RESEARCH_PAPER.content.title missing")
        if not isinstance(c.get("authors"), list):
            problems.append("RESEARCH_PAPER.content.authors must be a list")
        if not c.get("paper_url"):
            problems.append("RESEARCH_PAPER.content.paper_url missing")
        gs = c.get("github_stars")
        if gs is not None and not isinstance(gs, int):
            problems.append("RESEARCH_PAPER.content.github_stars must be int or null")
    elif rt == "JOB":
        if not c.get("company"):
            problems.append("JOB.content.company missing")
        if not isinstance(c.get("is_remote"), bool):
            problems.append("JOB.content.is_remote must be bool")
        if not c.get("role_family"):
            problems.append("JOB.content.role_family missing")
    elif rt == "NEWS":
        if not c.get("title"):
            problems.append("NEWS.content.title missing")
    return problems
