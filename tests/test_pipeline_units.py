"""Unit tests: date normalisation, entity resolution, chunking, schema."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dates import normalize_date, parse_relative, within_last_hours  # noqa: E402
from src.entity_resolution import EntityResolver  # noqa: E402
from src.llm_orchestrator import chunk_text  # noqa: E402
from src.schema import make_record, validate_record  # noqa: E402

NOW = datetime(2026, 11, 21, 4, 0, 0, tzinfo=timezone.utc)


# ---------------- dates ----------------

def test_iso_with_z():
    dt = normalize_date("2026-11-20T10:30:00Z")
    assert dt == datetime(2026, 11, 20, 10, 30, tzinfo=timezone.utc)


def test_iso_with_offset_converted_to_utc():
    dt = normalize_date("2026-11-20T16:00:00+05:30")
    assert dt == datetime(2026, 11, 20, 10, 30, tzinfo=timezone.utc)


def test_rfc822_rss_date():
    dt = normalize_date("Fri, 20 Nov 2026 10:30:00 GMT")
    assert dt == datetime(2026, 11, 20, 10, 30, tzinfo=timezone.utc)


def test_epoch_seconds_and_millis():
    assert normalize_date(1795163400) is not None
    assert normalize_date(1795163400000) == normalize_date(1795163400)


def test_relative_hours_ago():
    dt = parse_relative("2 hours ago", now=NOW)
    assert dt == datetime(2026, 11, 21, 2, 0, tzinfo=timezone.utc)


def test_relative_days_and_indefinite_article():
    assert parse_relative("3 days ago", now=NOW) == datetime(2026, 11, 18, 4, 0, tzinfo=timezone.utc)
    assert parse_relative("an hour ago", now=NOW) == datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc)


def test_relative_embedded_in_text():
    assert parse_relative("Posted 5 minutes ago by editor", now=NOW) == \
        datetime(2026, 11, 21, 3, 55, tzinfo=timezone.utc)


def test_just_now_and_yesterday():
    assert parse_relative("just now", now=NOW) == NOW
    assert parse_relative("yesterday", now=NOW) == datetime(2026, 11, 20, 4, 0, tzinfo=timezone.utc)


def test_garbage_returns_none():
    assert normalize_date("sometime last spring") is None
    assert normalize_date("") is None
    assert normalize_date(None) is None


def test_within_last_hours_window():
    assert within_last_hours(normalize_date("2026-11-20T10:30:00Z"), 24, now=NOW)
    assert not within_last_hours(normalize_date("2026-11-19T03:59:00Z"), 24, now=NOW)
    assert not within_last_hours(None, 24, now=NOW)


# ---------------- entity resolution ----------------

def _resolver():
    return EntityResolver()


def test_exact_alias():
    r = _resolver().resolve("OpenAI, Inc.")
    assert r["canonical_name"] == "OpenAI" and r["method"] == "exact" and r["confidence"] == 1.0


def test_space_variant_normalised():
    r = _resolver().resolve("Open AI")
    assert r["canonical_name"] == "OpenAI" and r["confidence"] >= 0.9


def test_legal_suffix_stripped():
    assert _resolver().canonical("Cohere Inc.") == "Cohere"
    assert _resolver().canonical("Groq, Inc.") == "Groq"


def test_known_alias_variants():
    res = _resolver()
    assert res.canonical("W&B") in ("Weights & Biases",)
    assert res.canonical("wandb") == "Weights & Biases"
    assert res.canonical("Langchain") == "LangChain"
    assert res.canonical("X.AI") == "xAI"
    assert res.canonical("trychroma") == "Chroma"


def test_unknown_self_canonicalises():
    r = _resolver().resolve("Frobnicator Labs, Inc.")
    assert r["canonical_name"] == "Frobnicator Labs"  # legal suffix dropped
    assert r["method"] in ("self_normalise", "normalised", "fuzzy_ngram")


def test_mapping_log_records_entries():
    res = _resolver()
    res.resolve("OpenAI, Inc.")
    res.resolve("Brand New Startup XYZ")
    raws = {e["raw_name"] for e in res.mapping_log}
    assert "OpenAI, Inc." in raws and "Brand New Startup XYZ" in raws


# ---------------- chunking (413 guard) ----------------

def test_short_text_single_chunk():
    assert chunk_text("hello world", max_chars=100) == ["hello world"]


def test_empty_text_no_chunks():
    assert chunk_text("", max_chars=100) == []


def test_long_text_chunks_within_budget():
    text = ("Paragraph one is here. " * 40 + "\n\n") * 30
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert all(c for c in chunks)


def test_chunks_cover_content():
    text = " ".join(f"word{i}" for i in range(5000))
    chunks = chunk_text(text, max_chars=2000, overlap=50)
    assert all(len(c) <= 2000 for c in chunks)
    assert chunks[0].startswith("word0")
    assert "word4999" in chunks[-1]


def test_invalid_budget_rejected():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0)


# ---------------- rule-based extraction ----------------

def test_role_family_classification():
    from src.llm_orchestrator import classify_role_family, detect_remote
    assert classify_role_family("Senior Data Scientist, Marketing") == "ML/AI Engineering"
    assert classify_role_family("Machine Learning Engineer") == "ML/AI Engineering"
    assert classify_role_family("LLM Platform Engineer") == "ML/AI Engineering"
    assert classify_role_family("Backend Developer (Python)") == "Engineering (General)"
    assert classify_role_family("Product Manager, Growth") == "Product"
    assert detect_remote("Remote-first role, work from anywhere")
    assert not detect_remote("On-site in Berlin office")


# ---------------- schema ----------------

def test_valid_startup_record():
    rec = make_record("STARTUP", "YC", "https://example.com",
                      {"entityName": "OpenAI", "employeeCount": 3000})
    assert validate_record(rec) == []


def test_record_without_url_refused():
    import pytest
    with pytest.raises(ValueError):
        make_record("STARTUP", "YC", "", {"entityName": "X", "employeeCount": None})


def test_product_pricing_enum_enforced():
    rec = make_record("PRODUCT", "HF", "https://example.com/p",
                      {"productName": "p", "startupName": "X", "pricingModel": "CHEAP"})
    assert any("pricingModel" in p for p in validate_record(rec))
