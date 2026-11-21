"""Phase IV -- deterministic entity resolution.

Canonicalises raw company/product names to a canonical entity using, in
strict priority order (all deterministic, no LLM involved):

  1. exact match against the seed alias table (confidence 1.0)
  2. normalised match: lowercase, strip legal suffixes (Inc, Ltd, LLC, GmbH,
     PBC, Corp, Technologies...), strip punctuation/spaces (confidence 0.95)
  3. token-set fuzzy match against seed canonicals/aliases using Jaccard
     similarity on character 3-grams, threshold 0.72 (confidence = score)
  4. self-canonicalisation: a cleaned version of the raw name becomes its own
     canonical form (method "normalise", confidence 0.6)

Every resolution is appended to an Entity Mapping Log entry:
{raw_name, canonical_name, method, confidence}.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_LEGAL_SUFFIXES = {
    "inc", "inc.", "ltd", "ltd.", "llc", "llp", "corp", "corp.", "corporation",
    "gmbh", "pbc", "bv", "b.v.", "sarl", "sa", "ag", "co", "co.", "company",
    "technologies", "technology", "labs", "systems", "industries", "group",
    "holdings", "the",
}
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")


def _basic_clean(name: str) -> str:
    """Human-readable cleanup: trim, drop trailing legal suffix after comma."""
    s = (name or "").strip()
    s = re.sub(r",\s*(inc\.?|ltd\.?|llc|corp\.?|gmbh|pbc)\s*$", "", s, flags=re.IGNORECASE)
    return _SPACE_RE.sub(" ", s).strip()


def _norm_key(name: str) -> str:
    """Aggressive normalisation key used for matching."""
    s = _PUNCT_RE.sub(" ", (name or "").lower())
    tokens = [t for t in s.split() if t not in _LEGAL_SUFFIXES]
    return "".join(tokens)


def _ngrams(s: str, n: int = 3) -> set[str]:
    s = f"  {s}  "
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


class EntityResolver:
    def __init__(self, seed_path: str | Path | None = None):
        if seed_path is None:
            seed_path = Path(__file__).with_name("seed_startups.json")
        seeds = json.loads(Path(seed_path).read_text(encoding="utf-8"))
        self._exact: dict[str, str] = {}
        self._norm: dict[str, str] = {}
        self._ngram_index: list[tuple[str, set[str]]] = []  # (canonical, ngrams of one name variant)
        for entry in seeds:
            canonical = entry["canonical"]
            names = [canonical, *entry.get("aliases", [])]
            for n in names:
                self._exact[n.strip().lower()] = canonical
                key = _norm_key(n)
                self._norm[key] = canonical
                self._ngram_index.append((canonical, _ngrams(key)))
        self.mapping_log: list[dict] = []
        self._cache: dict[str, dict] = {}

    def resolve(self, raw_name: str) -> dict:
        raw = (raw_name or "").strip()
        if not raw:
            return {"raw_name": raw, "canonical_name": raw, "method": "empty", "confidence": 0.0}
        if raw in self._cache:
            return self._cache[raw]
        result: dict | None = None
        # 1. exact
        hit = self._exact.get(raw.lower())
        if hit:
            result = {"raw_name": raw, "canonical_name": hit, "method": "exact", "confidence": 1.0}
        # 2. normalised
        if result is None:
            key = _norm_key(raw)
            hit = self._norm.get(key)
            if hit:
                result = {"raw_name": raw, "canonical_name": hit,
                          "method": "normalised", "confidence": 0.95}
        # 3. fuzzy n-gram Jaccard vs seed
        if result is None:
            key = _norm_key(raw)
            grams = _ngrams(key)
            best, best_score = None, 0.0
            for canonical, seed_grams in self._ngram_index:
                union = grams | seed_grams
                if not union:
                    continue
                score = len(grams & seed_grams) / len(union)
                if score > best_score:
                    best, best_score = canonical, score
            if best is not None and best_score >= 0.72:
                result = {"raw_name": raw, "canonical_name": best,
                          "method": "fuzzy_ngram", "confidence": round(best_score, 3)}
        # 4. self-canonicalise
        if result is None:
            result = {"raw_name": raw, "canonical_name": _basic_clean(raw),
                      "method": "self_normalise", "confidence": 0.6}
        self._cache[raw] = result
        self.mapping_log.append(result)
        return result

    def canonical(self, raw_name: str) -> str:
        return self.resolve(raw_name)["canonical_name"]
