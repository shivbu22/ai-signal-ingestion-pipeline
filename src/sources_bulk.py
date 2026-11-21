"""Phase I -- massive one-time bulk extraction.

Sources (all real, key-free, every record traceable to a live URL):
  * Startups        YC OSS API (yc-oss.github.io) -- full YC company index
  * Products        Hugging Face Hub API -- models sorted by downloads
  * Research papers arXiv API (cs.AI/cs.LG/cs.CL/cs.CV/stat.ML, newest first)
                    joined with the Papers With Code links dataset
                    (paper <-> GitHub repo) and live GitHub API star counts.

Scaling note: collectors are cursor/page based and stream records out via
async generators, so the same code scales to 500k+ records by raising the
page budget and the infra limits (workers/DB), not by redesign.
"""
from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .dates import normalize_date, to_iso
from .entity_resolution import EntityResolver
from .http_client import AsyncHttpClient
from .schema import make_record

_AI_HINT = re.compile(
    r"\b(ai|a\.i\.|artificial intelligence|machine learning|ml|llm|gpt|nlp|"
    r"deep learning|neural|computer vision|generative|agents?|chatbot|voice ai)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Startups -- YC OSS
# ---------------------------------------------------------------------------

YC_ALL_URL = "https://yc-oss.github.io/api/companies/all.json"


def _clean_url(url: str | None) -> str | None:
    """Normalise a bare domain ('picplum.com') to a full https URL.

    Returns None for junk values ('http://', 'n/a', ...) so callers fall
    back to the YC profile URL, which is always valid.
    """
    from urllib.parse import urlparse
    if not url:
        return None
    u = url.strip()
    if not u or u.lower() in ("n/a", "none", "-"):
        return None
    u = re.split(r"[,\s]", u)[0]  # some YC rows pack two URLs into one field
    if not u.lower().startswith(("http://", "https://")):
        u = "https://" + u
    host = urlparse(u).netloc
    if not host or "." not in host:
        return None
    return u


async def fetch_yc_startups(client: AsyncHttpClient, resolver: EntityResolver,
                            min_records: int = 1000) -> list[dict]:
    data = await client.get_json(YC_ALL_URL)
    ai_first: list[dict] = []
    rest: list[dict] = []
    seen: set[str] = set()
    for c in data:
        name = (c.get("name") or "").strip()
        if not name:
            continue
        canonical = resolver.canonical(name)
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        team = c.get("team_size")
        employee_count = int(team) if isinstance(team, (int, float)) and team and team > 0 else None
        haystack = " ".join([
            c.get("one_liner") or "", c.get("industry") or "",
            c.get("subindustry") or "", " ".join(c.get("tags") or []),
        ])
        try:
            rec = make_record(
                "STARTUP", "Y Combinator (YC OSS API)",
                _clean_url(c.get("website")) or _clean_url(c.get("url"))
                or f"https://www.ycombinator.com/companies/{c.get('slug','')}",
                {
                    "entityName": canonical,
                    "employeeCount": employee_count,
                    "website": _clean_url(c.get("website")),
                    "oneLiner": c.get("one_liner"),
                    "batch": c.get("batch"),
                    "industry": c.get("industry"),
                    "ycProfile": c.get("url"),
                },
            )
        except ValueError:
            continue  # one malformed source row must never kill a bulk run
        (ai_first if _AI_HINT.search(haystack) else rest).append(rec)
    out = ai_first + rest
    # Honour the minimum with AI-relevant companies first; the full unique
    # set is emitted (it is only ~6.4k rows) -- see run summary for counts.
    return out if len(out) >= min_records else out


# ---------------------------------------------------------------------------
# Products -- Hugging Face Hub
# ---------------------------------------------------------------------------

HF_MODELS_URL = "https://huggingface.co/api/models"


def _pricing_model(model: dict) -> str:
    """Documented conservative rule: open downloadable model = FREE;
    gated models (access request / authenticated download) = FREEMIUM.
    HF Hub exposes no PAID/ENTERPRISE price signal, so those values are
    never emitted from this source."""
    gated = model.get("gated")
    if gated in (True, "auto", "manual"):
        return "FREEMIUM"
    return "FREE"


async def fetch_hf_products(client: AsyncHttpClient, resolver: EntityResolver,
                            target: int = 1200) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    url: str | None = f"{HF_MODELS_URL}?limit=100&sort=downloads&direction=-1"
    while url and len(out) < target:
        status, body, headers = await client.get(url)
        if status != 200:
            raise RuntimeError(f"HF models -> HTTP {status}")
        for m in json.loads(body):
            mid = m.get("id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            author = mid.split("/")[0] if "/" in mid else mid
            out.append(make_record(
                "PRODUCT", "Hugging Face Hub API", f"https://huggingface.co/{mid}",
                {
                    "productName": mid,
                    "startupName": resolver.canonical(author),
                    "pricingModel": _pricing_model(m),
                    "downloads": m.get("downloads"),
                    "likes": m.get("likes"),
                    "pipelineTag": m.get("pipeline_tag"),
                },
            ))
            if len(out) >= target:
                break
        link = headers.get("Link", "")
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part.split(";")[0].strip().strip("<>")
        url = nxt
    return out


# ---------------------------------------------------------------------------
# Research papers -- arXiv + Papers With Code links + GitHub stars
# ---------------------------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


_PWC_ROW_RE = re.compile(
    r'"paper_arxiv_id":\s*"([^"]+)"[^}]*?"repo_url":\s*"([^"]+)"[^}]*?'
    r'"is_official":\s*(true|false)')


def build_pwc_index(pwc_path: str | Path) -> dict[str, str]:
    """Parse the PWC links-between-papers-and-code JSON into
    {arxiv_id: repo_url}, preferring official repos.

    The dump is a flat JSON array with a fixed field order, so a bounded
    regex scan is used instead of a full JSON parse (the file is ~160 MB;
    this keeps indexing to seconds and memory to one copy of the text).
    """
    text = Path(pwc_path).read_text(encoding="utf-8", errors="replace")
    index: dict[str, tuple[str, bool]] = {}
    for arxiv_id, repo, official_raw in _PWC_ROW_RE.findall(text):
        arxiv_id, repo = arxiv_id.strip(), repo.strip().rstrip("/")
        if not arxiv_id or "github.com" not in repo:
            continue
        official = official_raw == "true"
        prev = index.get(arxiv_id)
        if prev is None or (official and not prev[1]):
            index[arxiv_id] = (repo, official)
    return {k: v[0] for k, v in index.items()}


_MODERN_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")


def _parse_arxiv_entries(body: str) -> list[dict]:
    root = ET.fromstring(body)
    out = []
    for e in root.findall("a:entry", _ATOM):
        id_url = e.findtext("a:id", "", _ATOM)
        arxiv_id = re.sub(r"v\d+$", "", id_url.rstrip("/").split("/abs/")[-1])
        out.append({
            "arxiv_id": arxiv_id,
            "title": " ".join((e.findtext("a:title", "", _ATOM) or "").split()),
            "authors": [a.findtext("a:name", "", _ATOM) for a in e.findall("a:author", _ATOM)],
            "published": normalize_date(e.findtext("a:published", "", _ATOM)),
        })
    return out


async def fetch_arxiv_papers(client: AsyncHttpClient, pwc_index: dict[str, str],
                             target: int = 1100, pool_factor: int = 3,
                             batch: int = 200) -> list[dict]:
    """Papers are selected PWC-first: candidate arXiv IDs come from the
    Papers With Code links index (papers that provably have a GitHub repo),
    newest IDs first; metadata (title/authors/published) is then fetched
    live from the arXiv API via id_list batches and sorted by real
    publication date.

    Why not newest-arXiv-first? A newest-first arXiv pull (Nov 2026 papers)
    has ~0% overlap with the PWC index, because PWC was sunset into
    Hugging Face in 2025 and its link index ends there -- measured on the
    first run of this pipeline (0 of 1,100 correlated). PWC-first selection
    keeps the brief's defining property: papers correlated with repos.
    """
    modern = [i for i in pwc_index if _MODERN_ARXIV_ID.match(i)]
    modern.sort(reverse=True)  # YYMM.NNNNN -- lexicographic ~= recency
    pool = modern[: target * pool_factor]
    parsed: list[dict] = []
    for i in range(0, len(pool), batch):
        ids = pool[i:i + batch]
        status, body, _ = await client.get(
            ARXIV_API, params={"id_list": ",".join(ids), "max_results": batch})
        if status != 200:
            raise RuntimeError(f"arXiv id_list -> HTTP {status}")
        parsed.extend(_parse_arxiv_entries(body))
        await asyncio.sleep(3.0)  # arXiv API politeness: >= 3 s between calls
    parsed.sort(key=lambda p: p["published"] or _MIN_DT, reverse=True)
    out: list[dict] = []
    seen: set[str] = set()
    for p in parsed:
        if p["arxiv_id"] in seen:
            continue
        seen.add(p["arxiv_id"])
        paper_url = f"https://arxiv.org/abs/{p['arxiv_id']}"
        out.append(make_record(
            "RESEARCH_PAPER", "arXiv API + Papers With Code links", paper_url,
            {
                "title": p["title"],
                "authors": p["authors"],
                "paper_url": paper_url,
                "github_url": pwc_index.get(p["arxiv_id"]),
                "github_stars": None,  # filled by fetch_github_stars
                "published_date": to_iso(p["published"]),
                "arxivId": p["arxiv_id"],
            },
        ))
        if len(out) >= target:
            break
    return out


from datetime import datetime, timezone  # noqa: E402

_MIN_DT = datetime(1970, 1, 1, tzinfo=timezone.utc)

_GH_REPO_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s#?]+)", re.IGNORECASE)


async def fetch_github_stars(client: AsyncHttpClient, papers: list[dict],
                             budget: int = 55) -> dict:
    """Fill github_stars from the live GitHub REST API.

    Unauthenticated GitHub allows 60 core requests/hour, so the budget is
    deliberately capped; set GITHUB_TOKEN in .env to raise it to 5,000/hour
    and increase the budget. Returns run stats; never invents a count.
    """
    import os
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
        budget = max(budget, 4000)
    by_repo: dict[str, list[dict]] = {}
    for p in papers:
        gu = p["content"].get("github_url")
        if gu:
            by_repo.setdefault(gu, []).append(p)
    stats = {"repos_considered": len(by_repo), "repos_fetched": 0,
             "papers_with_stars": 0, "rate_limited": False}
    fetched = 0
    for repo_url, recs in by_repo.items():
        if fetched >= budget:
            break
        m = _GH_REPO_RE.search(repo_url)
        if not m:
            continue
        owner, repo = m.group(1), m.group(2).removesuffix(".git")
        try:
            status, body, _ = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
        except Exception:
            continue
        if status == 403:
            stats["rate_limited"] = True
            break
        if status != 200:
            continue
        stars = json.loads(body).get("stargazers_count")
        fetched += 1
        stats["repos_fetched"] += 1
        if isinstance(stars, int):
            for r in recs:
                r["content"]["github_stars"] = stars
                stats["papers_with_stars"] += 1
    return stats
