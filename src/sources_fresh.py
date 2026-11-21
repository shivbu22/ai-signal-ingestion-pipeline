"""Phase II -- fresh signal ingestion (last 24 hours only).

News: 5 AI news RSS feeds. Job boards: 5 sources (HN official jobs feed,
RemoteOK, Arbeitnow, Jobicy, We Work Remotely RSS).

Freshness rules:
  * Items with a parseable date are kept iff within the last 24 h
    (dates normalised by src/dates.py, incl. relative dates).
  * Undated items use the first-seen heuristic: their content hash is kept
    in a persistent state file; an undated item counts as fresh only when
    its hash is new AND a baseline run already exists (on the baseline run
    undated items are recorded but not emitted -- we cannot prove they are
    <24 h old, and this pipeline never guesses).
  * Full text is fetched for every kept news item (HTML -> text extractor).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import feedparser

from .dates import normalize_date, to_iso, within_last_hours
from .entity_resolution import EntityResolver
from .http_client import AsyncHttpClient
from .llm_orchestrator import ExtractionOrchestrator
from .schema import content_hash, make_record

AI_KEYWORDS = re.compile(
    r"\b(ai|artificial intelligence|machine learning|ml engineer|llm|gpt|"
    r"deep learning|nlp|data scien\w*|genai|generative ai|computer vision|"
    r"prompt engineer|mlops|neural network)\b", re.IGNORECASE)

NEWS_SOURCES = [
    {"name": "TechCrunch AI", "feed": "https://techcrunch.com/category/artificial-intelligence/feed/"},
    {"name": "The Verge AI", "feed": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"},
    {"name": "Google AI Blog", "feed": "https://blog.google/innovation-and-ai/technology/ai/rss/"},
    {"name": "MarkTechPost", "feed": "https://www.marktechpost.com/feed/"},
    {"name": "MIT News AI", "feed": "https://news.mit.edu/rss/topic/artificial-intelligence2"},
    # VentureBeat AI was in the original plan but its feed returns HTTP 429
    # from this network even at polite rates; MarkTechPost replaces it.
    # See README "Data sources" for the substitution record.
]


class StateStore:
    """Persistent first-seen registry: content_hash -> first_seen ISO."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.seen: dict[str, str] = {}
        self.baseline_done = False
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.seen = data.get("seen", {})
            self.baseline_done = bool(data.get("baseline_done"))

    def is_new(self, h: str) -> bool:
        return h not in self.seen

    def mark(self, h: str):
        self.seen.setdefault(h, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"baseline_done": True, "seen": self.seen}, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "header", "footer", "nav", "form"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag in ("p", "br", "div", "article", "h1", "h2", "h3", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str, cap: int = 20000) -> str:
    p = _TextExtractor()
    try:
        p.feed(html or "")
    except Exception:
        return ""
    text = re.sub(r"[ \t]+", " ", "".join(p.parts))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:cap]


async def _fetch_feed(client: AsyncHttpClient, url: str):
    status, body, _ = await client.get(url)
    if status != 200:
        raise RuntimeError(f"feed {url} -> HTTP {status}")
    return feedparser.parse(body)


async def fetch_news(client: AsyncHttpClient, state: StateStore,
                     orchestrator: ExtractionOrchestrator,
                     hours: float = 24.0) -> tuple[list[dict], dict]:
    out: list[dict] = []
    stats: dict[str, dict] = {}
    seen_links: set[str] = set()
    for src in NEWS_SOURCES:
        st = {"entries": 0, "kept": 0, "error": None}
        try:
            feed = await _fetch_feed(client, src["feed"])
            for e in feed.entries:
                st["entries"] += 1
                link = e.get("link") or ""
                title = " ".join((e.get("title") or "").split())
                if not link or not title or link in seen_links:
                    continue
                raw_date = e.get("published") or e.get("updated")
                dt = normalize_date(raw_date)
                date_method = "feed"
                if dt is not None:
                    if not within_last_hours(dt, hours):
                        continue
                else:
                    h = content_hash(link + title)
                    if not (state.baseline_done and state.is_new(h)):
                        state.mark(h)
                        continue
                    state.mark(h)
                    date_method = "first_seen"
                seen_links.add(link)
                summary = html_to_text(e.get("summary", ""), cap=1000)
                full_text = ""
                try:
                    s2, page, _ = await client.get(link)
                    if s2 == 200:
                        full_text = html_to_text(page)
                except Exception:
                    full_text = ""
                extracted = orchestrator.extract_news(
                    title, full_text or summary,
                    {"date": to_iso(dt) if dt else None})
                out.append(make_record("NEWS", src["name"], link, {
                    "title": title,
                    "publishedDate": extracted["date"],
                    "dateMethod": date_method,
                    "summary": summary or extracted["summary"],
                    "fullText": full_text,
                    "fullTextChars": len(full_text),
                }))
                st["kept"] += 1
        except Exception as exc:  # one bad feed must not kill the run
            st["error"] = str(exc)[:200]
        stats[src["name"]] = st
    return out, stats


# ---------------------------------------------------------------------------
# Job boards
# ---------------------------------------------------------------------------

def _job_record(source_name: str, url: str, company_raw: str, title: str,
                dt, is_remote_hint: bool | None, location: str | None,
                resolver: EntityResolver, orchestrator: ExtractionOrchestrator,
                text: str = "") -> dict | None:
    if not company_raw or not title or not url:
        return None
    extracted = orchestrator.extract_job(title, text, {
        "company": resolver.canonical(company_raw),
        "date": to_iso(dt),
        "is_remote": is_remote_hint if is_remote_hint is not None else False,
    })
    return make_record("JOB", source_name, url, {
        "company": extracted["company"],
        "title": title,
        "date": extracted["date"],
        "is_remote": bool(extracted["is_remote"]),
        "role_family": extracted["role_family"],
        "location": location,
    })


async def fetch_jobs(client: AsyncHttpClient, state: StateStore,
                     resolver: EntityResolver,
                     orchestrator: ExtractionOrchestrator,
                     hours: float = 24.0) -> tuple[list[dict], dict]:
    out: list[dict] = []
    stats: dict[str, dict] = {}
    seen_urls: set[str] = set()

    def fresh(dt) -> bool:
        return dt is not None and within_last_hours(dt, hours)

    # 1) Hacker News -- official Firebase job stories feed
    name = "Hacker News Jobs"
    st = {"entries": 0, "kept": 0, "error": None}
    try:
        ids = await client.get_json("https://hacker-news.firebaseio.com/v0/jobstories.json")
        for item_id in ids[:40]:
            st["entries"] += 1
            item = await client.get_json(
                f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json")
            if not item or item.get("type") != "job":
                continue
            dt = normalize_date(item.get("time"))
            title = item.get("title") or ""
            text = html_to_text(item.get("text") or "", cap=2000)
            if not fresh(dt) or not AI_KEYWORDS.search(f"{title} {text}"):
                continue
            url = item.get("url") or f"https://news.ycombinator.com/item?id={item_id}"
            company = title.split(" is hiring")[0].split(" (")[0].strip() if " is hiring" in title else None
            rec = _job_record(name, url, company or "Unknown (HN post)", title, dt,
                              None, None, resolver, orchestrator, text)
            if rec and url not in seen_urls:
                seen_urls.add(url)
                out.append(rec)
                st["kept"] += 1
    except Exception as exc:
        st["error"] = str(exc)[:200]
    stats[name] = st

    # 2) RemoteOK API
    name = "RemoteOK"
    st = {"entries": 0, "kept": 0, "error": None}
    try:
        data = await client.get_json("https://remoteok.com/api")
        for job in data[1:]:  # [0] is the legal notice
            st["entries"] += 1
            dt = normalize_date(job.get("date") or job.get("epoch"))
            title = job.get("position") or ""
            tags = " ".join(job.get("tags") or [])
            if not fresh(dt) or not AI_KEYWORDS.search(f"{title} {tags}"):
                continue
            url = job.get("url") or ""
            if url in seen_urls:
                continue
            rec = _job_record(name, url, job.get("company"), title, dt, True,
                              job.get("location"), resolver, orchestrator, tags)
            if rec:
                seen_urls.add(url)
                out.append(rec)
                st["kept"] += 1
    except Exception as exc:
        st["error"] = str(exc)[:200]
    stats[name] = st

    # 3) Arbeitnow API
    name = "Arbeitnow"
    st = {"entries": 0, "kept": 0, "error": None}
    try:
        data = await client.get_json("https://www.arbeitnow.com/api/job-board-api")
        for job in data.get("data", []):
            st["entries"] += 1
            dt = normalize_date(job.get("created_at"))
            title = job.get("title") or ""
            tags = " ".join(job.get("tags") or [])
            if not fresh(dt) or not AI_KEYWORDS.search(f"{title} {tags} {job.get('description','')[:500]}"):
                continue
            url = job.get("url") or ""
            if url in seen_urls:
                continue
            rec = _job_record(name, url, job.get("company_name"), title, dt,
                              bool(job.get("remote")), job.get("location"),
                              resolver, orchestrator, tags)
            if rec:
                seen_urls.add(url)
                out.append(rec)
                st["kept"] += 1
    except Exception as exc:
        st["error"] = str(exc)[:200]
    stats[name] = st

    # 4) Jobicy API
    name = "Jobicy"
    st = {"entries": 0, "kept": 0, "error": None}
    try:
        data = await client.get_json("https://jobicy.com/api/v2/remote-jobs?count=50")
        for job in data.get("jobs", []):
            st["entries"] += 1
            dt = normalize_date(job.get("pubDate"))
            title = job.get("jobTitle") or ""
            if not fresh(dt) or not AI_KEYWORDS.search(
                    f"{title} {job.get('jobIndustry','')} {job.get('jobDescription','')[:400]}"):
                continue
            url = job.get("url") or ""
            if url in seen_urls:
                continue
            rec = _job_record(name, url, job.get("companyName"), title, dt, True,
                              job.get("jobGeo"), resolver, orchestrator,
                              job.get("jobDescription", "")[:1000])
            if rec:
                seen_urls.add(url)
                out.append(rec)
                st["kept"] += 1
    except Exception as exc:
        st["error"] = str(exc)[:200]
    stats[name] = st

    # 5) We Work Remotely -- programming RSS, AI-filtered
    name = "We Work Remotely"
    st = {"entries": 0, "kept": 0, "error": None}
    try:
        feed = await _fetch_feed(
            client, "https://weworkremotely.com/categories/remote-programming-jobs.rss")
        for e in feed.entries:
            st["entries"] += 1
            dt = normalize_date(e.get("published") or e.get("updated"))
            title = e.get("title") or ""
            if not fresh(dt) or not AI_KEYWORDS.search(f"{title} {e.get('summary','')}"):
                continue
            url = e.get("link") or ""
            if url in seen_urls:
                continue
            company = title.split(":")[0].strip() if ":" in title else None
            rec = _job_record(name, url, company, title, dt, True, "Remote",
                              resolver, orchestrator, e.get("summary", ""))
            if rec:
                seen_urls.add(url)
                out.append(rec)
                st["kept"] += 1
    except Exception as exc:
        st["error"] = str(exc)[:200]
    stats[name] = st

    return out, stats
