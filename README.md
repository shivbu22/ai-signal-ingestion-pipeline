# The AI Signal — AI Engineer Internship Demo Task

**Employer:** GraphOne / FrontierAtlas ("The AI Signal" — AI/venture Intelligence Graph)
**Candidate:** Shivam Yadav · **Deadline:** Mon 23 Nov 2026, 7:00 PM IST
**Deliverable:** a scalable, fault-tolerant ingestion pipeline that builds an
intelligence graph of startups, products, research papers, jobs and news —
from real sources only. **Every record in `data/` traces back to a live
source URL. Nothing is generated, padded or hallucinated; where a source
could not deliver, this README says so plainly.**

---

## 1. Results of the verified run (21 Nov 2026, ~04:13–04:22 UTC)

| Dataset (Sheet tab) | Records | Minimum | Status |
|---|---|---|---|
| Startups | **6,301** | 1,000 | ✅ |
| Products | **1,200** | 1,000 | ✅ |
| Research Papers | **1,100** — 1,100/1,100 with a linked GitHub repo | 1,000 | ✅ |
| Jobs (last 24 h, AI-filtered) | **63** | all genuine finds | ✅ |
| News (last 24 h) | **14** (12 with full text extracted) | all genuine finds | ✅ |
| Entity Mapping Log | **6,705** resolutions | — | ✅ |

Fresh-signal window for this run: **20 Nov 2026 ~04:15 UTC → 21 Nov 2026 ~04:15 UTC**.
Per-source detail is in `data/run_summary.json`.

GitHub star counts: fetched **live from the GitHub REST API for 29 repos**
before the unauthenticated rate limit (60 req/hour, shared egress IP) was
reached; the remaining repos have `github_stars: null` rather than any
estimated number. Set `GITHUB_TOKEN` in `.env` (raises the limit to
5,000/hour) and re-run to fill the rest — the code path is identical.

## 2. Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # optional: add LLM / GitHub keys (see §5)
# one-time: download the Papers With Code links index (~160 MB, gitignored)
curl -L -o data/pwc_links.json \
  https://huggingface.co/datasets/cfahlgren1/links-between-papers-and-code/resolve/main/links-between-papers-and-code.json
.venv/bin/python -m src.run_all --data-dir data
.venv/bin/python -m pytest tests/ -q     # 25 unit tests
```

Outputs land in `data/` as `.jsonl` (canonical) + `.csv` (Sheet-ready),
plus `entity_mapping_log.*`, `run_summary.json` and `state_first_seen.json`.

## 3. Architecture in one page

Full document: **`architecture.pdf`** (2 pages; source in `docs/architecture.md`).

```
YC OSS · HF Hub · arXiv · PWC links · GitHub · 5× RSS · 5× job boards
        │  asyncio + aiohttp collectors (cursor/offset paginated, idempotent)
        ▼
raw landing ─► extraction tiers (Gemini → Groq → DeepSeek → rule-based)
        │     ─► deterministic entity resolution (seed list of 50 AI startups)
        ▼
canonical JSON (schema v1.0) ─► validation gate (invalid ⇒ dropped + counted)
        ▼
PostgreSQL (system of record) · Neo4j (graph projection) · Qdrant (vectors)
Redis (dedup / first-seen) · object storage (raw archive)
```

**Scale to 500k+:** collectors are stateless paginated pullers — more records
means more worker replicas and page budget, not new code. Dedup is a unique
content-hash index, so every write is an idempotent upsert. **413:** hard
per-request chunk budget (default 12,000 chars, boundary-aware splitting with
overlap, deterministic merge) makes oversize payloads impossible; a surprise
413 halves the budget and re-queues once. **429:** exponential backoff
(1 s × 2^attempt, full jitter, 60 s cap, `Retry-After` honoured), then tier
fallback, then the rule-based tier. **Freshness/dedup across nodes:** shared
hash registry (Redis/Postgres, never local disk); undated items count as
fresh only when their content hash is new *and* a baseline run exists.

## 4. Data sources — what was actually used

| Phase | Source | What it gave | Notes |
|---|---|---|---|
| I | YC OSS API `yc-oss.github.io/api/companies/all.json` | 6,301 unique startups | AI-tagged companies sorted first; `team_size` → `employeeCount` (null when absent — never guessed) |
| I | Hugging Face Hub API `/api/models` | 1,200 products (models, by downloads) | `pricingModel`: FREE, or FREEMIUM when the model is gated. HF exposes no paid signal, so PAID/ENTERPRISE are never emitted from this source |
| I | arXiv API (`id_list` batches) | 1,100 papers w/ title, authors, published date | Candidate IDs selected **from the PWC index** (see below) |
| I | Papers With Code links dataset (HF mirror `cfahlgren1/links-between-papers-and-code`) | 204,750 arXiv→GitHub links; official repos preferred | The PWC site/API itself is **dead** (sunset into Hugging Face in 2025; `paperswithcode.com/api/*` now redirects to HF). The mirror is the same official dump |
| I | GitHub REST API | live `stargazers_count` | 29 repos before the unauthenticated 60/hr cap; token raises it |
| II | TechCrunch AI, The Verge AI, Google AI Blog, MarkTechPost, MIT News AI (RSS) | 14 news items ≤ 24 h old, full text fetched for 12 | **VentureBeat's feed returns HTTP 429 from this network even at polite rates** — replaced by MarkTechPost and recorded here |
| II | Hacker News (official Firebase job feed), RemoteOK, Arbeitnow, Jobicy, We Work Remotely RSS | 63 AI jobs ≤ 24 h old (Arbeitnow 46, Jobicy 17) | RemoteOK/HN/WWR genuinely had **zero** AI postings inside this exact 24 h window (verified by hand against the raw API responses — e.g. RemoteOK's newest AI listing was 24 h 43 m old at run time) |

**Why papers are PWC-first:** the first pipeline run pulled the *newest*
arXiv papers (Nov 2026) and got **0 of 1,100** with a repo link — the PWC
index ends with its 2025 sunset, so brand-new papers are not in it yet.
Selection was switched to PWC-linked arXiv IDs (newest first, metadata
fetched live from arXiv), yielding **1,100 of 1,100 papers with a real
GitHub repo**. The trade-off is stated openly: paper *metadata* is current
from arXiv, but the paper set's publication dates end where PWC's index
ends (2025). No stars, repos or papers were invented to hide this.

## 5. Phase III — LLM extraction tiers (honest status)

**No LLM API keys are available in this environment**, so this run's
extraction tier is the **deterministic rule-based extractor (tier 0)** —
every field comes from source-structured data or transparent rules
(role-family classification, remote detection, pricing rule), and
`run_summary.json` records `"extraction_tiers_used": {"rule-based": …}`.

The full orchestrator is implemented in `src/llm_orchestrator.py`:
providers sit behind one interface in a fallback chain
(**Gemini Flash → Groq Llama 3 → DeepSeek**), payloads are chunked so a
request can never hit 413, and 429s get exponential backoff + jitter before
falling to the next tier. Drop keys into `.env` (see `.env.example`) and the
LLM tiers activate automatically — no code changes.

## 6. Phase IV — entity resolution

`src/entity_resolution.py` + `src/seed_startups.json` (50 known AI startups
with alias lists, e.g. "OpenAI, Inc." / "Open AI" → **OpenAI**). Deterministic
chain: exact (1.0) → normalised (0.95) → character-3-gram fuzzy (≥ 0.72) →
self-normalise (0.6). Every decision is logged to the Entity Mapping Log
with raw name, canonical name, method and confidence — 6,705 rows this run,
including 17 seed hits such as `Meta → Meta AI`.

## 7. Phase V — anti-bot & scale strategy

Async everywhere (`asyncio` + `aiohttp`); rotating realistic User-Agents;
per-host token-bucket rate limits (arXiv ≥ 3 s between calls per its API
etiquette); global concurrency semaphore; 429/5xx backoff with jitter;
official APIs and RSS feeds preferred over HTML scraping (robots-friendly
by construction). For JS-heavy/Cloudflare sources the documented fallback
is Playwright async with a real browser profile, used only where a site's
terms permit; CAPTCHA-solving and ToS evasion are out of scope by design.

## 8. Canonical schema (v1.0)

Every record: `schemaVersion`, `recordType`, `source{name,url}`,
`collectedAt`, `content{…}` with the per-type required fields from the
brief (STARTUP `entityName`/`employeeCount`; PRODUCT `startupName`/
`pricingModel`; RESEARCH_PAPER `title`/`authors`/`paper_url`/`github_url`/
`github_stars`/`published_date`; JOB `company`/`date`/`is_remote`/
`role_family`; NEWS `title`/`publishedDate`/`dateMethod`). Records failing
validation are dropped and counted in `run_summary.json` (0 dropped this run).

## 9. Tests

`tests/test_pipeline_units.py` — **25 tests, all passing**: date
normalisation (ISO, RFC-822, epoch s/ms, relative "2 hours ago" / "an hour
ago" / "yesterday", garbage rejection, 24 h window), entity resolution
(exact/normalised/alias/suffix/self + mapping-log), chunking (budget
guarantee, coverage, boundary cases), schema validation, role classification.

## 10. Known limitations (stated, not hidden)

1. GitHub stars cover 29 repos (unauthenticated rate limit) — token fills the rest.
2. Paper–repo correlation relies on the PWC dump, frozen at its 2025 sunset;
   papers published after that cannot have PWC links yet (HF Papers is the
   natural successor source).
3. News/job counts are whatever the real last-24 h window contained at run
   time (14 and 63); they will differ on every run — by design.
4. `employeeCount` is null where YC has no `team_size` (~a third of rows);
   it is never estimated.
5. LLM tiers are implemented and unit-structured but were not exercised
   live in this run, because no API keys exist in this environment (§5).
6. Full text was extracted for 12 of 14 news items; 2 pages failed fetch
   (paywall/JS) and carry the feed summary instead — flagged per record by
   `fullTextChars`.
