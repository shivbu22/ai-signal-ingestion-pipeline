# AI / Venture Intelligence Graph — Ingestion Architecture

**System:** fault-tolerant ingestion pipeline feeding an intelligence graph of startups,
products, research papers, jobs and news. **Stack:** Python 3.12, asyncio + aiohttp,
canonical JSON schema v1.0. Every record carries a verifiable `source.url`; no record is
ever synthesised — the anti-hallucination contract enforced by schema validation.

## 1. Pipeline shape

```
 Sources                Ingest workers              Extract            Store
 ─────────────          ──────────────────          ────────────────   ─────────────────
 YC OSS API        ┐
 HF Hub API        ├─►  async collectors ──► raw landing (JSONL) ──►  PostgreSQL (system of record)
 arXiv API         │    (per-source,          │                      Neo4j (entity graph)
 PWC links dump    │     cursor-paginated)    ├─► entity resolution  Qdrant (embeddings)
 GitHub API        │                          │   (deterministic)    Redis (dedup / first-seen)
 5× RSS news       ├─►  fresh-signal pollers ─┤                      Object storage (raw HTML/text)
 5× job boards     ┘    (24 h window)         └─► LLM tier chain ──►  validated canonical records
                                                     Gemini → Groq → DeepSeek → rule-based
```

Collectors are **stateless and idempotent**: each emits canonical records keyed by a
content hash (SHA-256 of normalised text) plus the source URL. Re-running any collector
over the same window produces the same record set, so retries never duplicate data.
Phases: **I** bulk backfill (startups/products/papers), **II** fresh signals (news/jobs,
24 h), **III** LLM extraction tiers, **IV** deterministic entity resolution,
**V** anti-bot transport, **VI** this document.

## 2. Scaling to 500k+ records — infra scaling only

* **Collectors scale horizontally.** Sources are independent; each collector is a
  paginated puller (cursor for HF, offset for arXiv, single-document for YC). Adding
  records = raising page budgets and adding worker replicas behind a queue
  (one queue message per page/cursor), no code changes.
* **Backpressure by design.** A global semaphore caps in-flight requests; per-host
  token buckets cap per-source rates. Workers slow down instead of falling over.
* **Storage scales vertically then by partition.** PostgreSQL partitioned by
  `recordType` + month; content-hash unique index makes inserts idempotent upserts.
  500k canonical records ≈ low single-digit GB — one managed Postgres instance is
  ample; read replicas are added before any sharding is needed.
* **Joins are precomputed.** The paper↔code index (204,750 arXiv→GitHub links) is a
  build-time lookup table, refreshed on a schedule, never queried live per record.
* **Freshness pollers** run on a 15-minute cadence; the 24 h window is a filter, not a
  storage horizon, so steady-state ingest volume stays small even as the corpus grows.

## 3. Exact 413 / 429 strategy

**413 (payload too large).** Prevention, not reaction: every LLM payload passes through
`chunk_text` with a hard per-request budget (default 12,000 chars, configurable via
`LLM_MAX_CHARS_PER_REQUEST`). Chunks split on paragraph/sentence boundaries with a
200-char overlap; per-chunk results merge deterministically (later chunks only fill
empty fields). A single request therefore *cannot* exceed the provider limit. If a 413
still occurs (provider lowers a limit), the chunk budget halves and the item is
re-queued once before being routed to the rule-based tier.

**429 (rate limited).** One shared `BackoffPolicy` for HTTP and LLM calls:
exponential backoff `base 1 s × 2^attempt`, **full jitter**, cap 60 s, max 4 retries;
a `Retry-After` header overrides the computed delay. After the retry budget, the
orchestrator **falls back to the next LLM tier** (Gemini Flash → Groq Llama 3 →
DeepSeek) and finally to the deterministic rule-based extractor, so a rate-limited
provider degrades quality gracefully instead of stalling the pipeline. Per-host token
buckets keep us below published limits in the first place (e.g. arXiv ≥ 3 s between
calls; GitHub star lookups budgeted to the unauthenticated 60 req/h, raised to
5,000/h when a token is configured).

## 4. Freshness & dedup across distributed nodes

* **Dedup key** = `SHA-256(normalised content)` + source URL, stored in Redis
  (`SET NX` with TTL) as the fast path and in PostgreSQL as the durable unique index.
  Any node can ingest any source; the unique index is the final arbiter, so duplicate
  delivery is harmless (idempotent upsert).
* **First-seen tracking** for undated items: a persistent hash registry records
  `hash → first_seen_at`. An undated item is "fresh" only if its hash is new *and* a
  baseline run exists — on the baseline run undated items are registered but not
  emitted, because their age cannot be proven. In distributed deployment the registry
  lives in Redis/PostgreSQL (never local disk), so all nodes share one first-seen truth.
* **Dated items** are normalised to UTC (ISO-8601, RFC-822, epoch, and relative forms
  like "2 hours ago") and filtered by the 24 h window with a 1 h publisher clock-skew
  tolerance.
* **Entity resolution** is deterministic (exact → normalised → n-gram fuzzy → self)
  against a seed alias table, so two nodes always canonicalise the same raw name to
  the same entity; every decision is written to the Entity Mapping Log for audit.

## 5. Storage justification

| Layer | Choice | Why |
|---|---|---|
| Primary DB | **PostgreSQL** | Canonical records are structured, relational (paper↔repo, product↔startup) and need transactional idempotent upserts + a unique content-hash index. JSONB columns hold schema-v1.0 payloads without migrations as the schema evolves. |
| Graph | **Neo4j** | The product *is* an intelligence graph: startup→product→paper→repo traversals ("papers by founders of X", "repos behind trending papers") are multi-hop queries that are painful in SQL and native in a graph store. Postgres remains the system of record; Neo4j is a derived projection, rebuildable from it. |
| Vector | **Qdrant** | Semantic dedup and "similar startup/paper" retrieval need ANN search over embeddings at 500k+ scale; Qdrant gives filtered vector search (by recordType/date) that a Postgres extension matches only at smaller scale. |
| Dedup / first-seen | **Redis** | Sub-millisecond `SET NX` checks on the hot ingest path; durable copy in Postgres. |
| Raw archive | **Object storage (S3-class)** | Raw HTML/full text is kept for re-extraction when schemas change — cheap, immutable, and the audit trail behind every canonical record. |

## 6. Fault tolerance summary

Per-source error isolation (one dead feed never fails a run), retry with backoff on
429/5xx/timeouts, tier fallback for extraction, idempotent writes everywhere, and a
run summary that reports per-source counts and errors — a partial run is always
visible as partial, never silently presented as complete.
