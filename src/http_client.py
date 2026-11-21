"""Phase V -- async HTTP client with anti-bot / politeness behaviour.

Strategy (also documented in README + architecture.pdf):
  * asyncio + aiohttp everywhere; Playwright async is the documented fallback
    for JS-heavy pages (see docs/playwright_fallback.py pattern in README).
  * Rotating realistic User-Agents, browser-like Accept headers.
  * Per-host token-bucket rate limiting (default 1 req / 1.5 s per host).
  * 429 / 5xx -> exponential backoff with full jitter, honouring Retry-After.
  * Global concurrency cap via semaphore; per-request timeout.
  * robots.txt awareness: hosts can be allow-listed/deny-listed in config;
    the bulk sources used here are official public APIs / RSS feeds whose
    terms permit this use.
"""
from __future__ import annotations

import asyncio
import random
import time
from urllib.parse import urlparse

import aiohttp

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


class BackoffPolicy:
    """Exponential backoff with full jitter for 429/5xx (Phase III/V rule)."""

    def __init__(self, base: float = 1.0, factor: float = 2.0, cap: float = 60.0,
                 max_retries: int = 4):
        self.base, self.factor, self.cap, self.max_retries = base, factor, cap, max_retries

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.cap * 2)
        ceiling = min(self.cap, self.base * (self.factor ** attempt))
        return random.uniform(0, ceiling)  # full jitter


class AsyncHttpClient:
    def __init__(self, max_concurrency: int = 8, per_host_interval: float = 1.5,
                 timeout: float = 40.0, backoff: BackoffPolicy | None = None):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._per_host_interval = per_host_interval
        self._host_last: dict[str, float] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._backoff = backoff or BackoffPolicy()
        self._session: aiohttp.ClientSession | None = None
        self.stats = {"requests": 0, "retries": 0, "rate_limited": 0, "failures": 0}

    async def __aenter__(self) -> "AsyncHttpClient":
        connector = aiohttp.TCPConnector(limit=32, ttl_dns_cache=300)
        # trust_env=True so http_proxy/https_proxy environment settings are
        # honoured (required on networks where egress goes through a proxy).
        self._session = aiohttp.ClientSession(connector=connector, timeout=self._timeout,
                                              trust_env=True)
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    async def _throttle(self, host: str):
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._per_host_interval - (time.monotonic() - self._host_last.get(host, 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
            self._host_last[host] = time.monotonic()

    def _headers(self, extra: dict | None) -> dict:
        h = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/json,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        }
        if extra:
            h.update(extra)
        return h

    async def get(self, url: str, headers: dict | None = None,
                  params: dict | None = None) -> tuple[int, str, dict]:
        """GET with throttle + retry. Returns (status, body_text, response_headers).

        Raises aiohttp.ClientError / asyncio.TimeoutError after retries on
        network errors; HTTP error statuses are returned, not raised, except
        that 429/5xx are retried first.
        """
        assert self._session is not None, "use 'async with AsyncHttpClient()'"
        host = urlparse(url).netloc
        attempt = 0
        while True:
            await self._throttle(host)
            try:
                async with self._sem:
                    self.stats["requests"] += 1
                    async with self._session.get(url, headers=self._headers(headers),
                                                 params=params) as resp:
                        body = await resp.text(errors="replace")
                        status = resp.status
                        resp_headers = dict(resp.headers)
                if status in (429, 500, 502, 503, 504) and attempt < self._backoff.max_retries:
                    if status == 429:
                        self.stats["rate_limited"] += 1
                    self.stats["retries"] += 1
                    ra = resp_headers.get("Retry-After")
                    delay = self._backoff.delay(attempt, float(ra) if ra and ra.isdigit() else None)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                return status, body, resp_headers
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt >= self._backoff.max_retries:
                    self.stats["failures"] += 1
                    raise
                self.stats["retries"] += 1
                await asyncio.sleep(self._backoff.delay(attempt))
                attempt += 1

    async def post_json(self, url: str, payload: dict,
                        headers: dict | None = None) -> tuple[int, str, dict]:
        """POST JSON with the same throttle + 429/5xx backoff policy as GET."""
        assert self._session is not None, "use 'async with AsyncHttpClient()'"
        host = urlparse(url).netloc
        attempt = 0
        while True:
            await self._throttle(host)
            try:
                async with self._sem:
                    self.stats["requests"] += 1
                    async with self._session.post(url, json=payload,
                                                  headers=self._headers(headers)) as resp:
                        body = await resp.text(errors="replace")
                        status = resp.status
                        resp_headers = dict(resp.headers)
                if status in (429, 500, 502, 503, 504) and attempt < self._backoff.max_retries:
                    if status == 429:
                        self.stats["rate_limited"] += 1
                    self.stats["retries"] += 1
                    ra = resp_headers.get("Retry-After")
                    delay = self._backoff.delay(attempt, float(ra) if ra and ra.isdigit() else None)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                return status, body, resp_headers
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt >= self._backoff.max_retries:
                    self.stats["failures"] += 1
                    raise
                self.stats["retries"] += 1
                await asyncio.sleep(self._backoff.delay(attempt))
                attempt += 1

    async def get_json(self, url: str, headers: dict | None = None,
                       params: dict | None = None):
        import json
        status, body, _ = await self.get(url, headers=headers, params=params)
        if status != 200:
            raise RuntimeError(f"GET {url} -> HTTP {status}")
        return json.loads(body)
