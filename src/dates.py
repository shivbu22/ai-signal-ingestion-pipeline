"""Date normalisation for the fresh-signal pipeline (Phase II).

Handles, in order:
  1. ISO-8601 strings (with Z or offset)
  2. RFC-822 / RFC-2822 feed dates (RSS pubDate)
  3. Epoch seconds / milliseconds (job-board APIs)
  4. Relative dates: "2 hours ago", "3 days ago", "just now", "yesterday"
  5. Common human formats: "Nov 20, 2026", "20 November 2026", "2026/11/20"

Returns an aware UTC datetime, or None when the value cannot be parsed.
Undated items are NOT dropped by callers: they fall back to the
first-seen heuristic (content hash tracked across runs) implemented in
sources_fresh.StateStore.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

_REL_RE = re.compile(
    r"(?P<n>\d+|a|an)\s+(?P<unit>second|minute|hour|day|week|month|year)s?\s+ago",
    re.IGNORECASE,
)
_UNIT_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,   # 30 days, documented approximation
    "year": 31536000,   # 365 days, documented approximation
}
_HUMAN_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d %b %Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_relative(text: str, now: datetime | None = None) -> datetime | None:
    now = _to_utc(now or datetime.now(timezone.utc))
    t = text.strip().lower()
    if t in ("just now", "now", "today"):
        return now
    if t == "yesterday":
        return now - timedelta(days=1)
    m = _REL_RE.search(t)
    if not m:
        return None
    n_raw = m.group("n")
    n = 1 if n_raw in ("a", "an") else int(n_raw)
    return now - timedelta(seconds=n * _UNIT_SECONDS[m.group("unit").lower()])


def normalize_date(value, now: datetime | None = None) -> datetime | None:
    """Best-effort conversion of any feed/API date value to aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _to_utc(value)
    if isinstance(value, (int, float)):
        # epoch seconds, or milliseconds when the magnitude demands it
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        if 946684800 <= ts <= 4102444800:  # 2000-01-01 .. 2100-01-01 sanity window
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        return None
    s = str(value).strip()
    if not s:
        return None
    # relative ("2 hours ago")
    rel = parse_relative(s, now=now)
    if rel is not None:
        return rel
    # ISO-8601
    iso = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        return _to_utc(datetime.fromisoformat(iso))
    except ValueError:
        pass
    # RFC-822 (RSS)
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            return _to_utc(dt)
    except (TypeError, ValueError):
        pass
    # human formats
    for fmt in _HUMAN_FORMATS:
        try:
            return _to_utc(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return None


def to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return _to_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def within_last_hours(dt: datetime | None, hours: float = 24.0,
                      now: datetime | None = None) -> bool:
    if dt is None:
        return False
    now = _to_utc(now or datetime.now(timezone.utc))
    delta = (now - _to_utc(dt)).total_seconds()
    return -3600 <= delta <= hours * 3600  # tolerate 1h of publisher clock skew
