"""Pipeline runner -- executes Phases I + II and writes the six datasets.

Usage:
    python -m src.run_all [--data-dir data] [--pwc-links PATH]
                          [--startup-min 1000] [--product-target 1200]
                          [--paper-target 1100] [--fresh-hours 24]
                          [--github-star-budget 55] [--skip-bulk] [--skip-fresh]

Outputs (in --data-dir):
    startups.jsonl/.csv, products.jsonl/.csv, papers.jsonl/.csv,
    jobs.jsonl/.csv, news.jsonl/.csv,
    entity_mapping_log.jsonl/.csv, run_summary.json, state_first_seen.json

Every record is validated against the canonical schema before writing;
a record that fails validation is dropped and counted, never repaired
with invented values.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import sources_bulk, sources_fresh  # noqa: E402
from src.entity_resolution import EntityResolver  # noqa: E402
from src.http_client import AsyncHttpClient  # noqa: E402
from src.llm_orchestrator import ExtractionOrchestrator  # noqa: E402
from src.schema import now_iso, validate_record  # noqa: E402

CSV_FIELDS = {
    "startups": ["entityName", "employeeCount", "website", "oneLiner", "batch",
                 "industry", "ycProfile", "source_name", "source_url", "collectedAt"],
    "products": ["productName", "startupName", "pricingModel", "downloads", "likes",
                 "pipelineTag", "source_name", "source_url", "collectedAt"],
    "papers": ["title", "authors", "paper_url", "github_url", "github_stars",
               "published_date", "arxivId", "source_name", "source_url", "collectedAt"],
    "jobs": ["company", "title", "date", "is_remote", "role_family", "location",
             "source_name", "source_url", "collectedAt"],
    "news": ["title", "publishedDate", "dateMethod", "summary", "fullTextChars",
             "source_name", "source_url", "collectedAt"],
}


def _flatten(rec: dict) -> dict:
    row = dict(rec["content"])
    row["source_name"] = rec["source"]["name"]
    row["source_url"] = rec["source"]["url"]
    row["collectedAt"] = rec["collectedAt"]
    if isinstance(row.get("authors"), list):
        row["authors"] = "; ".join(row["authors"])
    return row


def write_dataset(data_dir: Path, name: str, records: list[dict]) -> dict:
    valid, dropped = [], 0
    for r in records:
        if validate_record(r):
            dropped += 1
        else:
            valid.append(r)
    jsonl_path = data_dir / f"{name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in valid:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    fields = CSV_FIELDS[name]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in valid:
        w.writerow(_flatten(r))
    (data_dir / f"{name}.csv").write_text(buf.getvalue(), encoding="utf-8")
    return {"records": len(valid), "dropped_invalid": dropped}


def write_mapping_log(data_dir: Path, resolver: EntityResolver) -> int:
    rows, seen = [], set()
    for entry in resolver.mapping_log:
        key = (entry["raw_name"], entry["canonical_name"], entry["method"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(entry)
    with open(data_dir / "entity_mapping_log.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["raw_name", "canonical_name", "method", "confidence"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
    (data_dir / "entity_mapping_log.csv").write_text(buf.getvalue(), encoding="utf-8")
    return len(rows)


async def main_async(args) -> dict:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    resolver = EntityResolver()
    orchestrator = ExtractionOrchestrator()
    summary: dict = {"startedAt": now_iso(), "phases": {}}

    async with AsyncHttpClient(max_concurrency=8, per_host_interval=1.2) as client:
        if not args.skip_bulk:
            startups = await sources_bulk.fetch_yc_startups(client, resolver)
            summary["phases"]["startups"] = write_dataset(data_dir, "startups", startups)

            products = await sources_bulk.fetch_hf_products(
                client, resolver, target=args.product_target)
            summary["phases"]["products"] = write_dataset(data_dir, "products", products)

            pwc_index: dict = {}
            if args.pwc_links and Path(args.pwc_links).exists():
                pwc_index = sources_bulk.build_pwc_index(args.pwc_links)
            summary["pwc_index_size"] = len(pwc_index)
            papers = await sources_bulk.fetch_arxiv_papers(
                client, pwc_index, target=args.paper_target)
            star_stats = await sources_bulk.fetch_github_stars(
                client, papers, budget=args.github_star_budget)
            summary["github_stars"] = star_stats
            summary["papers_with_github_url"] = sum(
                1 for p in papers if p["content"].get("github_url"))
            summary["phases"]["papers"] = write_dataset(data_dir, "papers", papers)

        if not args.skip_fresh:
            state = sources_fresh.StateStore(data_dir / "state_first_seen.json")
            news, news_stats = await sources_fresh.fetch_news(
                client, state, orchestrator, hours=args.fresh_hours)
            summary["news_sources"] = news_stats
            summary["phases"]["news"] = write_dataset(data_dir, "news", news)
            jobs, job_stats = await sources_fresh.fetch_jobs(
                client, state, resolver, orchestrator, hours=args.fresh_hours)
            summary["job_sources"] = job_stats
            summary["phases"]["jobs"] = write_dataset(data_dir, "jobs", jobs)
            state.save()

        summary["http"] = client.stats
    summary["entity_mapping_rows"] = write_mapping_log(data_dir, resolver)
    summary["extraction_tiers_used"] = orchestrator.used_tiers
    summary["llm_tiers_configured"] = [p.name for p in orchestrator.active_llm_tiers()]
    summary["finishedAt"] = now_iso()
    (data_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pwc-links", default=os.environ.get(
        "PWC_LINKS_PATH", "data/pwc_links.json"))
    ap.add_argument("--product-target", type=int, default=1200)
    ap.add_argument("--paper-target", type=int, default=1100)
    ap.add_argument("--fresh-hours", type=float, default=24.0)
    ap.add_argument("--github-star-budget", type=int, default=55)
    ap.add_argument("--skip-bulk", action="store_true")
    ap.add_argument("--skip-fresh", action="store_true")
    args = ap.parse_args()
    summary = asyncio.run(main_async(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
