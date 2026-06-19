"""
Evaluate ingest speed, rate-limit behavior, and retrieval quality.

Usage (from project root):
    python test/run_eval.py
    python test/run_eval.py --config batch10_delay8
    python test/run_eval.py --skip-ingest --with-llm
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import collection
from services.chunk import chunk_code, parse_code
from services.embedding import clear_collection, embed_chunks

CONFIGS = {
    "batch30_delay1": {"BATCH_SIZE": "30", "EMBED_BATCH_DELAY": "1"},
    "batch10_delay8": {"BATCH_SIZE": "10", "EMBED_BATCH_DELAY": "8"},
}

DEFAULT_REPO = "https://github.com/arjunMandair17/RAG-Codebase-Query"


def load_golden_set() -> list[dict]:
    """Load evaluation questions and expected sources."""
    path = Path(__file__).parent / "golden_set.json"
    return json.loads(path.read_text(encoding="utf-8"))


def score_retrieval(query: str, expected_paths: list[str], n_results: int = 10) -> dict:
    """Score whether retrieved chunks include expected file paths."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas"],
    )
    paths = [m["path"] for m in results["metadatas"][0]] if results["metadatas"] else []
    expected = set(expected_paths)

    hit = any(p in paths for p in expected_paths)
    mrr = 0.0
    for rank, path in enumerate(paths, 1):
        if path in expected:
            mrr = 1.0 / rank
            break
    recall = len(expected & set(paths)) / len(expected) if expected else 0.0

    return {"hit_at_k": hit, "mrr": mrr, "recall_at_k": recall, "retrieved_paths": paths[:5]}


def score_keywords(text: str, expected_keywords: list[str]) -> float:
    """Return fraction of expected keywords found in text."""
    if not expected_keywords:
        return 0.0
    lower = text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in lower)
    return found / len(expected_keywords)


async def run_ingest(repo_url: str) -> dict:
    """Fetch, chunk, and embed a repo while collecting timing and error stats."""
    stats = {
        "batches_ok": 0,
        "batches_failed": 0,
        "chunks_embedded": 0,
        "rate_limit_retries": 0,
        "other_errors": 0,
    }
    metrics = {"success": False}

    t0 = time.perf_counter()
    files = await parse_code(repo_url)
    metrics["fetch_s"] = round(time.perf_counter() - t0, 2)

    t1 = time.perf_counter()
    chunks = []
    for file in files:
        chunks.extend(chunk_code(file["content"], file["language"], file["path"], file["extension"]))
    metrics["chunk_s"] = round(time.perf_counter() - t1, 2)
    metrics["chunk_count"] = len(chunks)

    t2 = time.perf_counter()
    metrics["success"] = embed_chunks(chunks, stats=stats)
    metrics["embed_s"] = round(time.perf_counter() - t2, 2)
    metrics["total_ingest_s"] = round(time.perf_counter() - t0, 2)
    metrics.update(stats)

    return metrics


def run_retrieval_eval(golden_set: list[dict], with_llm: bool = False) -> dict:
    """Run retrieval metrics (and optional LLM keyword checks) on the golden set."""
    if with_llm:
        from services.llm import generate_response

    hits, mrrs, recalls, keyword_scores = [], [], [], []

    for item in golden_set:
        scores = score_retrieval(item["query"], item["expected_paths"])
        hits.append(scores["hit_at_k"])
        mrrs.append(scores["mrr"])
        recalls.append(scores["recall_at_k"])

        if with_llm:
            answer = generate_response(item["query"])
            keyword_scores.append(score_keywords(answer, item.get("expected_keywords", [])))

    result = {
        "hit_at_k": round(sum(hits) / len(hits), 3) if hits else 0,
        "mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else 0,
        "recall_at_k": round(sum(recalls) / len(recalls), 3) if recalls else 0,
        "questions": len(golden_set),
    }
    if with_llm and keyword_scores:
        result["keyword_score"] = round(sum(keyword_scores) / len(keyword_scores), 3)

    return result


async def run_config(name: str, repo_url: str, golden_set: list[dict], skip_ingest: bool, with_llm: bool) -> dict:
    """Run a full evaluation for one batch/delay configuration."""
    os.environ.update(CONFIGS[name])

    result = {"config": name, "batch_size": CONFIGS[name]["BATCH_SIZE"], "embed_delay": CONFIGS[name]["EMBED_BATCH_DELAY"]}

    if not skip_ingest:
        clear_collection()
        result["ingest"] = await run_ingest(repo_url)

    result["retrieval"] = run_retrieval_eval(golden_set, with_llm=with_llm)
    return result


def print_summary(results: list[dict]) -> None:
    """Print a comparison table to the console."""
    print("\n=== Eval Summary ===")
    header = f"{'Config':<18} {'Ingest(s)':<10} {'429 retries':<12} {'Hit@k':<8} {'MRR':<8} {'OK':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        ingest = r.get("ingest", {})
        retr = r.get("retrieval", {})
        print(
            f"{r['config']:<18} "
            f"{str(ingest.get('total_ingest_s', '-')):<10} "
            f"{str(ingest.get('rate_limit_retries', '-')):<12} "
            f"{str(retr.get('hit_at_k', '-')):<8} "
            f"{str(retr.get('mrr', '-')):<8} "
            f"{str(ingest.get('success', '-')):<6}"
        )


async def main() -> None:
    """Run evaluation across selected configs and save results."""
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline performance and retrieval quality.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo URL to ingest")
    parser.add_argument("--config", choices=list(CONFIGS), help="Run a single config (default: all)")
    parser.add_argument("--skip-ingest", action="store_true", help="Only run retrieval eval on existing DB")
    parser.add_argument("--with-llm", action="store_true", help="Also score LLM answers by keyword overlap")
    args = parser.parse_args()

    golden_set = load_golden_set()
    config_names = [args.config] if args.config else list(CONFIGS)

    results = []
    for name in config_names:
        print(f"\nRunning config: {name}")
        results.append(await run_config(name, args.repo, golden_set, args.skip_ingest, args.with_llm))

    print_summary(results)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
