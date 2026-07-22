#!/usr/bin/env python3
"""
Recompute per-config aggregates and the comparison table from the saved per-question CSVs.

The sweep runner originally treated a question as answerable whenever its PDF file existed,
but a handful of downloaded PDFs are corrupt and yield no text. This recomputes the answerable
subset using the presence of usable extracted text (the same rule the fixed runner now uses),
so the reported 114-question subset matches the report. Retrieval and generation are not
re-run: every per-question metric is already stored in detailed_all.csv.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results/experiments")
METRIC_COLS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "rouge_l",
               "exact_match", "semantic_similarity", "citation_precision",
               "citation_recall", "citation_f1"]


def usable_docs() -> set:
    """Document names whose cached extraction holds non-empty text."""
    cache = Path("data/extracted_text/pdfplumber")
    out = set()
    for jf in cache.glob("*.json"):
        try:
            if json.load(open(jf)).get("text", "").strip():
                out.add(jf.stem)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def aggregate(df: pd.DataFrame) -> dict:
    """Mean, std, median, min and max for each metric column present in df."""
    agg = {}
    for col in METRIC_COLS:
        if col in df.columns:
            vals = df[col].astype(float)
            agg[f"{col}_mean"] = float(vals.mean())
            agg[f"{col}_std"] = float(vals.std())
            agg[f"{col}_median"] = float(vals.median())
            agg[f"{col}_min"] = float(vals.min())
            agg[f"{col}_max"] = float(vals.max())
    return agg


def main():
    usable = usable_docs()
    summaries = []

    for run_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        detailed = run_dir / "detailed_all.csv"
        summ_path = run_dir / "summary.json"
        if not detailed.exists() or not summ_path.exists():
            print(f"skip {run_dir.name}: missing detailed_all.csv or summary.json")
            continue

        df = pd.read_csv(detailed)
        df["answerable"] = df["doc_name"].isin(usable)
        ans = df[df["answerable"]].copy()

        # Keep the run's descriptive fields, drop the stale scoped metrics.
        summary = json.load(open(summ_path))
        summary = {k: v for k, v in summary.items()
                   if not k.startswith("all.") and not k.startswith("answerable.")}

        summary["n_all"] = len(df)
        summary["n_answerable"] = int(df["answerable"].sum())
        for metric, value in aggregate(df).items():
            summary[f"all.{metric}"] = value
        for metric, value in aggregate(ans).items():
            summary[f"answerable.{metric}"] = value

        ans.to_csv(run_dir / "detailed_answerable.csv", index=False)
        json.dump(summary, open(summ_path, "w"), indent=2)
        summaries.append(summary)
        print(f"{run_dir.name}: n_answerable {summary['n_answerable']}, "
              f"answerable recall@5 {summary.get('answerable.recall@5_mean', float('nan')):.3f}")

    if summaries:
        pd.DataFrame(summaries).to_csv(RESULTS / "comparison.csv", index=False)
        print(f"\nWrote {RESULTS / 'comparison.csv'} with {len(summaries)} configs")


if __name__ == "__main__":
    main()
