#!/usr/bin/env python3
"""
Turn the ablation sweep output into the figures and tables the report needs.

Reads results/experiments/comparison.csv and the per-run detailed CSVs, writes PNGs to
results/figures/ and a Markdown results section to results/results_summary.md.

Run after run_experiments.py:
    python -m src.evaluation.make_figures
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

RETRIEVAL_METRICS = ["recall@1", "recall@5", "recall@10", "mrr"]
GENERATION_METRICS = ["rouge_l", "semantic_similarity", "exact_match"]
CITATION_METRICS = ["citation_precision", "citation_recall", "citation_f1"]


def load_comparison(root: Path) -> pd.DataFrame:
    """
    Load the across-experiment comparison table.

    Args:
        root: results/experiments directory

    Returns:
        The comparison DataFrame
    """
    path = root / "comparison.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run run_experiments.py first.")
    return pd.read_csv(path)


def plot_axis(df: pd.DataFrame, axis: str, metric_col: str, fname: Path, title: str):
    """
    Bar plot of one metric across the experiments that share an ablation axis.

    Args:
        df: Comparison DataFrame
        axis: Ablation axis to filter on, or "all"
        metric_col: Column to plot
        fname: Output PNG path
        title: Plot title
    """
    sub = df if axis == "all" else df[df["axis"].isin([axis, "reference"])]
    sub = sub[sub[metric_col].notna()]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(sub) * 1.3), 5))
    colors = ["#4C72B0" if a == "reference" else "#55A868" for a in sub["axis"]]
    ax.bar(sub["experiment"], sub[metric_col], color=colors)
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, sub[metric_col].max() * 1.25) or 1.0)
    for x, v in zip(sub["experiment"], sub[metric_col]):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_recall_curve(df: pd.DataFrame, scope: str, fname: Path):
    """
    Plot recall against k for every experiment that reports it.

    Args:
        df: Comparison DataFrame
        scope: "all" or "answerable"
        fname: Output PNG path
    """
    ks = [1, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = False
    for _, row in df.iterrows():
        ys = [row.get(f"{scope}.recall@{k}_mean") for k in ks]
        if any(pd.notna(y) for y in ys):
            ax.plot(ks, ys, marker="o", label=row["experiment"])
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("k (retrieved passages)")
    ax.set_ylabel(f"Recall@k ({scope})")
    ax.set_title(f"Retrieval recall vs k ({scope})")
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def per_type_breakdown(root: Path, experiment: str, scope: str) -> pd.DataFrame:
    """
    Mean metrics grouped by question type for one experiment.

    Args:
        root: results/experiments directory
        experiment: Experiment name
        scope: "all" or "answerable"

    Returns:
        DataFrame indexed by question type, empty if the file is missing
    """
    path = root / experiment / f"detailed_{scope}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "question_type" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ["recall@5", "mrr", "rouge_l", "semantic_similarity",
                        "citation_f1"] if c in df.columns]
    return df.groupby("question_type")[cols].mean().round(3)


def plot_error_by_type(root: Path, experiment: str, fname: Path):
    """
    Heatmap of mean metrics by question type for the baseline run.

    Args:
        root: results/experiments directory
        experiment: Experiment name
        fname: Output PNG path
    """
    tbl = per_type_breakdown(root, experiment, "answerable")
    if tbl.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(tbl, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"{experiment}: metrics by question type (answerable)")
    plt.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, cols: List[str]) -> str:
    """
    Render selected columns of a DataFrame as a GitHub Markdown table.

    Args:
        df: Source DataFrame
        cols: Columns to include, in order

    Returns:
        Markdown table string
    """
    cols = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) and pd.notna(v) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def main():
    parser = argparse.ArgumentParser(description="Generate report figures and tables")
    parser.add_argument("--root", default="results/experiments")
    parser.add_argument("--figures", default="results/figures")
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()

    root = Path(args.root)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_comparison(root)

    # Figures, per ablation axis and scope.
    for scope in ("all", "answerable"):
        plot_axis(df, "chunk_size", f"{scope}.recall@5_mean", fig_dir / f"chunk_recall_{scope}.png",
                  f"Chunk strategy vs Recall@5 ({scope})")
        plot_axis(df, "embedding_model", f"{scope}.recall@5_mean",
                  fig_dir / f"embedding_recall_{scope}.png",
                  f"Embedding model vs Recall@5 ({scope})")
        plot_axis(df, "generation_model", f"{scope}.rouge_l_mean",
                  fig_dir / f"generation_rouge_{scope}.png",
                  f"Generation model vs ROUGE-L ({scope})")
        plot_recall_curve(df, scope, fig_dir / f"recall_curve_{scope}.png")

    plot_error_by_type(root, args.baseline, fig_dir / "baseline_by_type.png")

    # Written results summary. The answerable count is read from the runs rather than
    # hardcoded, so it stays correct if the corpus or the answerable rule changes.
    n_ans = int(df["n_answerable"].max()) if "n_answerable" in df.columns else 0
    n_all = int(df["n_all"].max()) if "n_all" in df.columns else 0
    lines = ["# Results summary", "",
             "Generated from `results/experiments/`. Every metric is shown for the answerable",
             f"subset ({n_ans} questions whose source document yields usable text) and, where",
             f"noted, for all {n_all} questions.", ""]

    lines += ["## Retrieval across configurations (answerable subset)", ""]
    retr_cols = ["experiment", "axis", "answerable.recall@1_mean", "answerable.recall@5_mean",
                 "answerable.recall@10_mean", "answerable.mrr_mean"]
    lines += [markdown_table(df, retr_cols), ""]

    gen = df[df["generation_model"].notna()]
    if not gen.empty:
        lines += ["## Generation and citations (answerable subset)", ""]
        gen_cols = ["experiment", "generation_model", "answerable.rouge_l_mean",
                    "answerable.semantic_similarity_mean", "answerable.citation_f1_mean",
                    "abstained"]
        lines += [markdown_table(gen, gen_cols), ""]

    lines += ["## Answerable vs all (retrieval Recall@5)", ""]
    delta_cols = ["experiment", "all.recall@5_mean", "answerable.recall@5_mean"]
    lines += [markdown_table(df, delta_cols), ""]

    tbl = per_type_breakdown(root, args.baseline, "answerable")
    if not tbl.empty:
        lines += [f"## Baseline ({args.baseline}) by question type (answerable)", "",
                  tbl.to_markdown(), ""]

    out = root.parent / "results_summary.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    print(f"Wrote figures to {fig_dir}/")
    for p in sorted(fig_dir.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
