#!/usr/bin/env python3
"""
Turn the ablation sweep output into the figures and tables the report needs.

Reads results/experiments/comparison.csv and the per-run detailed CSVs, writes PNGs to
results/figures/ and a Markdown results section to results/results_summary.md.

The figures use the Okabe-Ito colourblind-safe palette, thin marks, and direct value labels
rather than colour-only encoding, so every panel is legible in greyscale and to colourblind
readers. Metrics are shown on the 114-question answerable subset unless a panel is explicitly
about the answerable-versus-all gap.

Run after run_experiments.py:
    python -m src.evaluation.make_figures
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Okabe-Ito palette (colourblind-safe). Assigned by role, in fixed order.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e2e2e0"
SURFACE = "#ffffff"
BLUE = "#0072B2"      # reference / primary
ORANGE = "#E69F00"    # second category
GREEN = "#009E73"     # third category / "good"
VERMILLION = "#D55E00"  # contrast / "miss"
SKY = "#56B4E9"       # fourth category
PURPLE = "#CC79A7"    # fifth category
GREY = "#7a7a7a"      # sixth category
AXIS_COLOR = {"chunk_size": ORANGE, "embedding_model": GREEN,
              "retrieval_k": SKY, "generation_model": PURPLE, "reference": BLUE,
              "retrieval_filter": VERMILLION, "reranking": "#117733",
              "prompting": GREY}

RETRIEVAL_METRICS = ["recall@1", "recall@5", "recall@10", "mrr"]


def set_style():
    """Apply a clean, low-chartjunk matplotlib style shared by every figure."""
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })


def _despine(ax, keep_left=True):
    """Remove top and right spines, and optionally the left one, for a lighter frame."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)


def fig_coverage_funnel(fig_dir: Path):
    """
    Corpus-coverage funnel: catalogued documents down to answerable questions.

    Makes the central data limitation visible in one panel, the decay from the full manifest to
    the questions the system can actually answer.
    """
    cache = Path("data/extracted_text/pdfplumber")
    catalogued = len({json.loads(l)["doc_name"]
                      for l in open("data/financebench_document_information.jsonl")})
    downloaded = len(list(Path("data/raw_pdfs").glob("*.pdf")))
    extracted = sum(1 for p in cache.glob("*.json")
                    if json.load(open(p)).get("text", "").strip()) if cache.exists() else 0

    stages = [("Catalogued documents", catalogued, MUTED),
              ("Downloaded PDFs", downloaded, SKY),
              ("Extracted to usable text", extracted, BLUE)]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [s[2] for s in stages]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(stages))[::-1]
    ax.barh(y, values, color=colors, height=0.62, zorder=3)
    for yi, v, total in zip(y, values, [catalogued] * len(values)):
        ax.text(v + catalogued * 0.01, yi, f"{v}  ({v / total:.0%})",
                va="center", ha="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, catalogued * 1.18)
    ax.set_xlabel("documents")
    ax.set_title("Corpus coverage: 78 dead links and corrupt files thin the manifest")
    ax.grid(axis="y", visible=False)
    _despine(ax, keep_left=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage_funnel.png", bbox_inches="tight")
    plt.close(fig)


def fig_retrieval_ablation(df: pd.DataFrame, fig_dir: Path):
    """
    Recall@5 for every configuration, coloured by the ablation axis it varies.

    One panel carries the whole retrieval story: which single change helps or hurts, with the
    reference configuration marked as the point of comparison.
    """
    order = ["chunk_256", "embed_minilm", "k_3", "baseline", "prompt_derive", "k_10",
             "chunk_structure", "hybrid", "filtered_meta", "rerank_filtered"]
    sub = df.set_index("experiment").reindex([o for o in order if o in df["experiment"].values])
    labels = {"chunk_256": "256-tok chunks", "baseline": "512-tok (reference)",
              "chunk_structure": "structure-aware", "embed_minilm": "MiniLM embed",
              "k_3": "k=3", "k_10": "k=10", "filtered_meta": "company filter",
              "hybrid": "dense+BM25", "rerank_filtered": "filter+rerank",
              "prompt_derive": "derive prompt"}
    vals = sub["answerable.recall@5_mean"].values
    colors = [AXIS_COLOR.get(a, BLUE) for a in sub["axis"].values]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(sub))
    bars = ax.bar(x, vals, color=colors, width=0.66, zorder=3,
                  edgecolor=SURFACE, linewidth=1.5)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10.5, color=INK, fontweight="bold")
    ax.axhline(sub.loc["baseline", "answerable.recall@5_mean"], color=BLUE,
               ls="--", lw=1, alpha=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(i, i) for i in sub.index], rotation=18, ha="right")
    ax.set_ylabel("Recall@5 (answerable)")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("Retrieval: company filtering and reranking beat every chunking or k change")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    # legend by axis, placed over the short leftmost bar to avoid the taller bars' labels
    from matplotlib.patches import Patch
    seen = dict(zip(sub["axis"], colors))
    legend = [Patch(facecolor=AXIS_COLOR[a], label=a.replace("_", " "))
              for a in ["reference", "chunk_size", "embedding_model", "retrieval_k",
                        "retrieval_filter", "reranking", "prompting"] if a in seen]
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=9.5, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "retrieval_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig_recall_curves(df: pd.DataFrame, fig_dir: Path):
    """Recall@k for the reference, the best chunker, and k=10, showing depth versus rank."""
    ks = [1, 3, 5, 10]
    picks = [("baseline", "512-tok, k=5 (reference)", BLUE, "o"),
             ("chunk_structure", "structure-aware", GREEN, "s"),
             ("k_10", "k=10", SKY, "^")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    d = df.set_index("experiment")
    for name, label, color, marker in picks:
        if name not in d.index:
            continue
        ys = [d.loc[name, f"answerable.recall@{k}_mean"] for k in ks]
        ax.plot(ks, ys, color=color, lw=2, marker=marker, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
        ax.text(ks[-1] + 0.15, ys[-1], f"{ys[-1]:.3f}", va="center", ha="left",
                color=color, fontsize=10, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xlabel("k (retrieved passages)")
    ax.set_ylabel("Recall@k (answerable)")
    ax.set_xlim(0.5, 11.5)
    ax.set_ylim(0, None)
    ax.set_title("More passages add recall depth, not ranking quality")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "recall_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_answerable_gap(df: pd.DataFrame, fig_dir: Path):
    """Paired Recall@5 over all 150 questions versus the 114 answerable, per configuration."""
    order = ["baseline", "chunk_structure", "chunk_256", "embed_minilm", "k_3", "k_10"]
    sub = df.set_index("experiment").reindex([o for o in order if o in df["experiment"].values])
    labels = {"baseline": "reference", "chunk_structure": "structure", "chunk_256": "256-tok",
              "embed_minilm": "MiniLM", "k_3": "k=3", "k_10": "k=10"}
    all_v = sub["all.recall@5_mean"].values
    ans_v = sub["answerable.recall@5_mean"].values

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(sub))
    w = 0.38
    ax.bar(x - w / 2, all_v, width=w, color=MUTED, zorder=3, label="all 150 questions",
           edgecolor=SURFACE, linewidth=1.2)
    ax.bar(x + w / 2, ans_v, width=w, color=BLUE, zorder=3, label="114 answerable",
           edgecolor=SURFACE, linewidth=1.2)
    for xi, v in zip(x - w / 2, all_v):
        ax.text(xi, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=MUTED)
    for xi, v in zip(x + w / 2, ans_v):
        ax.text(xi, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                color=BLUE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(i, i) for i in sub.index])
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0, max(ans_v) * 1.25)
    ax.set_title("The missing documents, not the model, hold down the full-set scores")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "answerable_gap.png", bbox_inches="tight")
    plt.close(fig)


def fig_generation_models(df: pd.DataFrame, fig_dir: Path):
    """Qwen3.5-4B versus gemma-4-12B on generation and citation metrics, same retrieval."""
    d = df.set_index("experiment")
    if "gen_gemma" not in d.index or "baseline" not in d.index:
        return
    metrics = [("answerable.rouge_l_mean", "ROUGE-L"),
               ("answerable.semantic_similarity_mean", "semantic sim"),
               ("answerable.citation_f1_mean", "citation F1")]
    qwen = [d.loc["baseline", m] for m, _ in metrics]
    gemma = [d.loc["gen_gemma", m] for m, _ in metrics]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.3),
                                  gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(metrics))
    w = 0.38
    ax.bar(x - w / 2, qwen, width=w, color=BLUE, zorder=3, label="Qwen3.5-4B",
           edgecolor=SURFACE, linewidth=1.2)
    ax.bar(x + w / 2, gemma, width=w, color=ORANGE, zorder=3, label="gemma-4-12B",
           edgecolor=SURFACE, linewidth=1.2)
    for xi, v in zip(x - w / 2, qwen):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                color=BLUE, fontweight="bold")
    for xi, v in zip(x + w / 2, gemma):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("score (answerable)")
    ax.set_ylim(0, max(qwen + gemma) * 1.25)
    ax.set_title("A 3x larger generator does not answer better")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    _despine(ax)

    # abstention panel
    ab = [int(d.loc["baseline", "abstained"]), int(d.loc["gen_gemma", "abstained"])]
    ax2.bar([0, 1], ab, color=[BLUE, ORANGE], width=0.6, zorder=3,
            edgecolor=SURFACE, linewidth=1.2)
    for xi, v in zip([0, 1], ab):
        ax2.text(xi, v + 1, f"{v}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Qwen", "gemma"])
    ax2.set_ylabel("abstained (of 150)")
    ax2.set_ylim(0, 150)
    ax2.set_title("Abstentions", fontsize=11)
    ax2.grid(axis="x", visible=False)
    _despine(ax2)
    fig.tight_layout()
    fig.savefig(fig_dir / "generation_models.png", bbox_inches="tight")
    plt.close(fig)


def fig_failure_decomposition(root: Path, fig_dir: Path, experiment: str):
    """
    Where answerable questions are lost: retrieval miss versus generation miss.

    A stacked bar splitting the 114 answerable questions into evidence-never-retrieved,
    retrieved-but-answered-poorly, and retrieved-and-answered-well.
    """
    path = root / experiment / "detailed_answerable.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    n = len(df)
    retrieved = df["recall@5"] > 0
    good = retrieved & (df["rouge_l"] > 0.1)
    n_miss = int((~retrieved).sum())
    n_gen_miss = int((retrieved & ~good).sum())
    n_good = int(good.sum())

    segs = [("Retrieval miss\n(evidence never retrieved)", n_miss, VERMILLION),
            ("Generation miss\n(retrieved, weak answer)", n_gen_miss, ORANGE),
            ("Answered well\n(retrieved + ROUGE-L>0.1)", n_good, GREEN)]

    fig, ax = plt.subplots(figsize=(9, 2.9))
    left = 0
    for label, val, color in segs:
        ax.barh(0, val, left=left, color=color, height=0.5, zorder=3,
                edgecolor=SURFACE, linewidth=2)
        ax.text(left + val / 2, 0, f"{val}\n{val / n:.0%}", ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
        left += val
    ax.set_xlim(0, n)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel(f"answerable questions (n={n})")
    ax.set_title("Half the answerable questions are lost at retrieval, half of the rest at generation")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    # legend below
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=l.replace("\n", " ")) for l, _, c in segs]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "failure_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def fig_by_type(root: Path, fig_dir: Path, experiment: str):
    """Grouped bars of retrieval and generation quality by question type for the reference run."""
    tbl = per_type_breakdown(root, experiment, "answerable")
    if tbl.empty:
        return
    metrics = [("recall@5", "Recall@5", BLUE), ("mrr", "MRR", SKY),
               ("semantic_similarity", "semantic sim", GREEN), ("citation_f1", "citation F1", ORANGE)]
    metrics = [m for m in metrics if m[0] in tbl.columns]
    if not metrics:
        return
    types = list(tbl.index)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(types))
    w = 0.8 / len(metrics)
    for i, (col, label, color) in enumerate(metrics):
        offs = (i - (len(metrics) - 1) / 2) * w
        vals = tbl[col].values
        ax.bar(x + offs, vals, width=w * 0.92, color=color, zorder=3, label=label,
               edgecolor=SURFACE, linewidth=0.8)
        for xi, v in zip(x + offs, vals):
            ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("score (answerable)")
    ax.set_ylim(0, min(1.0, tbl[[m[0] for m in metrics]].values.max() * 1.25))
    ax.set_title("By question type: numbers-only answers depress lexical scores")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "by_question_type.png", bbox_inches="tight")
    plt.close(fig)


def per_type_breakdown(root: Path, experiment: str, scope: str) -> pd.DataFrame:
    """Mean metrics grouped by question type for one experiment, empty if unavailable."""
    path = root / experiment / f"detailed_{scope}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "question_type" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ["recall@5", "mrr", "rouge_l", "semantic_similarity", "citation_f1"]
            if c in df.columns]
    return df.groupby("question_type")[cols].mean().round(3)


def markdown_table(df: pd.DataFrame, cols, rename=None) -> str:
    """Render selected columns of a DataFrame as a GitHub Markdown table."""
    cols = [c for c in cols if c in df.columns]
    header_names = [rename.get(c, c) if rename else c for c in cols]
    header = "| " + " | ".join(header_names) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) and pd.notna(v)
                         else ("" if pd.isna(v) else str(v)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_summary(df: pd.DataFrame, root: Path):
    """Write results/results_summary.md with the retrieval, generation and coverage tables."""
    n_ans = int(df["n_answerable"].max()) if "n_answerable" in df.columns else 0
    n_all = int(df["n_all"].max()) if "n_all" in df.columns else 0
    lines = ["# Results summary", "",
             f"Generated from `results/experiments/`. Metrics are on the {n_ans}-question",
             f"answerable subset (documents that yield usable text); the full set is {n_all}.", ""]

    ren = {"experiment": "config", "axis": "axis",
           "answerable.recall@1_mean": "R@1", "answerable.recall@5_mean": "R@5",
           "answerable.recall@10_mean": "R@10", "answerable.mrr_mean": "MRR"}
    lines += ["## Retrieval (answerable subset)", "",
              markdown_table(df.sort_values("answerable.recall@5_mean", ascending=False),
                             list(ren)[:2] + ["answerable.recall@1_mean", "answerable.recall@5_mean",
                             "answerable.recall@10_mean", "answerable.mrr_mean"], ren), ""]

    gen = df[df["generation_model"].notna()]
    if not gen.empty:
        gren = {"experiment": "config", "generation_model": "model",
                "answerable.rouge_l_mean": "ROUGE-L",
                "answerable.semantic_similarity_mean": "semantic",
                "answerable.citation_f1_mean": "cit F1", "abstained": "abstained"}
        lines += ["## Generation and citations (answerable subset)", "",
                  markdown_table(gen, list(gren), gren), ""]

    gren2 = {"experiment": "config", "all.recall@5_mean": "R@5 (all 150)",
             "answerable.recall@5_mean": "R@5 (answerable)"}
    lines += ["## Answerable versus all, Recall@5", "",
              markdown_table(df, list(gren2), gren2), ""]

    tbl = per_type_breakdown(root, "baseline", "answerable")
    if not tbl.empty:
        lines += ["## Reference run by question type (answerable)", "", tbl.to_markdown(), ""]

    out = root.parent / "results_summary.md"
    out.write_text("\n".join(lines))
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate report figures and tables")
    parser.add_argument("--root", default="results/experiments")
    parser.add_argument("--figures", default="results/figures")
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()

    set_style()
    root = Path(args.root)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    comparison = root / "comparison.csv"
    if not comparison.exists():
        raise SystemExit(f"{comparison} not found. Run run_experiments.py first.")
    df = pd.read_csv(comparison)

    fig_coverage_funnel(fig_dir)
    fig_retrieval_ablation(df, fig_dir)
    fig_recall_curves(df, fig_dir)
    fig_answerable_gap(df, fig_dir)
    fig_generation_models(df, fig_dir)
    fig_failure_decomposition(root, fig_dir, args.baseline)
    fig_by_type(root, fig_dir, args.baseline)

    out = write_summary(df, root)
    print(f"Wrote {out}")
    print(f"Wrote figures to {fig_dir}/")
    for p in sorted(fig_dir.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
