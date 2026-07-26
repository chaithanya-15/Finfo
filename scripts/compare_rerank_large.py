#!/usr/bin/env python3
"""
Compare rerank_large_filtered against committed rerank_filtered:
- numeric_agreement_mean
- refusal rate (abstained rate)
- McNemar test p-value on numeric agreement
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from src.evaluation.evaluate import GenerationEvaluator

gen_eval = GenerationEvaluator()

# 1. Load detailed CSVs
df_base = pd.read_csv("results/experiments/rerank_filtered/detailed_answerable.csv")
df_large = pd.read_csv("results/experiments_rerank_large/rerank_large_filtered/detailed_answerable.csv")

# Compute numeric_agreement for base if missing
if "numeric_agreement" not in df_base.columns:
    df_base["numeric_agreement"] = [
        gen_eval.calculate_numeric_agreement(str(row["generated_answer"]), str(row["ground_truth_answer"]))
        for _, row in df_base.iterrows()
    ]

if "numeric_agreement" not in df_large.columns:
    df_large["numeric_agreement"] = [
        gen_eval.calculate_numeric_agreement(str(row["generated_answer"]), str(row["ground_truth_answer"]))
        for _, row in df_large.iterrows()
    ]

# Aggregated metrics
valid_base = [v for v in df_base["numeric_agreement"] if v is not None and not np.isnan(v)]
valid_large = [v for v in df_large["numeric_agreement"] if v is not None and not np.isnan(v)]

mean_num_base = float(np.mean(valid_base)) if valid_base else 0.0
mean_num_large = float(np.mean(valid_large)) if valid_large else 0.0

r5_base = float(df_base["recall@5"].mean())
r5_large = float(df_large["recall@5"].mean())

f1_base = float(df_base["citation_f1"].mean())
f1_large = float(df_large["citation_f1"].mean())

print("=== AGGREGATED METRICS COMPARISON (114 Answerable Questions) ===")
print(f"Base  (bge-reranker-base):  numeric_agreement = {mean_num_base:.4f} ({int(sum(valid_base))}/{len(valid_base)}) | R@5 = {r5_base:.4f} | Citation F1 = {f1_base:.4f}")
print(f"Large (bge-reranker-large): numeric_agreement = {mean_num_large:.4f} ({int(sum(valid_large))}/{len(valid_large)}) | R@5 = {r5_large:.4f} | Citation F1 = {f1_large:.4f}")

# Refusal rate (abstained responses)
def get_refusal_rate(df):
    if "abstained" in df.columns:
        return df["abstained"].mean()
    refusal_phrases = ["i am sorry", "i cannot answer", "information not provided", "does not contain", "cannot be answered"]
    refusals = df["generated_answer"].astype(str).str.lower().apply(lambda a: any(p in a for p in refusal_phrases))
    return refusals.mean()

ref_base = get_refusal_rate(df_base)
ref_large = get_refusal_rate(df_large)
print(f"Refusal Rate: Base = {ref_base:.4f} ({int(round(ref_base*len(df_base)))}/{len(df_base)}) | Large = {ref_large:.4f} ({int(round(ref_large*len(df_large)))}/{len(df_large)})")

# McNemar Test on numeric_agreement
valid_mask = df_base["numeric_agreement"].notna() & df_large["numeric_agreement"].notna()
b_agree = (df_base.loc[valid_mask, "numeric_agreement"] >= 0.9).astype(int).values
l_agree = (df_large.loc[valid_mask, "numeric_agreement"] >= 0.9).astype(int).values

n00 = int(np.sum((b_agree == 0) & (l_agree == 0)))
n01 = int(np.sum((b_agree == 0) & (l_agree == 1))) # Base fail, Large success
n10 = int(np.sum((b_agree == 1) & (l_agree == 0))) # Base success, Large fail
n11 = int(np.sum((b_agree == 1) & (l_agree == 1)))

# Exact binomial test on discordant pairs
b = n01
c = n10
n_discordant = b + c

if n_discordant > 0:
    res = stats.binomtest(b, n=n_discordant, p=0.5)
    p_val = res.pvalue
else:
    p_val = 1.0

print("\n=== McNEMAR TEST ON NUMERIC AGREEMENT ===")
print(f"Contingency Table (N={len(b_agree)}):")
print(f"                Large Fail (0)   Large Success (1)")
print(f"  Base Fail (0)        {n00:14d}      {n01:17d}")
print(f"  Base Succ (1)        {n10:14d}      {n11:17d}")
print(f"Discordant pairs: Base=0/Large=1: {n01}, Base=1/Large=0: {n10}")
print(f"McNemar exact p-value: {p_val:.6f}")
if p_val < 0.05:
    print("Result: STATISTICALLY SIGNIFICANT DIFFERENCE (p < 0.05)")
else:
    print("Result: NO STATISTICALLY SIGNIFICANT DIFFERENCE (p >= 0.05)")
