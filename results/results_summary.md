# Results summary

Generated from `results/experiments/`. Every metric is shown for the answerable
subset (114 questions whose source document yields usable text) and, where
noted, for all 150 questions.

## Retrieval across configurations (answerable subset)

| experiment | axis | answerable.recall@1_mean | answerable.recall@5_mean | answerable.recall@10_mean | answerable.mrr_mean |
| --- | --- | --- | --- | --- | --- |
| baseline | reference | 0.232 | 0.456 | 0.456 | 0.323 |
| chunk_256 | chunk_size | 0.161 | 0.282 | 0.282 | 0.203 |
| chunk_structure | chunk_size | 0.244 | 0.475 | 0.475 | 0.333 |
| embed_minilm | embedding_model | 0.149 | 0.395 | 0.395 | 0.243 |
| gen_gemma | generation_model | 0.232 | 0.456 | 0.456 | 0.323 |
| k_10 | retrieval_k | 0.232 | 0.456 | 0.504 | 0.329 |
| k_3 | retrieval_k | 0.232 | 0.409 | 0.409 | 0.311 |

## Generation and citations (answerable subset)

| experiment | generation_model | answerable.rouge_l_mean | answerable.semantic_similarity_mean | answerable.citation_f1_mean | abstained |
| --- | --- | --- | --- | --- | --- |
| baseline | Qwen3.5-4B | 0.100 | 0.329 | 0.080 | 99.000 |
| chunk_256 | Qwen3.5-4B | 0.101 | 0.306 | 0.113 | 101.000 |
| chunk_structure | Qwen3.5-4B | 0.102 | 0.321 | 0.070 | 102.000 |
| gen_gemma | gemma-4-12B | 0.096 | 0.253 | 0.077 | 112.000 |

## Answerable vs all (retrieval Recall@5)

| experiment | all.recall@5_mean | answerable.recall@5_mean |
| --- | --- | --- |
| baseline | 0.360 | 0.456 |
| chunk_256 | 0.214 | 0.282 |
| chunk_structure | 0.374 | 0.475 |
| embed_minilm | 0.300 | 0.395 |
| gen_gemma | 0.360 | 0.456 |
| k_10 | 0.360 | 0.456 |
| k_3 | 0.324 | 0.409 |

## Baseline (baseline) by question type (answerable)

| question_type     |   recall@5 |   mrr |   rouge_l |   semantic_similarity |   citation_f1 |
|:------------------|-----------:|------:|----------:|----------------------:|--------------:|
| domain-relevant   |      0.357 | 0.207 |     0.154 |                 0.455 |         0.101 |
| metrics-generated |      0.347 | 0.295 |     0.001 |                 0.204 |         0.06  |
| novel-generated   |      0.681 | 0.485 |     0.135 |                 0.307 |         0.076 |
