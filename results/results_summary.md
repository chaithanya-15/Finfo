# Results summary

Generated from `results/experiments/`. Metrics are on the 114-question
answerable subset (documents that yield usable text); the full set is 150.

## Retrieval (answerable subset)

| config | axis | R@1 | R@5 | R@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| rerank_filtered | reranking | 0.249 | 0.516 | 0.516 | 0.360 |
| filtered_meta | retrieval_filter | 0.259 | 0.509 | 0.509 | 0.358 |
| chunk_structure | chunk_size | 0.244 | 0.475 | 0.475 | 0.333 |
| rerank | reranking | 0.235 | 0.472 | 0.472 | 0.329 |
| baseline | reference | 0.232 | 0.456 | 0.456 | 0.323 |
| gen_gemma | generation_model | 0.232 | 0.456 | 0.456 | 0.323 |
| k_10 | retrieval_k | 0.232 | 0.456 | 0.504 | 0.329 |
| prompt_derive | prompting | 0.232 | 0.456 | 0.456 | 0.323 |
| k_3 | retrieval_k | 0.232 | 0.409 | 0.409 | 0.311 |
| embed_minilm | embedding_model | 0.149 | 0.395 | 0.395 | 0.243 |
| hybrid | retrieval_filter | 0.124 | 0.379 | 0.379 | 0.218 |
| chunk_256 | chunk_size | 0.161 | 0.282 | 0.282 | 0.203 |

R@10 equals R@5 for every run that retrieves five passages, since there is no
sixth to tenth result to find the evidence in. Only `k_10` retrieves ten, so it
is the only row where the two columns can differ.

## Generation and citations (answerable subset)

| config | model | ROUGE-L | semantic | cit F1 | abstained |
| --- | --- | --- | --- | --- | --- |
| baseline | Qwen3.5-4B | 0.100 | 0.329 | 0.213 | 102.000 |
| chunk_256 | Qwen3.5-4B | 0.101 | 0.306 | 0.192 | 105.000 |
| chunk_structure | Qwen3.5-4B | 0.102 | 0.321 | 0.192 | 108.000 |
| filtered_meta | Qwen3.5-4B | 0.105 | 0.355 | 0.218 | 95.000 |
| gen_gemma | gemma-4-12B | 0.096 | 0.253 | 0.143 | 112.000 |
| prompt_derive | Qwen3.5-4B | 0.084 | 0.329 | 0.280 | 100.000 |
| rerank_filtered | Qwen3.5-4B | 0.116 | 0.380 | 0.211 | 89.000 |

## Answerable versus all, Recall@5

| config | R@5 (all 150) | R@5 (answerable) |
| --- | --- | --- |
| baseline | 0.360 | 0.456 |
| chunk_256 | 0.214 | 0.282 |
| chunk_structure | 0.374 | 0.475 |
| embed_minilm | 0.300 | 0.395 |
| filtered_meta | 0.400 | 0.509 |
| gen_gemma | 0.360 | 0.456 |
| hybrid | 0.294 | 0.379 |
| k_10 | 0.360 | 0.456 |
| k_3 | 0.324 | 0.409 |
| prompt_derive | 0.360 | 0.456 |
| rerank | 0.376 | 0.472 |
| rerank_filtered | 0.409 | 0.516 |

## Reference run by question type (answerable)

| question_type     |   recall@5 |   mrr |   rouge_l |   semantic_similarity |   citation_f1 |
|:------------------|-----------:|------:|----------:|----------------------:|--------------:|
| domain-relevant   |      0.357 | 0.207 |     0.154 |                 0.455 |         0.312 |
| metrics-generated |      0.347 | 0.295 |     0.001 |                 0.204 |         0.103 |
| novel-generated   |      0.681 | 0.485 |     0.135 |                 0.307 |         0.209 |
