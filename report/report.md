# Building and Evaluating a Retrieval-Augmented Generation System on FinanceBench

Karthik Reddy Changal, Chaithanya Anugu, Nithin Sujith Nair

## Abstract

We build and evaluate a retrieval-augmented generation (RAG) pipeline that answers questions about
SEC filings on free models that run locally, comparing fourteen configurations that vary chunking,
retrieval, generation, and prompting one factor at a time. On the 114 questions whose source
document is available, the best configuration restricts retrieval to the company named in the
question and reorders a deep pool with a cross-encoder, reaching Recall@5 0.52 against 0.46 for
pure dense search. The main finding is about measurement: Recall@5 credits nearly all of that gain
to the company filter and almost none to reranking, but on whether the system answers correctly the
ordering reverses, because reranking raises how often the right filing reaches the generator
without changing the token-overlap score Recall@5 uses. Reranking doubles the
correct answers, from 8 to 16 of the 84 questions with a numeric gold answer, while a BM25 hybrid,
a three-times-larger generator, and a refusal-reducing prompt all fail to help. Every metric is
reported over both the full 150-question set and the answerable subset, because a fifth of the
corpus no longer downloads, and we treat that coverage gap as a result.

## 1. Introduction

### 1.1 Problem statement

Financial filings are long and table-heavy; a 10-K runs past 100 pages, and the target fact
usually sits in one row of one table. Answering a question means finding the passage with the fact
and reading it correctly, and model knowledge does not help because the answer is specific to one
company and year. Retrieval-augmented generation (RAG) matches this: a retriever selects a few
relevant passages and a language model writes a cited answer from them.

### 1.2 Related work

RAG pairs a dense retriever with a generator so a model answers from an external corpus rather
than its parameters (Lewis et al., 2020). We use FinanceBench (Islam et al., 2023), which gives
each question a gold answer and the exact evidence span it draws on, so both answer quality and
whether retrieval found the right passage are measurable.

Retrieval is conventionally scored by whether the gold evidence appears among the retrieved
passages (Recall@k, MRR), generation against the reference answer (ROUGE, semantic similarity, or
an LLM judge such as RAGAS). We avoid an LLM judge, which adds a second large model and ties scores
to its quirks, and use lexical and semantic overlap plus a citation check.

### 1.3 Contributions

We ablate eight design decisions one variable at a time and report two findings. Retrieval, not
generation, is the binding constraint: the best configuration finds the gold evidence for barely
half the answerable questions. And the retrieval metric is an unreliable guide to which changes
help, ranking our two interventions in the opposite order to their effect on answer quality, a
caution for anyone tuning against Recall@k alone.

## 2. Corpus description

We use FinanceBench, distributed as two JSONL files linked by `doc_name`. The document
catalogue lists 360 filings; the question file holds 150 questions with gold answers and the
evidence text each answer is drawn from.

The filings span several document types (269 10-Ks, 30 8-Ks, 29 earnings releases, 27 10-Qs,
6 annual reports) across eight GICS sectors, led by Information Technology (80), Consumer
Discretionary (77), and Consumer Staples (63). The 150 questions divide evenly into three types,
metrics-generated, domain-relevant, and novel-generated (50 each), and 43 need numerical
reasoning. Answers range from single dollar figures ("$1577.00") to short sentences.

**Corpus coverage.** Only 282 of the 360 catalogued PDFs still download; the other 78 URLs
return timeouts, 404s, or 403s from the filers' investor-relations hosts. A further set are
HTML error pages saved with a .pdf extension (such as several AMD 10-Ks) or near-empty stubs, so extraction succeeds
for 263 documents and 114 of the 150 questions reference a document that yields usable text. We
therefore report each metric twice, over all 150 questions and over the 114-question answerable
subset: the first describes the system against the corpus as it exists today, the second isolates
retrieval and generation quality from missing inputs.

![Figure 1](../results/figures/coverage_funnel.png)

**Figure 1. Corpus coverage.** Of 360 catalogued filings, 282 still download and 263 extract
to usable text, leaving 114 of the 150 questions answerable from the local corpus.

## 3. Methodology

### 3.1 System architecture

The pipeline has four stages: document processing (`src/data_processing`), retrieval
(`src/retrieval`), generation (`src/generation`), and evaluation (`src/evaluation`), tied
together by `run_experiments.py`. Each stage writes its output to disk so later stages and
repeated runs reuse it.

### 3.2 Document processing

PDFs are downloaded with SEC-compliant headers and retry logic (`download_pdfs.py`). Text is
extracted with pdfplumber, which recovers both body text and table content. Extraction is cached
per document so the three chunk strategies do not each re-parse the same PDF, turning roughly four
and a half hours of repeated extraction into a single ninety-minute pass. Extracted text is then
cleaned to normalise whitespace and de-hyphenate line breaks.

### 3.3 Chunking strategies

We compare three strategies: fixed windows of 256 tokens with 32-token overlap, fixed windows
of 512 tokens with 64-token overlap, and a structure-aware strategy that packs whole sentences
up to a 512-token limit so a chunk never ends mid-sentence. Each chunk carries its document name,
company, sector, period, and a chunk index used to build citation tags.

### 3.4 Embedding models

The baseline embedding model is BAAI/bge-base-en-v1.5 (768 dimensions), compared against the
smaller, faster all-MiniLM-L6-v2 (384 dimensions). Embeddings are normalised and indexed for
inner-product search, which for unit vectors is cosine similarity.

### 3.5 Retrieval

Chunks are embedded and stored in a FAISS flat inner-product index, one index per embedding
model and chunk strategy. At query time the question is embedded with the same model and the
top-k chunks are returned. We vary k over 3, 5, and 10. Beyond plain dense search we test three
strategies (Section 5.2): restricting candidates to the company named in the question, fusing the
dense ranking with a BM25 lexical ranking, and reranking. The reranker retrieves a pool of 50 and
reorders it with bge-reranker-base, a cross-encoder that scores a question and passage together
rather than embedding them separately, which is slower but better at telling whether a passage
answers the question or only resembles it. It runs locally like everything else.

### 3.6 Generation

Answers are generated by Qwen3.5-4B, run as 4-bit GGUF weights through llama.cpp with Metal on
Apple silicon. (The brief's suggested Llama-3.1-8B with bitsandbytes is not usable here, since
bitsandbytes has no macOS build; llama.cpp gives the equivalent quantised local inference.) The
generator receives the retrieved passages, each prefixed with its citation tag, and is instructed
to answer only from them, cite each claim, and say when the passages do not contain the answer. As
a second generator we compare gemma-4-12B, a larger model from a different family. Decoding uses
temperature 0.1.

### 3.7 Evaluation metrics

Retrieval is scored with Recall@k, Precision@k, and mean reciprocal rank; a retrieved chunk
counts as matching the gold evidence when it covers at least half of the evidence's content
words (Section 7.3). Precision@k divides by k rather than by the number of passages returned, so
a run that retrieves five passages cannot score above 0.5 at k = 10. Generation is scored with
ROUGE-L, embedding-based semantic similarity, and exact match against the gold answer (exact match is 0.0 across all configurations by design because reference answers are concise figures while generated responses are complete sentences; `numeric_agreement` is introduced to evaluate numerical factual accuracy). Citation
quality is precision, recall, and F1 over the `[DOCNAME_c<index>]` tags, each checked against the
passages actually retrieved.

We add one further generation metric. FinanceBench answers are frequently bare figures like
"$1577.00", so a correct answer written as a sentence scores near zero on word overlap: ROUGE-L is
0.001 on metrics-generated questions whatever the model replies. Numeric agreement counts an answer
correct when a gold figure appears in it within 0.5%, allowing a thousand-fold difference since
filings quote millions and answers sometimes restate in billions. Fiscal years are stripped from
both sides, since otherwise "FY2023" matches "FY2023" and scores a wrong answer correct. It is
undefined for the 30 answerable questions whose gold answer is prose, leaving 84 scorable.

Two controls check it measures answers rather than noise. Refusals score zero in every
configuration. And scoring each answer against a different question's gold figure gives 0.004 to
0.015 across runs, against 0.24 to 0.34 for the true pairing; this matters because a wordier
configuration quotes more figures and so gets more chances to match by luck, yet the derive-prompt
run, which writes twice as many figures per answer as the reference, still scores 0.015 at chance.

The brief suggests RAGAS, TruLens, or an LLM-as-a-judge as evaluation frameworks. We use the
metrics above instead, for three reasons: they are deterministic and need no second large model to
audit, they score the two properties this task actually turns on (evidence overlap for retrieval,
figure agreement for answers), and they keep the whole pipeline free and local as the brief
requires. RAGAS's headline generation metrics are themselves LLM-judge-based, which is the
dependency we are avoiding; its embedding-only metrics, context precision and recall, would
largely duplicate the Recall@k and evidence-overlap scoring we already report.

## 4. Experimental setup

### 4.1 Implementation

The pipeline is Python 3.12: sentence-transformers and faiss-cpu for retrieval, llama-cpp-python
built with Metal for generation, rouge-score and rapidfuzz for evaluation. Embedding runs on the
Apple GPU through MPS. Hardware is an Apple M4 Pro with 24 GB unified memory, and versions are
pinned in `requirements.txt`.

### 4.2 Ablation design

The reference is 512-token chunks, bge-base embeddings, k=5, Qwen3.5-4B. Each other run changes
one variable: chunk strategy (256-token and structure-aware), k (3 and 10), embedding model
(MiniLM), generator (gemma-4-12B), retrieval strategy (company filtering, a dense + BM25 hybrid,
and reranking with and without that filter), and the generator's instructions (a prompt permitting
derived figures). The matrix is `configs/experiments.yaml`, and indexes are cached across runs so
each embedding-and-strategy pair is built once.

The best configuration changes two things at once, so two further arms separate them: reranking is
run over both a company-filtered and an unfiltered pool, and the filter is run alone with
generation enabled (Table 2). Of the fourteen arms, seven generate answers; the other seven vary
retrieval only, which keeps the sweep affordable since generation dominates its runtime.

## 5. Results

Full per-configuration tables are in `results/results_summary.md`. Retrieval numbers are on the
114-question answerable subset unless stated.

### 5.1 Overall performance

The reference configuration retrieves the gold evidence into its top 5 for 45.6% of answerable
questions, with an MRR of 0.323. Over all 150 questions, including the 36 whose document is
missing or corrupt, Recall@5 falls to 0.360: those questions can never be retrieved and enter the
full-set average as zeros. We carry both numbers throughout; Figure 2 shows the gap for every
configuration.

![Figure 2](../results/figures/answerable_gap.png)

**Figure 2. Recall@5 over all 150 questions against the 114 answerable.** The gap is roughly
constant across configurations, as expected if it comes from missing data rather than any one
design choice.

Table 1 gives retrieval quality for all fourteen configurations on the answerable subset,
ordered by Recall@5.

**Table 1. Retrieval metrics on the 114-question answerable subset.**

| Configuration | Ablation axis | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| company filter + reranking | reranking | 0.249 | **0.516** | 0.516 | **0.360** |
| company filter | retrieval filter | 0.259 | 0.509 | 0.509 | 0.358 |
| structure-aware chunks | chunk size | 0.244 | 0.475 | 0.475 | 0.333 |
| reranking, no company filter | reranking | 0.235 | 0.472 | 0.472 | 0.329 |
| 128-token overlap | chunk overlap | 0.262 | 0.462 | 0.462 | 0.335 |
| 512-token chunks (reference) | reference | 0.232 | 0.456 | 0.456 | 0.323 |
| k = 10 | retrieval k | 0.232 | 0.456 | 0.504 | 0.329 |
| derive prompt | prompting | 0.232 | 0.456 | 0.456 | 0.323 |
| gemma-4-12B generator | generation model | 0.232 | 0.456 | 0.456 | 0.323 |
| 32-token overlap | chunk overlap | 0.240 | 0.439 | 0.439 | 0.307 |
| k = 3 | retrieval k | 0.232 | 0.409 | 0.409 | 0.311 |
| MiniLM embeddings | embedding model | 0.149 | 0.395 | 0.395 | 0.243 |
| dense + BM25 hybrid | retrieval filter | 0.124 | 0.379 | 0.379 | 0.218 |
| 256-token chunks | chunk size | 0.161 | 0.282 | 0.282 | 0.203 |

Two rows match the reference by construction: the derive-prompt and gemma runs change only the
generator's instructions and the generator itself, so neither touches retrieval. Recall@10
likewise equals Recall@5 everywhere except k = 10, since a run that retrieves five passages has
no sixth to tenth result for the evidence to appear in.

Generation is weaker and limited by retrieval. The reference system reaches ROUGE-L 0.10 and
semantic similarity 0.33 on the answerable subset and declines to answer on 102 of 150 questions.
That abstention rate follows from retrieval: when the evidence does not reach the model, the
prompt tells it to refuse, so a retrieval miss becomes an abstention rather than a wrong figure.

### 5.2 Ablation analysis

![Figure 3](../results/figures/retrieval_ablation.png)

**Figure 3. Recall@5 for all fourteen configurations, coloured by ablation axis.** The dashed line
marks the reference. Company filtering followed by reranking gives the highest score;
structure-aware chunking is the best of the content-only changes. The derive-prompt and gemma
runs sit exactly on the reference line because neither changes retrieval.

#### Chunk size, structure, and overlap

Chunking parameters have a major effect on retrieval. Cutting the fixed window from 512 to
256 tokens is clearly harmful: Recall@5 drops from 0.456 to 0.282 and MRR from 0.323 to 0.203,
because smaller chunks split a table or paragraph across more pieces and a single retrieved chunk
covers less of the evidence. Structure-aware chunking, which packs whole sentences to a 512-token
limit so a chunk never ends mid-sentence, is nominally best at Recall@5 0.475 and MRR 0.333.

Varying the sliding overlap at a fixed 512-token chunk size reveals a monotonic relationship with retrieval quality. Reducing overlap from 64 to 32 tokens lowers Recall@5 from 0.456 to 0.439 and MRR from 0.323 to 0.307 (with total chunks falling from 62,339 to 58,214). Increasing overlap to 128 tokens improves Recall@5 to 0.462 and MRR to 0.335 (index size growing to 72,661 chunks). Larger overlap windows ensure financial tables and multi-sentence figures spanning chunk boundaries remain intact in at least one adjacent chunk, raising retrieval recall.

This margin has a caveat. Both 512-token strategies produce chunks longer than the embedder can
read, and unequally: 47% of reference chunks are truncated against 23% of structure-aware ones
(Section 7.3). A 0.019 gap is small enough that the difference in indexed
text could account for it, so we do not claim structure-aware chunking is better on this evidence.
The 256-token result is unaffected, since those chunks fit the limit.

#### Embedding model

The larger embedding model helps. Swapping bge-base for all-MiniLM-L6-v2 on the same chunks lowers
Recall@5 from 0.456 to 0.395 and MRR from 0.323 to 0.243. MiniLM is faster and a third of the
dimensionality, but on terse questions where the answer is one row of a table, bge-base finds that
row more often.

#### Retrieval k

Increasing k buys recall depth, not rank quality. From k=5 to k=10, Recall@5 is unchanged at
0.456 (the same top-5) but Recall@10 rises to 0.504, so a handful of questions have their evidence
in positions 6 through 10; MRR barely moves (0.323 to 0.329). At k=3 recall drops to 0.409. For
the generator k is a trade-off: more passages raise the chance the evidence is present but also
add distractors. Figure 4 shows the recall-versus-k curves.

![Figure 4](../results/figures/recall_curves.png)

**Figure 4. Recall@k for the reference, the structure-aware chunker, and k=10.** Raising k
lifts the tail of the curve (Recall@10) without changing where the first correct passage lands.

#### Retrieval filtering and hybrid search

The content-only changes above move Recall@5 within a narrow band (0.28 to 0.48), consistent with
dense similarity over the whole corpus being near its ceiling. Two changes attack that ceiling.
Each question names its target company and every chunk carries that company in its metadata, so we
restrict the search to the company named in the question (inferred from the question text, never
the gold answer). This company filter beats every chunking, k, and embedding change: Recall@5
0.509, MRR 0.358.

Its headroom is smaller than it looks. A name match resolves the company in about 85% of questions
and misses fall back to unfiltered search, which invites a better entity extractor. But filtering
to the gold document itself, which no extractor can beat, reaches only Recall@5 0.567. Almost all
of what filtering can deliver is already delivered, and the remaining gap is a ranking problem.

Hybrid retrieval, fusing the dense ranking with BM25 by reciprocal rank fusion, does the opposite:
Recall@5 falls to 0.379. BM25 rewards shared boilerplate vocabulary, and the natural-language
question terms rarely match the tabular numbers holding the answer, so the lexical signal adds
noise, so hybrid retrieval hurts on this corpus.

#### Reranking

Under the company filter, recall keeps rising past k=10: 0.702 at k=20 and 0.801 at k=50. The
evidence is usually in a deep pool but ranked too low, so we retrieve a company-filtered pool of 50
and reorder it with bge-reranker-base, a local cross-encoder that scores each question-and-passage
pair jointly.

By Recall@5 the effect is small: 0.509 to 0.516, against an available 0.29. Reranking without the
filter behaves the same, 0.456 to 0.472, so the small movement is a property of the metric, not
the filtered pool. By answer quality it is the largest single improvement in the study: numeric
agreement rises from 0.095 to 0.190, doubling the correct answers from 8 to 16 of 84, significant
by a McNemar exact test (p = 0.039), and abstentions fall from 102 to 89 of 150.

Running the company filter with generation separates the two changes and inverts the retrieval
picture:

**Table 2. Decomposing the gain, answerable subset.** Answer quality is numeric agreement over
the 84 questions with a numeric gold answer; each step is tested against the one above it.

| Step | Recall@5 | Answer quality | Questions gained / lost | McNemar p |
| --- | --- | --- | --- | --- |
| reference | 0.456 | 0.095 (8 of 84) | | |
| + company filter | 0.509 (+0.053) | 0.107 (9 of 84) | 1 / 0 | 1.00 |
| + cross-encoder reranking | 0.516 (+0.007) | 0.190 (16 of 84) | 9 / 2 | 0.065 |
| reference to both | +0.060 | +0.095 | 10 / 2 | **0.039** |

The two orderings are opposite. Recall@5 assigns almost the whole retrieval gain to the company
filter, while on answer quality the filter moves one question and reranking moves nine. We state
the direction of that inversion rather than a ratio, because at this sample size the individual
steps do not separate from noise: only the combined change is significant, the isolated reranking
step is marginal (p = 0.065), and the filter's one-question effect is indistinguishable from zero.
The claim the decomposition supports is that essentially all of the answer-quality gain traces to
reranking, not that the filter contributes nothing.

This does not depend on the small counts. Reranking raises the share of questions whose correct
filing reaches the top 5 from 0.605 to 0.746, while barely changing the token-overlap score
Recall@5 uses. The generator needs the right filing, and Recall@5 does not register that change.

Two examples show this. Asked for 3M's FY2018 capital expenditure (gold $1,577m), the
filtered system answered "$1,493 million" from a 2015 filing and the reranked system "$1,577
million" from the 2020 filing; both scored Recall@5 of 0.00. Asked for Amazon's FY2019 net income
(gold $11,588m), both scored Recall@5 of 1.00, but the filter answered "$33,364 million" and the
reranker "$11,588 million". Of the nine questions reranking fixed, Recall@5 improved on only
three. It is not uniformly better either: two questions the filter had answered correctly were
refused after reranking.

#### Prompting and abstention

The system refuses on 102 of 150 questions, and on 40% of questions whose evidence was retrieved,
which points at instructions rather than evidence for some. Asked for a capital-expenditure
figure, the model replied that the excerpts did not contain it "in a cash flow statement format",
a refusal about presentation rather than absence. We therefore tested a prompt stating that
deriving a figure from present inputs is a valid answer, permitting one line of arithmetic, and
demoting refusal from the first rule to the last.

The prompt does not help. Refusals move from 102 to 100 of 150 and numeric agreement from 0.095 to
0.119, a difference of two questions (p = 0.50). The model mostly rephrases its refusals rather than
attempting more answers, which also exposed a measurement bug: the abstention counter matched only
the exact instructed refusal string, so reworded refusals counted as attempts. Correcting it
raised the reference abstention count from 99 to 102, which is why these figures differ from an
earlier version of this report. Accuracy on questions the model does answer improves (0.242 to
0.278), so permitting arithmetic helps once it commits; it just does not make it commit more
often. We report it as a negative result.

#### Generation model

Making the generator larger does not help. Holding retrieval fixed and swapping Qwen3.5-4B for
gemma-4-12B, three times the size and from a different family, moves no generation metric in
gemma's favour: ROUGE-L 0.096 against 0.100, semantic similarity 0.253 against 0.329, citation F1
0.143 against 0.213. gemma declines more often (112 of 150 against 102) and gets the same number
of figures right, while running at a third of Qwen's tokens per second. When the evidence reaches
the model less than half the time, a bigger generator has little to work with.

**Table 3. Generation and citation metrics, answerable subset.** Numeric agreement is over the
84 questions with a numeric gold answer (Section 3.7).

| Configuration | Generator | ROUGE-L | Semantic sim | Numeric agr. | Citation F1 | Declined (of 150) |
| --- | --- | --- | --- | --- | --- | --- |
| reference | Qwen3.5-4B | 0.100 | 0.329 | 0.095 | 0.213 | 102 |
| 256-token chunks | Qwen3.5-4B | 0.101 | 0.306 | 0.107 | 0.192 | 105 |
| structure-aware | Qwen3.5-4B | 0.102 | 0.321 | 0.107 | 0.192 | 108 |
| derive prompt | Qwen3.5-4B | 0.084 | 0.329 | 0.119 | **0.280** | 100 |
| company filter | Qwen3.5-4B | 0.105 | 0.355 | 0.107 | 0.218 | 95 |
| **company filter + reranking** | Qwen3.5-4B | **0.116** | **0.380** | **0.190** | 0.211 | **89** |
| gemma-4-12B generator | gemma-4-12B | 0.096 | 0.253 | 0.095 | 0.143 | 112 |

![Figure 5](../results/figures/generation_models.png)

**Figure 5. Qwen3.5-4B against gemma-4-12B on identical retrieval.** The larger generator
matches or trails on every metric and abstains more.

## 6. Error analysis

Splitting the answerable questions by where they fail is more informative than the aggregates.
For the reference configuration the gold evidence reaches the top 5 for 56 of 114 questions, and
of those 56 the generator produces a recognisable answer (ROUGE-L above 0.1) only 23 times. The
two failure modes are of similar size (Figure 6), so improving either stage alone leaves most of
the gap in place.

![Figure 6](../results/figures/failure_decomposition.png)

**Figure 6. Where the 114 answerable questions are lost.** 51% never retrieve the evidence,
29% retrieve it but answer weakly, and 20% are answered well.

The failures also split by question type (Figure 7). Novel questions retrieve best at Recall@5
0.68, well above domain-relevant (0.36) and metrics-generated (0.35). Metrics-generated questions
expose a measurement artifact as much as a model failure: their gold answers are bare figures, so
ROUGE-L is near zero (0.001) even when the number is present, which is what numeric agreement was
added to see. Domain-relevant answers are sentences and score highest on semantic similarity
(0.455), since lexical and semantic overlap reward sentences and penalise bare numbers regardless
of correctness.

![Figure 7](../results/figures/by_question_type.png)

**Figure 7. Reference-run metrics by question type, answerable subset.** Metrics-generated
questions score near zero on ROUGE-L because their answers are bare numbers.

**Table 4. Reference run by question type (answerable subset).**

| Question type | Recall@5 | MRR | ROUGE-L | Semantic sim | Citation F1 |
| --- | --- | --- | --- | --- | --- |
| domain-relevant | 0.357 | 0.207 | 0.154 | 0.455 | 0.312 |
| metrics-generated | 0.347 | 0.295 | 0.001 | 0.204 | 0.103 |
| novel-generated | 0.681 | 0.485 | 0.135 | 0.307 | 0.209 |

## 7. Discussion

### 7.1 Key findings

Retrieval is the binding constraint. The content-only changes all land within a narrow band, so
dense search over the whole corpus is near its ceiling. The two changes that beat it use structure
the corpus already carries rather than a bigger model: filtering to the company named in the
question, and reordering a deep pool with a cross-encoder. A BM25 hybrid hurts, and a
three-times-larger generator on identical retrieval changes nothing, so the constraint is in
retrieval, not the model.

The retrieval metric is also misleading. Recall@5 credits nearly all of the gain to the company
filter and almost none to reranking; on whether the system answers correctly, the ordering
reverses. The metric asks whether one chunk shares half its content words with a gold span, which
cannot see reordering, and reordering decides whether the right filing reaches the generator.
ROUGE-L has the same problem on the generation side, scoring 0.001 wherever the gold answer is a
bare figure. Both metrics are standard, and both would have picked the wrong change here.

### 7.2 Validating the retrieval metric

Section 7.1 claims Recall@5 ranked our two retrieval changes in the wrong order. That is the
strongest claim in this report, so we tested it three more times. All three experiments sit
outside the fourteen-arm ablation and use the same answerable subset and the same index.

The first repeats the reranking comparison with a larger cross-encoder. Swapping
bge-reranker-base for bge-reranker-large, with the company filter and the pool of 50 held fixed,
raises Recall@5 from 0.516 to 0.626. Numeric agreement moves from 16 correct of 84 to 17, which a
McNemar exact test cannot separate from chance (p = 1.00). The disagreement is the same as before
with its sign reversed: the base cross-encoder gained 0.007 on Recall@5 and doubled the correct
answers, while the large one gains 0.110 and answers one more question correctly. Ranking the two
by retrieval score picks the wrong one both times.

The second experiment holds retrieval fixed and changes only the text being scored. Indexing
128-token chunks and then scoring the 1024-token window centred on each retrieved chunk raises
Recall@5 from 0.135 to 0.602, though the retrieved chunks are identical. The metric is responding
to window width.

The third varies chunk size and reports Recall@5 beside the share of questions whose gold filing
reaches the top five.

**Table 5. Chunk size against two retrieval measures, answerable subset.**

| Chunk size | Chunks | Recall@5 | Gold filing in top 5 |
| --- | --- | --- | --- |
| 256 | 124,563 | 0.282 | 0.579 |
| 512 | 62,339 | 0.456 | 0.535 |
| 1024 | 31,211 | 0.629 | 0.500 |

The two columns move in opposite directions. Larger chunks retrieve the correct filing less often
and still score higher on Recall@5, because a wider window clears the 0.5 overlap threshold more
easily regardless of what it contains. Read alone, the Recall@5 column recommends the chunk size
that finds the right document least often.

These three results mark where the metric can be trusted. Its threshold is sensitive to chunk
length independently of retrieval quality, so Recall@5 is usable for comparing arms at a fixed
chunk size and unsafe across chunk sizes or against a step that only reorders results. That is why
we report the gold-filing hit rate beside it. This metric is not unusually poor: it is standard,
as is ROUGE-L, which scores 0.001 wherever the gold answer is a bare figure. Both were the obvious
choice for this task, and both would have selected the wrong change.

### 7.3 Limitations

Corpus coverage is the largest limitation: a fifth of the filings no longer download and a further
set are corrupt, so 36 of the 150 questions ask about a document the system never sees. Reporting
the answerable subset keeps that apart from model quality, but the smaller subset makes the
per-type numbers noisier.

The retrieval metric is a proxy. It counts a chunk as relevant when it contains at least half of
the evidence's content words, which handles the length mismatch that breaks exact-span matching
but can still credit a chunk that shares vocabulary without the figure, or miss a paraphrase. The
0.5 threshold is defensible rather than tuned.

Chunks are cut on a tiktoken count, but bge-base reads 512 of its own finer wordpiece tokens, so a
512-token chunk usually exceeds the embedder's limit and its tail is dropped before the vector is
computed. Over a 40-document sample this affects 47% of reference chunks, 23% of structure-aware
ones, and none at 256 tokens. The full text is still stored and shown to the generator, so it
costs retrieval rather than answers, but it means the chunk-size comparison is not clean: the
256-token arm is the only fully searchable one, and structure-aware chunking wins by 0.019 while
being truncated half as often. Chunking on the embedder's own tokenizer would remove the confound.
Relatedly, pdfplumber extracts and caches tables but nothing consumes them, so table content
reaches the index as flattened prose, a likely reason table-heavy questions are hardest.

We ran several pairwise comparisons and report the one that reached significance, so p = 0.039
should be read with that in mind. Reranking was chosen from the Recall@50 headroom before the
generation runs rather than after seeing outcomes, which is why we still treat it as the main
result.

Generation uses small 4-bit models, so absolute answer quality trails a larger hosted model,
especially on arithmetic over a table, and because the models refuse when the passages do not
support an answer, most error surfaces as abstention. Generation is reproducible: re-running a
configuration returns bit-identical answers and therefore identical ROUGE-L and exact-match
scores. An earlier version of this report blamed observed run-to-run drift on the llama.cpp Metal
backend; that was wrong. The drift came from our citation-recall metric, which sampled keywords
from a Python set and so depended on per-process hash ordering, the only metric that moved. It is
fixed, and the citation figures here are recomputed with the corrected metric from the stored
answers; citation precision, which never had the defect, comes out unchanged to every decimal we
report, confirming the recomputation reproduced the original retrieval exactly.

## 8. Conclusion

We built a RAG pipeline for financial filings on free, local models and measured eight design
choices on FinanceBench. Retrieval sets the ceiling: the gold evidence reaches the model for
barely half the answerable questions, and what helped was structure the corpus already carries,
not a bigger model. Filtering to the company named in the question and reordering a deep pool with
a cross-encoder together lift Recall@5 from 0.46 to 0.52 and double the questions answered
correctly, from 8 to 16, while a BM25 hybrid, a larger generator, and a refusal-reducing prompt
all failed. The main lesson is that Recall@5 ranked our two interventions in the opposite order to
their effect on answers, so trusting it would have kept the weaker change and dropped the stronger
one. The pipeline is not specific to finance: any closed PDF corpus with question-and-evidence
data runs the same way, and the coverage gap matters wherever a benchmark's source links decay.

Three next steps follow. Chunking on the embedder's own tokenizer would stop 47% of reference
chunks being truncated before indexing, a correctness fix that also makes the chunk-size
comparison clean. Reranking has more to give, since Recall@50 under the company filter is 0.80
against the 0.52 currently reaching the generator. And an exact-span evidence metric would replace
the token-coverage proxy behind the inversion.

## References

Islam, P. et al. (2023). FinanceBench: A New Benchmark for Financial Question Answering.
https://github.com/patronus-ai/financebench

## Appendix

Reproduction instructions are in `README.md`. Per-configuration metrics and detailed
per-question results are in `results/experiments/`.
