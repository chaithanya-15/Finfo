#!/usr/bin/env python3
"""
Optional Gradio demo for the FinanceBench RAG system.

Ask a question about a company's filings; the app retrieves passages (restricted to the
company named in the question), generates a cited answer with the local model, and shows the
retrieved evidence alongside it. This is a demo, not part of the graded pipeline.

Install Gradio first (it is not in requirements.txt):
    pip install gradio
Then:
    python app.py
and open the printed local URL.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from src.retrieval.retrieve import load_index
from src.retrieval.metadata_filter import load_company_list, filtered_search
from src.generation.generate import create_qa_pipeline

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
INDEX = "indexes/BAAI_bge-base-en-v1.5__fixed_512_overlap64"

print("Loading index and generator (first load takes a moment)...")
STORE = load_index(INDEX, "faiss")
COMPANIES = load_company_list("data/financebench_document_information.jsonl")
MODEL = create_qa_pipeline(model_name="Qwen3.5-4B")


def answer(question: str, k: int):
    """
    Retrieve, generate, and format the answer plus the evidence for the UI.

    Args:
        question: The user's question
        k: Number of passages to retrieve

    Returns:
        Tuple of (answer markdown, evidence markdown)
    """
    if not question.strip():
        return "Ask a question about a company's filing.", ""

    contexts = filtered_search(STORE, question, EMBED_MODEL, int(k), COMPANIES)
    result = MODEL.answer_question(question, contexts)

    valid = set(result["valid_citations"])
    evidence = []
    for c in contexts:
        tag = f"[{c['doc_name']}_c{c['chunk_index']}]"
        mark = "cited" if tag in valid else ""
        snippet = c["text"].strip().replace("\n", " ")[:400]
        evidence.append(f"**{tag}** {mark}  (score {c['score']:.2f})\n\n{snippet}")

    return result["answer"], "\n\n---\n\n".join(evidence) if evidence else "No passages retrieved."


with gr.Blocks(title="FinanceBench RAG") as demo:
    gr.Markdown("# FinanceBench RAG\nAsk a question about a company's SEC filing. "
                "The answer cites the retrieved passages it used.")
    with gr.Row():
        question = gr.Textbox(label="Question", scale=4,
                              placeholder="What were Apple's total net sales in FY2022?")
        k = gr.Slider(1, 10, value=5, step=1, label="Passages (k)", scale=1)
    ask = gr.Button("Answer", variant="primary")
    out_answer = gr.Markdown(label="Answer")
    with gr.Accordion("Retrieved evidence", open=False):
        out_evidence = gr.Markdown()
    ask.click(answer, inputs=[question, k], outputs=[out_answer, out_evidence])
    question.submit(answer, inputs=[question, k], outputs=[out_answer, out_evidence])

if __name__ == "__main__":
    demo.launch()
