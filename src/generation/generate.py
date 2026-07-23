#!/usr/bin/env python3
"""
Generation module for FinanceBench RAG pipeline.
Loads a locally executable LLM and produces grounded, cited answers from retrieved chunks.
"""

import os
import glob
import json
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Local GGUF weights, resolved from the Hugging Face cache. Keys are the names used in
# configs/*.yaml so an experiment can switch generator without touching code.
LOCAL_MODEL_PATTERNS = {
    "Qwen3.5-4B": "models--unsloth--Qwen3.5-4B-GGUF/snapshots/*/Qwen3.5-4B-UD-Q4_K_XL.gguf",
    # gemma-4-12B has a standard architecture, so unlike gemma-4-E2B it runs on Metal without
    # crashing. Get it with: huggingface-cli download lmstudio-community/gemma-4-12B-it-GGUF
    "gemma-4-12B": "models--lmstudio-community--gemma-4-12B-it-GGUF/snapshots/*/gemma-4-12B-it-Q8_0.gguf",
    "gemma-4-E2B": "models--unsloth--gemma-4-E2B-it-GGUF/snapshots/*/gemma-4-E2B-it-UD-Q4_K_XL.gguf",
}

# The same weights, when present in a local LM Studio install, are found here first so an
# existing download is reused rather than fetched again.
LMSTUDIO_MODEL_PATTERNS = {
    "gemma-4-12B": "lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q8_0.gguf",
}

SYSTEM_PROMPT = """You answer questions about financial filings using only the excerpts provided.

Rules:
1. Use only the excerpts. If they do not contain the answer, reply exactly: Not enough information in the provided context.
2. Cite every factual claim with the tag shown above the excerpt you used, for example [APPLE_2022_10K_c17].
3. Put the citation directly after the claim it supports.
4. Be brief. State the figure or fact and stop. Do not restate the question or explain your reasoning."""


# Second prompt, written against two measured failure modes of the one above. The model
# abstained on 30% of questions whose evidence had actually been retrieved, and it refused on
# presentation rather than absence: asked for 3M's FY2018 capital expenditure it replied that
# the excerpts held no such figure "in a cash flow statement format".
#
# Two changes follow from that. Deriving a figure from inputs that are present is stated to be
# a valid answer, since most FinanceBench metrics answers must be computed rather than read off.
# And the ban on reasoning is lifted for arithmetic, because 43 of the 150 questions need
# numerical reasoning and forbidding the working is close to the worst instruction for a small
# model. Refusal moves last and gets a tighter test, but keeps its exact wording, which both
# run_experiments.py and the evaluator match on.
SYSTEM_PROMPT_DERIVE = """You answer questions about financial filings using only the excerpts provided.

Rules:
1. Answer from the excerpts. A figure often has to be derived rather than read off: when the excerpts contain the inputs, do the arithmetic and give the result. An answer is supported when the numbers behind it appear in the excerpts, even if the final figure never appears verbatim or in the format the question uses.
2. When a value is calculated, show the arithmetic in one short line, then state the result.
3. Cite every factual claim with the tag shown above the excerpt you used, for example [APPLE_2022_10K_c17]. Put the citation directly after the claim it supports.
4. Be brief. State the figure or fact and stop. Do not restate the question.
5. Only when the excerpts lack the figures needed to answer or to derive the answer, reply exactly: Not enough information in the provided context."""

PROMPTS = {
    "default": SYSTEM_PROMPT,
    "derive": SYSTEM_PROMPT_DERIVE,
}


def resolve_prompt(system_prompt: str) -> str:
    """
    Turn a configured prompt name into prompt text.

    A bare unknown name is an error rather than a fallback: treating it as literal prompt text
    would let `prompt: derve` in a config run for half an hour on a one-word system prompt and
    report the result as if it meant something.

    Args:
        system_prompt: Key from PROMPTS, or the prompt text itself

    Returns:
        The prompt text

    Raises:
        ValueError: If given a short bare string that is not a known key
    """
    if system_prompt in PROMPTS:
        return PROMPTS[system_prompt]
    if "\n" in system_prompt or len(system_prompt) > 80:
        return system_prompt
    raise ValueError(f"Unknown prompt '{system_prompt}'. Known: {sorted(PROMPTS)}. "
                     "Pass the prompt text itself to use something else.")


def resolve_model_path(model_name: str, cache_dir: Optional[str] = None) -> str:
    """
    Resolve a configured model name to a GGUF file on this machine.

    Args:
        model_name: Key from LOCAL_MODEL_PATTERNS, or a direct path to a .gguf file
        cache_dir: Hugging Face hub cache directory

    Returns:
        Absolute path to the GGUF weights

    Raises:
        FileNotFoundError: If the weights are not present locally
    """
    if model_name.endswith(".gguf"):
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"GGUF file not found: {model_name}")
        return model_name

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    known = sorted(set(LOCAL_MODEL_PATTERNS) | set(LMSTUDIO_MODEL_PATTERNS))
    if model_name not in known:
        raise FileNotFoundError(
            f"Unknown model '{model_name}'. Known names: {known}. "
            "Pass a path to a .gguf file to use something else."
        )

    # Try an existing LM Studio download first, then the Hugging Face cache. Listing both
    # keeps the resolver portable: a machine with only the HF download still finds the file.
    search = []
    if model_name in LMSTUDIO_MODEL_PATTERNS:
        search.append((os.path.expanduser("~/.lmstudio/models"), LMSTUDIO_MODEL_PATTERNS[model_name]))
    if model_name in LOCAL_MODEL_PATTERNS:
        search.append((cache_dir, LOCAL_MODEL_PATTERNS[model_name]))

    for root, pattern in search:
        matches = glob.glob(os.path.join(root, pattern))
        if matches:
            return matches[0]

    tried = "; ".join(os.path.join(r, p) for r, p in search)
    raise FileNotFoundError(f"No local weights for '{model_name}'. Looked for: {tried}")


def citation_key(context: Dict[str, Any]) -> str:
    """
    Build the citation tag for a retrieved chunk.

    The format must match the pattern validated in src/evaluation/evaluate.py,
    which accepts [doc_name_c<chunk_index>].

    Args:
        context: Retrieved chunk dictionary from search_index

    Returns:
        Citation tag including the surrounding brackets
    """
    return f"[{context.get('doc_name', 'unknown')}_c{context.get('chunk_index', 0)}]"


class FinancialQAModel:
    """Generates cited answers from retrieved financial filing excerpts."""

    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1,
                 seed: int = 1234, verbose: bool = False,
                 system_prompt: str = "default"):
        """
        Load a GGUF model through llama.cpp.

        Args:
            model_path: Path to GGUF weights
            n_ctx: Context window in tokens
            n_gpu_layers: Layers to offload to the GPU, -1 for all
            seed: Sampling seed, fixed so runs are reproducible
            verbose: Pass llama.cpp's own logging through
            system_prompt: Key from PROMPTS, or the prompt text itself
        """
        from llama_cpp import Llama

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.system_prompt = resolve_prompt(system_prompt)
        logger.info(f"Loading generator from {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string using the model's own tokenizer."""
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def build_prompt(self, question: str, contexts: List[Dict[str, Any]],
                     max_context_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Assemble the user prompt from retrieved chunks.

        Chunks are added in retrieval order until the token budget is reached. Any chunk
        that does not fit is dropped and reported, so truncation is visible in the results
        rather than silent.

        Args:
            question: The user question
            contexts: Retrieved chunks, best first
            max_context_tokens: Budget for excerpts, defaults to half the context window

        Returns:
            Dictionary with the prompt text, the contexts actually included, and the
            number dropped for lack of room
        """
        if max_context_tokens is None:
            max_context_tokens = self.n_ctx // 2

        blocks = []
        used = []
        spent = 0

        for ctx in contexts:
            block = f"{citation_key(ctx)}\n{ctx.get('text', '').strip()}"
            cost = self.count_tokens(block)
            if spent + cost > max_context_tokens:
                continue
            blocks.append(block)
            used.append(ctx)
            spent += cost

        dropped = len(contexts) - len(used)
        if dropped:
            logger.warning(f"Dropped {dropped} of {len(contexts)} chunks to fit the context window")

        excerpts = "\n\n".join(blocks) if blocks else "(no excerpts retrieved)"
        prompt = f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}\n\nAnswer:"

        return {
            "prompt": prompt,
            "used_contexts": used,
            "dropped_contexts": dropped,
            "context_tokens": spent,
        }

    def answer_question(self, question: str, contexts: List[Dict[str, Any]],
                        max_new_tokens: int = 256, temperature: float = 0.1,
                        top_p: float = 0.95, repetition_penalty: float = 1.1,
                        **kwargs) -> Dict[str, Any]:
        """
        Answer a question from retrieved excerpts.

        Args:
            question: The user question
            contexts: Retrieved chunks from search_index
            max_new_tokens: Cap on generated tokens
            temperature: Sampling temperature, low for factual extraction
            top_p: Nucleus sampling cutoff
            repetition_penalty: Penalty applied to repeated tokens
            **kwargs: Ignored, so config files can carry extra keys

        Returns:
            Dictionary with the answer, the citation tags it used, which of those were
            valid, and bookkeeping about context that did not fit
        """
        built = self.build_prompt(question, contexts)

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": built["prompt"]},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
        )

        answer = response["choices"][0]["message"]["content"].strip()

        import re
        cited = re.findall(r'\[[^\]]+_[cp]\d+\]', answer)
        valid_keys = {citation_key(c) for c in built["used_contexts"]}

        return {
            "answer": answer,
            "question": question,
            "citations": cited,
            "valid_citations": [c for c in cited if c in valid_keys],
            "n_contexts": len(built["used_contexts"]),
            "dropped_contexts": built["dropped_contexts"],
            "context_tokens": built["context_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
        }


def create_qa_pipeline(model_name: str = "Qwen3.5-4B", device: str = "auto",
                       load_in_4bit: bool = True, n_ctx: int = 8192,
                       system_prompt: str = "default", **kwargs) -> FinancialQAModel:
    """
    Build a generation pipeline from config values.

    The weights are 4-bit GGUF files run through llama.cpp with Metal, which is how
    quantised local inference works on Apple silicon. bitsandbytes, which the original
    config assumed, has no macOS build.

    Args:
        model_name: Key from LOCAL_MODEL_PATTERNS or a path to a .gguf file
        device: Kept for interface compatibility, llama.cpp selects the backend itself
        load_in_4bit: Kept for interface compatibility, the GGUF weights are already 4-bit
        n_ctx: Context window in tokens
        system_prompt: Key from PROMPTS ("default" or "derive"), or the prompt text itself
        **kwargs: Passed through to FinancialQAModel

    Returns:
        A loaded FinancialQAModel
    """
    if not load_in_4bit:
        logger.warning("load_in_4bit=False ignored: the local GGUF weights are 4-bit quantised")

    model_path = resolve_model_path(model_name)
    n_gpu_layers = 0 if device == "cpu" else -1
    return FinancialQAModel(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                            system_prompt=system_prompt, **kwargs)


if __name__ == "__main__":
    # Smoke test against a hand-written context, no index required.
    model = create_qa_pipeline()
    demo_contexts = [
        {
            "doc_name": "APPLE_2022_10K",
            "chunk_index": 17,
            "text": "Total net sales were $394,328 million in 2022 compared to $365,817 million in 2021.",
        },
        {
            "doc_name": "APPLE_2022_10K",
            "chunk_index": 41,
            "text": "Research and development expense was $26,251 million in 2022.",
        },
    ]
    out = model.answer_question("What were Apple's total net sales in FY2022?", demo_contexts)
    print(json.dumps(out, indent=2))
