# Generation

Not implemented yet. This package sits between retrieval and evaluation. It should take the
chunks returned by `src/retrieval/retrieve.py` and produce an answer with citations that
`src/evaluation/evaluate.py` can score.

Whatever goes here has to read the `generation:` block in `configs/base_config.yaml`:

- `model_name`, currently `meta-llama/Llama-3.1-8B-Instruct`
- `device` and `load_in_4bit`, for loading the quantised model locally
- `generation_kwargs`: `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty`

The project brief also constrains the output. Answers must cite the retrieved chunks they come
from, and they must not use anything outside the provided corpus. The citation checks in
`src/evaluation/evaluate.py` assume both.
