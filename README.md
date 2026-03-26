# PRISM

**PRobabilistic Inference for Structured Measurement**

Like a prism splits white light into its full spectrum, PRISM splits an LLM's point-estimate predictions into their underlying probability distributions.

PRISM is a Python package for extracting full probability distributions from local LLMs over discrete label sets. Given text and a set of labels, PRISM doesn't just tell you which label the model picked — it tells you the probability the model assigns to *every* label, revealing the model's uncertainty, confidence, and the full shape of its beliefs.

## Status

**Alpha.** PRISM currently targets single-machine local inference (Apple Silicon via MLX, CUDA/CPU via PyTorch). Breaking changes are still expected while the probability-extraction contract is being hardened.

The current near-term focus is correctness at the prompt/token boundary: in-context label tokenization, prompt-boundary absorption, prefix-overlap label handling, and parity between cached and uncached extraction. Distributed computing (SLURM, Grid Engine) and checkpointing remain future work.

The current top-level task surface is `classify`, `rate`, and `label`.

## Documentation

- [`docs/overview.md`](docs/overview.md) for the project overview
- [`docs/realignment.md`](docs/realignment.md) for the active roadmap
- [`docs/beta_task_list.md`](docs/beta_task_list.md) for the prioritized beta task list
- [`docs/prompt_alignment.md`](docs/prompt_alignment.md) for the current prompt-template comparison against GABRIEL
- [`spec.md`](spec.md) for the detailed reference

## Relationship to GABRIEL

[GABRIEL](https://github.com/openai/GABRIEL) is a Python library that wraps the OpenAI API and provides a clean, practical interface for researchers to classify and rate qualitative data at scale — in effect, Stata for LLM-based text measurement. Asirvatham, Mokski, & Shleifer introduced GABRIEL in their paper *GPT as a Measurement Tool*, which demonstrates that LLM-generated labels match human labels in reliability across hundreds of datasets.

PRISM builds on GABRIEL's contributions and is designed to complement it. Where GABRIEL calls the OpenAI API and returns a point estimate (a single label or rating), PRISM runs models locally and returns the full probability distribution over all possible labels. This reveals the model's uncertainty — two texts rated "50" might have very different distributions, one sharply peaked and the other bimodal — which is invisible in point estimates. The point estimate can always be recovered (it's just the mode or expected value), but the reverse is impossible.

We aim to keep PRISM's API consistent with GABRIEL's so that researchers can move between the two packages depending on what their current project needs.

### When to Use Each

**GABRIEL is the better choice when you:**
- Want access to the most capable proprietary models (GPT-5, etc.)
- Need to scale to very large datasets quickly (API parallelism is simpler than managing local compute)
- Are working on tasks where a point estimate is sufficient
- Don't need uncertainty quantification
- Want the simplest possible setup
- Want results that don't come from a finite set of labels

**PRISM is the better choice when you:**
- Need the full probability distribution, not just a point estimate
- Want uncertainty quantification (entropy, standard deviation)
- Need to avoid per-token API costs (one-time local compute instead)
- Want to use open-source models
- Need to inspect model internals (logits, token probabilities)

**On scaling:** GABRIEL scales naturally through OpenAI's API infrastructure. PRISM currently runs on a single machine. We plan to add support for compute clusters (SLURM, Grid Engine), but local inference will always require more setup than making API calls.

## Installation

PRISM is not yet on PyPI. Install directly from GitHub:

```bash
pip install "prism-lm @ git+https://github.com/jestover/PRISM.git"
```

Install the backend for your hardware:

```bash
# Apple Silicon (MLX)
pip install "prism-lm[mlx] @ git+https://github.com/jestover/PRISM.git"

# CUDA / CPU (PyTorch)
pip install "prism-lm[torch] @ git+https://github.com/jestover/PRISM.git"
```

## Quick Start

### Load a Model

```python
import prism

model = prism.load_model(
    "mlx-community/gpt-oss-20b-MXFP4-Q8",
    backend="mlx",       # "mlx", "torch", or "auto"
)
```

PRISM is designed to work with HuggingFace-compatible chat-template/tokenizer setups without hardcoded model configs, but verified support is still being narrowed to a small tested model matrix for beta. PRISM auto-detects reasoning models (QwQ, DeepSeek-R1, Qwen3, gpt-oss, etc.) from their chat template and configures the think-end sequence automatically. You can override with `think_end="</think>"` if needed.

### Classify

Probability distribution over mutually exclusive labels:

```python
result = prism.classify(
    df,
    column_name="text",
    labels=["positive", "negative", "neutral"],
    model=model,
)
```

Returns the input DataFrame with added columns:

| Column | Description |
|--------|-------------|
| `prob_positive`, `prob_negative`, `prob_neutral` | P(label) for each label |
| `predicted_class` | Argmax label |
| `max_prob` | Confidence of prediction |
| `entropy` | Shannon entropy (bits) — higher means more uncertain |

### Rate

Probability distribution over an integer scale:

```python
result = prism.rate(
    df,
    column_name="text",
    attribute="populism",
    model=model,
    scale_min=0,
    scale_max=100,
)
```

Returns columns: `prob_0` through `prob_100`, `expected_value`, `std_dev`, `mode`, `entropy`.

### Label

Independent true/false evaluation for multiple labels:

```python
result = prism.label(
    df,
    column_name="text",
    labels={
        "toxic": "Contains toxic language",
        "sarcastic": "Uses sarcasm or irony",
    },
    model=model,
)
```

Returns columns: `prob_true_toxic`, `predicted_toxic`, `prob_true_sarcastic`, `predicted_sarcastic`.

## How It Works

PRISM works by examining the raw logits (pre-softmax scores) that an LLM produces at the token position where it would generate a label. Rather than letting the model sample a single token, PRISM computes the probabilities the model assigns to each possible label token and returns the full distribution over the candidate labels.

### Chain of Thought

For models that support reasoning, PRISM can let the model think before extracting probabilities:

```python
model = prism.load_model(
    "mlx-community/gpt-oss-20b-MXFP4-Q8",
    backend="mlx",
)

result = prism.classify(
    df, "text",
    labels=["positive", "negative", "neutral"],
    model=model,
    use_reasoning=True,
)
```

PRISM auto-detects the think-end sequence from the model's chat template (e.g. `</think>` for QwQ/Qwen3, `<|channel|>final<|message|>` for gpt-oss). The model generates thinking tokens freely, and once it produces the think-end sequence, PRISM extracts the probability distribution from that position. Please note that this will often compress the probability distribution since the model often decides on a label in the thinking phase. When `use_reasoning=True`, PRISM adds a `thinking_text` column to the output DataFrame containing the model's reasoning for each row.

## License

MIT. See [LICENSE](LICENSE).

## Citation

A paper describing PRISM's measurement technique is in preparation.

If you use PRISM in academic work, for now please cite

- Stover, J. (2026). *PRISM: PRobabilistic Inference for Structured Measurement* (software). GitHub repository: https://github.com/jestover/PRISM

This project and its structure owes a lot to the work on GABRIEL, you might also consider citing:

- Asirvatham, H., Mokski, E., and Shleifer, A. (2026). *GPT as a Measurement Tool*. NBER Working Paper No. 34834.

- Asirvatham, H. and Mokski, E. (2026). *GABRIEL: Generalized Attribute-Based Ratings Information Extraction Library* (software). GitHub repository: https://github.com/openai/GABRIEL
