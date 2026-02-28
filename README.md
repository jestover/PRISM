# PRISM

**PRobabilistic Inference for Structured Measurement**

Like a prism splits white light into its full spectrum, PRISM splits an LLM's point-estimate predictions into their underlying probability distributions.

PRISM is a Python package for extracting full probability distributions from local LLMs over discrete label sets. Given text and a set of labels, PRISM doesn't just tell you which label the model picked — it tells you the probability the model assigns to *every* label, revealing the model's uncertainty, confidence, and the full shape of its beliefs.

## Status

**Work in progress.** The core engine and high-level API work on a single machine (Apple Silicon via MLX, CUDA/CPU via PyTorch). Distributed computing (SLURM, Grid Engine) and checkpointing are planned but not yet implemented.

## Relationship to GABRIEL

[GABRIEL](https://github.com/openai/GABRIEL) is a Python library that wraps the OpenAI API and provides a clean, practical interface for researchers to classify and rate qualitative data at scale — in effect, Stata for LLM-based text measurement. Asirvatham, Mokski, & Shleifer introduced GABRIEL in their paper *GPT as a Measuremnt Tool*, which demonstrates that LLM-generated labels match human labels in reliability across hundreds of datasets.

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

Any HuggingFace-compatible model works — no hardcoded model configs.

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

### Binary Classify

Independent true/false evaluation for multiple labels:

```python
result = prism.binary_classify(
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

PRISM works by examining the raw logits (pre-softmax scores) that an LLM produces at the token position where it would generate a label. Rather than letting the model sample a single token, PRISM computes the probabilities the model assigns to each possible label token and returns .

### Chain of Thought

For models that support reasoning, PRISM can let the model think before extracting probabilities:

```python
model = prism.load_model(
    "mlx-community/gpt-oss-20b-MXFP4-Q8",
    backend="mlx",
    reasoning_prefix="<|channel|>final<|message|>",  # marks end of thinking
)

result = prism.classify(
    df, "text",
    labels=["positive", "negative", "neutral"],
    model=model,
    use_reasoning=True,
)
```

The model generates thinking tokens freely, and once it signals that it's ready to answer, PRISM extracts the probability distribution from that position. Please note that this will often compress the probability distribution since the model often decides on a label in the thinking phase. When `use_reasoning=True`, PRISM adds a `thinking_text` column to the output DataFrame containing the model's reasoning for each row.

## License

MIT. See [LICENSE](LICENSE).

## Citation

A paper describing PRISM's measurement technique is in preparation.

If you use PRISM in academic work, for now please cite

- Stover, J. (2026). *PRISM: PRobabilistic Inference for Structured Measurement* (software). GitHub repository: https://github.com/jestover/PRISM

This project and its structure owes a lot to the work on GABRIEL, you might also consider citing:

- Asirvatham, H., Mokski, E., and Shleifer, A. (2026). *GPT as a Measurement Tool*. NBER Working Paper No. 34834.

- Asirvatham, H. and Mokski, E. (2026). *GABRIEL: Generalized Attribute-Based Ratings Information Extraction Library* (software). GitHub repository: https://github.com/openai/GABRIEL
