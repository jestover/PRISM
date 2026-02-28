# PRISM

**PRobabilistic Inference for Structured Measurement**

Like a prism splits white light into its full spectrum, PRISM splits an LLM's point-estimate predictions into their underlying probability distributions.

PRISM is a Python package for extracting full probability distributions from local LLMs over discrete label sets. Given text and a set of labels, PRISM doesn't just tell you which label the model picked — it tells you the probability the model assigns to *every* label, revealing the model's uncertainty, confidence, and the full shape of its beliefs.

## Status

**Work in progress.** The core engine and high-level API work on a single machine (Apple Silicon via MLX, CUDA/CPU via PyTorch). Distributed computing (SLURM, Grid Engine) and checkpointing are planned but not yet implemented.

## Relationship to GABRIEL

[GABRIEL](https://github.com/openai/GABRIEL) (Asirvatham, Mokski, & Shleifer 2026, "GPT as a Measurement Tool") is a landmark Python library that makes LLMs practical as a measurement tool for social science research. GABRIEL wraps the OpenAI API to classify and rate qualitative data at scale, demonstrating that LLM-generated labels match human labels in reliability across hundreds of datasets. It is, in effect, Stata for LLM-based text measurement.

PRISM builds on GABRIEL's contributions and is designed to complement it. Where GABRIEL calls the OpenAI API and returns a point estimate (a single label or rating), PRISM runs models locally and returns the full probability distribution over all possible labels. This reveals the model's uncertainty — two texts rated "50" might have very different distributions, one sharply peaked and the other bimodal — which is invisible in point estimates. The point estimate can always be recovered (it's just the mode or expected value), but the reverse is impossible.

We aim to keep PRISM's API consistent with GABRIEL's so that researchers can move between the two packages depending on what their current project needs.

### When to Use Each

**GABRIEL is the better choice when you:**
- Want access to the most capable proprietary models (GPT-5, etc.)
- Need to scale to very large datasets quickly (API parallelism is simpler than managing local compute)
- Are working on tasks where a point estimate is sufficient
- Don't need uncertainty quantification
- Want the simplest possible setup

**PRISM is the better choice when you:**
- Need the full probability distribution, not just a point estimate
- Want uncertainty quantification (entropy, standard deviation)
- Need to avoid per-token API costs (one-time local compute instead)
- Want to use open-source models
- Need to inspect model internals (logits, token probabilities)

**On scaling:** GABRIEL scales naturally through OpenAI's API infrastructure. PRISM currently runs on a single machine. We plan to add support for compute clusters (SLURM, Grid Engine), but local inference will always require more setup than making API calls.

## Installation

```bash
pip install prism-lm
```

Install the backend for your hardware:

```bash
# Apple Silicon (MLX)
pip install "prism-lm[mlx]"

# CUDA / CPU (PyTorch)
pip install "prism-lm[torch]"
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

PRISM works by examining the raw logits (pre-softmax scores) that an LLM produces at the token position where it would generate a label. Rather than letting the model sample a single token, PRISM reads the probability the model assigns to each possible label token and computes a normalized distribution.

For labels that span multiple tokens, PRISM builds a trie of token sequences and identifies branch points where sequences diverge, evaluating logits only where needed. For example, classifying into three labels that tokenize as `[1,5,9]`, `[1,5,10]`, and `[2,6,8]` requires only 2 logit evaluations instead of 3, because the first two labels share a prefix.

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

The model generates thinking tokens freely, and once it signals that it's ready to answer, PRISM extracts the probability distribution from that position.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use PRISM in academic work, please also cite GABRIEL, which established the paradigm of using LLMs as measurement tools for social science:

> Asirvatham, Mokski, & Shleifer (2026). "GPT as a Measurement Tool." Available at: https://github.com/openai/GABRIEL
