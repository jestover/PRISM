# PRISM

**PRobabilistic Inference for Structured Measurement**

PRISM is a Python package for extracting full probability distributions from local LLMs over discrete label sets.

Where API-based tools return point estimates (a single label or number), PRISM runs models locally and returns the entire distribution — giving you uncertainty quantification, distribution shape, and richer downstream analysis for free.

## Status

**Work in progress.** Not yet ready for use. The core probability extraction engine works; the high-level API and packaging are under active development.

## Installation

```bash
pip install prism-lm
```

Backends are installed separately depending on your hardware:

```bash
# Apple Silicon (MLX)
pip install "prism-lm[mlx]"

# CUDA / CPU (PyTorch)
pip install "prism-lm[torch]"
```

## Quick Example

```python
import prism

model = prism.load_model("mlx-community/gpt-oss-20b-MXFP4-Q8", backend="mlx")

result = prism.classify(
    df, "text",
    labels=["positive", "negative", "neutral"],
    model=model,
)
# Returns DataFrame with prob_positive, prob_negative, prob_neutral,
# predicted_class, max_prob, entropy columns
```

## License

MIT
