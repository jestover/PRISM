"""PRISM — PRobabilistic Inference for Structured Measurement.

Extract full probability distributions from local LLMs over discrete label sets.

Quick start::

    import prism

    model = prism.load_model("mlx-community/gpt-oss-20b-MXFP4-Q8", backend="mlx")

    result = prism.classify(df, "text", labels=["positive", "negative", "neutral"], model=model)
"""

from prism.api import binary_classify, classify, rate
from prism.model import Model, load_model
from prism.utils import set_log_level

__all__ = [
    "binary_classify",
    "classify",
    "load_model",
    "Model",
    "rate",
    "set_log_level",
]
