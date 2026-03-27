"""PRISM — PRobabilistic Inference for Structured Measurement.

Extract full probability distributions from local LLMs over discrete label sets.

Quick start::

    import prism

    model = prism.load_model("mlx-community/gpt-oss-20b-MXFP4-Q8", backend="mlx")

    result = prism.classify(df, "text", labels=["positive", "negative", "neutral"], model=model)
"""

from prism.api import classify, label, rate
from prism.model import Model, load_model
from prism.tasks import Classify, Label, Rate
from prism.utils import set_log_level

__all__ = [
    "Classify",
    "classify",
    "Label",
    "label",
    "load_model",
    "Model",
    "Rate",
    "rate",
    "set_log_level",
]
