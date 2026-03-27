"""Shared helpers for task execution."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

from prism.core.label_probs import LabelProbabilityComputer
from prism.model import Model

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pandas as pd
except ImportError:
    pd = None


def is_polars(df) -> bool:
    """Return whether *df* is a Polars DataFrame."""
    return pl is not None and isinstance(df, pl.DataFrame)


def is_pandas(df) -> bool:
    """Return whether *df* is a Pandas DataFrame."""
    return pd is not None and isinstance(df, pd.DataFrame)


def get_column(df, column_name: str) -> List[str]:
    """Extract a text column as a plain Python list."""
    if is_polars(df):
        return df[column_name].to_list()
    return df[column_name].tolist()


def add_columns(df, columns: Dict[str, list]):
    """Return a copy of *df* with extra columns attached."""
    if is_polars(df):
        return df.hstack([pl.Series(name=name, values=values) for name, values in columns.items()])

    df = df.copy()
    for name, values in columns.items():
        df[name] = values
    return df


def entropy(probs: Dict[str, float]) -> float:
    """Compute Shannon entropy in bits."""
    result = 0.0
    for probability in probs.values():
        if probability > 0:
            result -= probability * math.log2(probability)
    return result


def build_probability_computer(
    model: Model,
    labels: List[str],
    system_message: str,
    user_message: str,
) -> Tuple[LabelProbabilityComputer, int]:
    """Create the probability computer for a concrete prompt shape."""
    prompt_text = model.format_prompt(system_message, user_message)
    if model.think_end is not None:
        prompt_text += model.think_end

    tokenization = model.tokenize_labels_in_context(labels, prompt_text)
    computer = LabelProbabilityComputer(
        tokenization.label_tokens,
        model.backend,
        decode=model.decode,
    )
    return computer, tokenization.n_absorbed


def direct_prompt_tokens(prompt_tokens: List[int], model: Model) -> List[int]:
    """Append the think-end sequence when direct mode must skip reasoning."""
    if model.think_end_tokens:
        return prompt_tokens + model.think_end_tokens
    return prompt_tokens


def effective_prompt_tokens(prompt_tokens: List[int], n_absorbed: int) -> List[int]:
    """Back up any prompt tokens absorbed into the in-context label boundary."""
    if n_absorbed <= 0:
        return prompt_tokens
    if n_absorbed >= len(prompt_tokens):
        raise ValueError(
            f"n_absorbed ({n_absorbed}) must be smaller than prompt length ({len(prompt_tokens)})"
        )
    return prompt_tokens[:-n_absorbed]


def resolve_contexts(
    context: Optional[Union[str, List[Optional[str]]]],
    n_rows: int,
) -> List[Optional[str]]:
    """Normalize context into one value per row."""
    if context is None:
        return [None] * n_rows
    if isinstance(context, str):
        return [context] * n_rows
    if len(context) != n_rows:
        raise ValueError(f"context list length ({len(context)}) != number of rows ({n_rows})")
    return context


def is_context_constant(contexts: List[Optional[str]]) -> bool:
    """Return whether all context values are identical."""
    if not contexts:
        return True
    first = contexts[0]
    return all(context == first for context in contexts)


def get_constant_context(contexts: List[Optional[str]]) -> Optional[str]:
    """Return the shared context value when the caller already knows it is constant."""
    if not contexts:
        return None
    return contexts[0]
