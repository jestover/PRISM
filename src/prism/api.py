"""High-level API: classify, rate, binary_classify."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

from prism.core.label_probs import LabelProbabilityComputer
from prism.model import Model
from prism.prompts.templates import PromptBuilder
from prism.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional DataFrame library imports
# ---------------------------------------------------------------------------

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def _is_polars(df) -> bool:
    return pl is not None and isinstance(df, pl.DataFrame)


def _is_pandas(df) -> bool:
    return pd is not None and isinstance(df, pd.DataFrame)


def _get_column(df, column_name: str) -> List[str]:
    """Extract a column as a plain list, regardless of Polars or Pandas."""
    if _is_polars(df):
        return df[column_name].to_list()
    return df[column_name].tolist()


def _add_columns(df, columns: Dict[str, list]):
    """Return a copy of the DataFrame with new columns added."""
    if _is_polars(df):
        return df.hstack([pl.Series(name=k, values=v) for k, v in columns.items()])
    # Pandas
    df = df.copy()
    for k, v in columns.items():
        df[k] = v
    return df


def _entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy in bits."""
    h = 0.0
    for p in probs.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def classify(
    df,
    column_name: str,
    labels: List[str],
    model: Model,
    *,
    label_descriptions: Optional[Dict[str, str]] = None,
    use_reasoning: bool = False,
    max_thinking_tokens: int = 2048,
    additional_instructions: Optional[str] = None,
    context: Optional[Union[str, List[Optional[str]]]] = None,
    shuffle_labels: bool = True,
    random_seed: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Classify text into one of several mutually exclusive labels.

    Args:
        df: Polars or Pandas DataFrame.
        column_name: Column containing text to classify.
        labels: Possible labels (e.g. ``["positive", "negative", "neutral"]``).
        model: Model from :func:`~prism.load_model`.
        label_descriptions: Optional ``{label: description}`` dict.
        use_reasoning: Allow chain-of-thought before answering.
        max_thinking_tokens: Max tokens for thinking phase.
        additional_instructions: Appended to system prompt.
        context: Optional context string (applied to all rows) or list of
            per-row context strings.
        shuffle_labels: Randomize label order per prompt to prevent position bias.
        random_seed: Seed for label shuffling and reproducibility.
        save_dir: Not yet implemented (placeholder for checkpointing).

    Returns:
        Input DataFrame with added columns: ``prob_{label}`` for each label,
        ``predicted_class``, ``max_prob``, and ``entropy``.
    """
    texts = _get_column(df, column_name)
    n = len(texts)
    logger.info(f"classify: {n} texts, {len(labels)} labels, use_reasoning={use_reasoning}")

    builder = PromptBuilder(random_seed=random_seed)
    label_token_sequences = model.tokenize_labels(labels)
    computer = LabelProbabilityComputer(
        label_token_sequences, model.backend, decode=model.decode
    )

    contexts = _resolve_contexts(context, n)
    all_probs: List[Dict[str, float]] = []

    for i, text in enumerate(texts):
        system_msg, user_msg = builder.render_classify(
            text=text,
            labels=labels,
            label_descriptions=label_descriptions,
            context=contexts[i],
            additional_instructions=additional_instructions,
            shuffle=shuffle_labels,
        )
        prompt_tokens = model.tokenize_prompt(system_msg, user_msg)

        if use_reasoning and model.can_reason:
            result = computer.compute_probabilities_with_cot(
                prompt_tokens, model.reasoning_prefix_tokens, max_thinking_tokens
            )
            probs = result.probabilities
        else:
            # Append reasoning prefix to skip thinking phase on reasoning models
            direct_tokens = _direct_prompt_tokens(prompt_tokens, model)
            probs = computer.compute_probabilities(direct_tokens)

        all_probs.append(probs)

        if (i + 1) % 100 == 0 or i == n - 1:
            logger.info(f"classify: processed {i + 1}/{n}")

    # Build output columns
    columns: Dict[str, list] = {}
    for label in labels:
        columns[f"prob_{label}"] = [p[label] for p in all_probs]

    columns["predicted_class"] = [max(p, key=p.get) for p in all_probs]
    columns["max_prob"] = [max(p.values()) for p in all_probs]
    columns["entropy"] = [_entropy(p) for p in all_probs]

    return _add_columns(df, columns)


# ---------------------------------------------------------------------------
# rate
# ---------------------------------------------------------------------------


def rate(
    df,
    column_name: str,
    attribute: str,
    model: Model,
    *,
    attribute_description: Optional[str] = None,
    scale_min: int = 0,
    scale_max: int = 100,
    use_reasoning: bool = False,
    max_thinking_tokens: int = 2048,
    additional_instructions: Optional[str] = None,
    context: Optional[Union[str, List[Optional[str]]]] = None,
    random_seed: Optional[int] = None,
    save_dir: Optional[str] = None,
):
    """Rate text on an attribute, returning a distribution over an integer scale.

    Args:
        df: Polars or Pandas DataFrame.
        column_name: Column containing text to rate.
        attribute: What to rate (e.g. ``"populism"``).
        model: Model from :func:`~prism.load_model`.
        attribute_description: Optional longer description of the attribute.
        scale_min: Minimum of the rating scale (inclusive).
        scale_max: Maximum of the rating scale (inclusive).
        use_reasoning: Allow chain-of-thought before answering.
        max_thinking_tokens: Max tokens for thinking phase.
        additional_instructions: Appended to system prompt.
        context: Optional context string or per-row list.
        random_seed: Seed for reproducibility.
        save_dir: Not yet implemented.

    Returns:
        Input DataFrame with added columns: ``prob_{i}`` for each integer,
        ``expected_value``, ``std_dev``, ``mode``, and ``entropy``.
    """
    texts = _get_column(df, column_name)
    n = len(texts)
    scale_labels = [str(i) for i in range(scale_min, scale_max + 1)]
    logger.info(f"rate: {n} texts, attribute={attribute!r}, scale={scale_min}-{scale_max}")

    builder = PromptBuilder(random_seed=random_seed)
    label_token_sequences = model.tokenize_labels(scale_labels)
    computer = LabelProbabilityComputer(
        label_token_sequences, model.backend, decode=model.decode
    )

    contexts = _resolve_contexts(context, n)
    all_probs: List[Dict[str, float]] = []

    for i, text in enumerate(texts):
        system_msg, user_msg = builder.render_rate(
            text=text,
            attribute=attribute,
            attribute_description=attribute_description,
            scale_min=scale_min,
            scale_max=scale_max,
            context=contexts[i],
            additional_instructions=additional_instructions,
        )
        prompt_tokens = model.tokenize_prompt(system_msg, user_msg)

        if use_reasoning and model.can_reason:
            result = computer.compute_probabilities_with_cot(
                prompt_tokens, model.reasoning_prefix_tokens, max_thinking_tokens
            )
            probs = result.probabilities
        else:
            direct_tokens = _direct_prompt_tokens(prompt_tokens, model)
            probs = computer.compute_probabilities(direct_tokens)

        all_probs.append(probs)

        if (i + 1) % 100 == 0 or i == n - 1:
            logger.info(f"rate: processed {i + 1}/{n}")

    # Build output columns
    columns: Dict[str, list] = {}
    for label in scale_labels:
        columns[f"prob_{label}"] = [p[label] for p in all_probs]

    # Summary statistics
    scale_values = list(range(scale_min, scale_max + 1))
    expected_values = []
    std_devs = []
    modes = []

    for probs in all_probs:
        ev = sum(v * probs[str(v)] for v in scale_values)
        var = sum((v - ev) ** 2 * probs[str(v)] for v in scale_values)
        mode_val = max(scale_values, key=lambda v: probs[str(v)])
        expected_values.append(ev)
        std_devs.append(math.sqrt(var))
        modes.append(mode_val)

    columns["expected_value"] = expected_values
    columns["std_dev"] = std_devs
    columns["mode"] = modes
    columns["entropy"] = [_entropy(p) for p in all_probs]

    return _add_columns(df, columns)


# ---------------------------------------------------------------------------
# binary_classify
# ---------------------------------------------------------------------------


def binary_classify(
    df,
    column_name: str,
    labels: Dict[str, Optional[str]],
    model: Model,
    *,
    use_reasoning: bool = False,
    max_thinking_tokens: int = 2048,
    additional_instructions: Optional[str] = None,
    context: Optional[Union[str, List[Optional[str]]]] = None,
    random_seed: Optional[int] = None,
    save_dir: Optional[str] = None,
):
    """Evaluate independent true/false for each label.

    Args:
        df: Polars or Pandas DataFrame.
        column_name: Column containing text to evaluate.
        labels: ``{label_name: description}`` dict.  Description can be
            ``None``.
        model: Model from :func:`~prism.load_model`.
        use_reasoning: Allow chain-of-thought before answering.
        max_thinking_tokens: Max tokens for thinking phase.
        additional_instructions: Appended to system prompt.
        context: Optional context string or per-row list.
        random_seed: Seed for reproducibility.
        save_dir: Not yet implemented.

    Returns:
        Input DataFrame with added columns: ``prob_true_{label}`` and
        ``predicted_{label}`` (bool) for each label.
    """
    texts = _get_column(df, column_name)
    n = len(texts)
    logger.info(f"binary_classify: {n} texts, {len(labels)} labels")

    builder = PromptBuilder(random_seed=random_seed)
    binary_labels = ["true", "false"]
    label_token_sequences = model.tokenize_labels(binary_labels)
    computer = LabelProbabilityComputer(
        label_token_sequences, model.backend, decode=model.decode
    )

    contexts = _resolve_contexts(context, n)
    columns: Dict[str, list] = {}

    for label_name, label_description in labels.items():
        prob_trues: List[float] = []
        logger.info(f"binary_classify: starting label {label_name!r}")

        for i, text in enumerate(texts):
            system_msg, user_msg = builder.render_binary_classify(
                text=text,
                label=label_name,
                label_description=label_description,
                context=contexts[i],
                additional_instructions=additional_instructions,
            )
            prompt_tokens = model.tokenize_prompt(system_msg, user_msg)

            if use_reasoning and model.can_reason:
                result = computer.compute_probabilities_with_cot(
                    prompt_tokens, model.reasoning_prefix_tokens, max_thinking_tokens
                )
                probs = result.probabilities
            else:
                direct_tokens = _direct_prompt_tokens(prompt_tokens, model)
                probs = computer.compute_probabilities(direct_tokens)

            prob_trues.append(probs["true"])

            if (i + 1) % 100 == 0 or i == n - 1:
                logger.info(f"binary_classify [{label_name}]: processed {i + 1}/{n}")

        columns[f"prob_true_{label_name}"] = prob_trues
        columns[f"predicted_{label_name}"] = [p > 0.5 for p in prob_trues]

    return _add_columns(df, columns)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _direct_prompt_tokens(prompt_tokens: List[int], model: Model) -> List[int]:
    """Append reasoning prefix to skip thinking phase on reasoning models.

    For models that support chain-of-thought, the generation prompt
    (e.g. ``<|start|>assistant``) puts the model into thinking mode.
    Appending the reasoning prefix (e.g. ``<|channel|>final<|message|>``)
    tells the model to skip thinking and answer directly.

    For models without a reasoning prefix, returns prompt_tokens unchanged.
    """
    if model.reasoning_prefix_tokens:
        return prompt_tokens + model.reasoning_prefix_tokens
    return prompt_tokens


def _resolve_contexts(
    context: Optional[Union[str, List[Optional[str]]]], n: int
) -> List[Optional[str]]:
    """Normalize context into a per-row list."""
    if context is None:
        return [None] * n
    if isinstance(context, str):
        return [context] * n
    if len(context) != n:
        raise ValueError(f"context list length ({len(context)}) != number of rows ({n})")
    return context
