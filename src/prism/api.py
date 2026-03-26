"""High-level API: classify, rate, label."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

from prism.core.label_probs import LabelProbabilityComputer
from prism.core.prompt_cache import CascadingCache
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
        ``predicted_class``, ``max_prob``, ``entropy``, and ``thinking_text``
        (when ``use_reasoning=True``).
    """
    texts = _get_column(df, column_name)
    n = len(texts)
    reasoning_active = use_reasoning and model.can_reason
    logger.info(f"classify: {n} texts, {len(labels)} labels, use_reasoning={reasoning_active}")

    builder = PromptBuilder(random_seed=random_seed)

    contexts = _resolve_contexts(context, n)
    all_probs: List[Optional[Dict[str, float]]] = [None] * n
    all_thinking: List[Optional[str]] = [None] * n

    if reasoning_active:
        # --- COT path: uncached (generate_until handles its own cache) ---
        for i, text in enumerate(texts):
            system_msg, user_msg = builder.render_classify(
                text=text,
                labels=labels,
                label_descriptions=label_descriptions,
                context=contexts[i],
                additional_instructions=additional_instructions,
                shuffle=shuffle_labels,
            )
            computer, n_absorbed = _build_probability_computer(
                model, labels, system_msg, user_msg
            )
            prompt_tokens = model.tokenize_prompt(system_msg, user_msg)
            result = computer.compute_probabilities_with_cot(
                prompt_tokens,
                model.think_end_tokens,
                max_thinking_tokens,
                n_absorbed=n_absorbed,
            )
            all_probs[i] = result.probabilities
            all_thinking[i] = result.thinking_text

            if (i + 1) % 100 == 0 or i == n - 1:
                logger.info(f"classify: processed {i + 1}/{n}")
    else:
        # --- Non-COT path: CascadingCache with label ordering batching ---

        # Pre-generate orderings (same RNG sequence = reproducible)
        orderings: List[Tuple[str, ...]] = []
        for _ in range(n):
            ordered = list(labels)
            if shuffle_labels:
                builder.rng.shuffle(ordered)
            orderings.append(tuple(ordered))

        # Group rows by ordering
        groups: Dict[Tuple[str, ...], List[int]] = {}
        for i, ordering in enumerate(orderings):
            if ordering not in groups:
                groups[ordering] = []
            groups[ordering].append(i)

        logger.info(f"classify: {len(groups)} ordering groups for {n} rows")

        # Determine if context is constant
        ctx_is_constant = _is_context_constant(contexts)
        constant_ctx = _get_constant_context(contexts) if ctx_is_constant else None

        # Build render functions for CascadingCache
        def render_sys(ordering: Tuple[str, ...]) -> str:
            return builder.render_classify_system(
                ordering,
                label_descriptions=label_descriptions,
                additional_instructions=additional_instructions,
            )

        render_usr = PromptBuilder.render_classify_user

        # Build CascadingCache (Level 0)
        cascading = CascadingCache(
            backend=model.backend,
            model=model,
            render_system_fn=render_sys,
            render_user_fn=render_usr,
            labels=labels,
            context_is_constant=ctx_is_constant,
            build_level0=shuffle_labels,
        )

        processed = 0
        for ordering, row_indices in groups.items():
            # Level 1: cache ordering-specific prefix
            cascading.set_ordering(ordering, constant_context=constant_ctx)
            system_msg = render_sys(ordering)
            probe_user_msg = render_usr("", context=constant_ctx if ctx_is_constant else None)
            computer, n_absorbed = _build_probability_computer(
                model, list(ordering), system_msg, probe_user_msg
            )

            for i in row_indices:
                usr_msg = render_usr(texts[i], context=contexts[i])
                prompt_tokens = model.tokenize_prompt(system_msg, usr_msg)
                direct_tokens = _direct_prompt_tokens(prompt_tokens, model)
                effective_tokens = _effective_prompt_tokens(direct_tokens, n_absorbed)

                # Level 2: forward row-specific suffix
                logits, row_cache = cascading.forward_row(effective_tokens)

                # Level 3: compute probabilities at branch points
                all_probs[i] = computer.compute_probabilities_cached(
                    logits, row_cache
                )

                processed += 1
                if processed % 100 == 0 or processed == n:
                    logger.info(f"classify: processed {processed}/{n}")

    # Build output columns
    columns: Dict[str, list] = {}
    for label in labels:
        columns[f"prob_{label}"] = [p[label] for p in all_probs]  # type: ignore[index]

    columns["predicted_class"] = [max(p, key=p.get) for p in all_probs]  # type: ignore[arg-type]
    columns["max_prob"] = [max(p.values()) for p in all_probs]  # type: ignore[union-attr]
    columns["entropy"] = [_entropy(p) for p in all_probs]  # type: ignore[arg-type]
    if reasoning_active:
        columns["thinking_text"] = all_thinking

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
        ``expected_value``, ``std_dev``, ``mode``, ``entropy``, and
        ``thinking_text`` (when ``use_reasoning=True``).
    """
    texts = _get_column(df, column_name)
    n = len(texts)
    reasoning_active = use_reasoning and model.can_reason
    scale_labels = [str(i) for i in range(scale_min, scale_max + 1)]
    logger.info(f"rate: {n} texts, attribute={attribute!r}, scale={scale_min}-{scale_max}")

    builder = PromptBuilder(random_seed=random_seed)

    contexts = _resolve_contexts(context, n)
    all_probs: List[Optional[Dict[str, float]]] = [None] * n
    all_thinking: List[Optional[str]] = [None] * n
    system_msg = PromptBuilder.render_rate_system(
        attribute=attribute,
        attribute_description=attribute_description,
        scale_min=scale_min,
        scale_max=scale_max,
        additional_instructions=additional_instructions,
    )
    probe_user_msg = PromptBuilder.render_rate_user(
        "",
        context=_get_constant_context(contexts) if _is_context_constant(contexts) else None,
    )
    computer, n_absorbed = _build_probability_computer(
        model, scale_labels, system_msg, probe_user_msg
    )

    if reasoning_active:
        # --- COT path: uncached ---
        for i, text in enumerate(texts):
            _, user_msg = builder.render_rate(
                text=text,
                attribute=attribute,
                attribute_description=attribute_description,
                scale_min=scale_min,
                scale_max=scale_max,
                context=contexts[i],
                additional_instructions=additional_instructions,
            )
            prompt_tokens = model.tokenize_prompt(system_msg, user_msg)
            result = computer.compute_probabilities_with_cot(
                prompt_tokens,
                model.think_end_tokens,
                max_thinking_tokens,
                n_absorbed=n_absorbed,
            )
            all_probs[i] = result.probabilities
            all_thinking[i] = result.thinking_text

            if (i + 1) % 100 == 0 or i == n - 1:
                logger.info(f"rate: processed {i + 1}/{n}")
    else:
        # --- Non-COT path: CascadingCache (fixed prefix, no label shuffling) ---
        ctx_is_constant = _is_context_constant(contexts)
        constant_ctx = _get_constant_context(contexts) if ctx_is_constant else None

        render_usr = PromptBuilder.render_rate_user

        cascading = CascadingCache(
            backend=model.backend,
            model=model,
            render_system_fn=lambda _ordering: system_msg,
            render_user_fn=render_usr,
            labels=scale_labels,
            context_is_constant=ctx_is_constant,
            build_level0=False,
        )
        cascading.set_fixed_prefix(system_msg, constant_context=constant_ctx)

        for i, text in enumerate(texts):
            usr_msg = render_usr(text, context=contexts[i])
            prompt_tokens = model.tokenize_prompt(system_msg, usr_msg)
            direct_tokens = _direct_prompt_tokens(prompt_tokens, model)
            effective_tokens = _effective_prompt_tokens(direct_tokens, n_absorbed)

            logits, row_cache = cascading.forward_row(effective_tokens)
            all_probs[i] = computer.compute_probabilities_cached(
                logits, row_cache
            )

            if (i + 1) % 100 == 0 or i == n - 1:
                logger.info(f"rate: processed {i + 1}/{n}")

    # Build output columns
    columns: Dict[str, list] = {}
    for label in scale_labels:
        columns[f"prob_{label}"] = [p[label] for p in all_probs]  # type: ignore[index]

    # Summary statistics
    scale_values = list(range(scale_min, scale_max + 1))
    expected_values = []
    std_devs = []
    modes = []

    for probs in all_probs:
        ev = sum(v * probs[str(v)] for v in scale_values)  # type: ignore[index]
        var = sum((v - ev) ** 2 * probs[str(v)] for v in scale_values)  # type: ignore[index]
        mode_val = max(scale_values, key=lambda v: probs[str(v)])  # type: ignore[index]
        expected_values.append(ev)
        std_devs.append(math.sqrt(var))
        modes.append(mode_val)

    columns["expected_value"] = expected_values
    columns["std_dev"] = std_devs
    columns["mode"] = modes
    columns["entropy"] = [_entropy(p) for p in all_probs]  # type: ignore[arg-type]
    if reasoning_active:
        columns["thinking_text"] = all_thinking

    return _add_columns(df, columns)


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def label(
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
    """Evaluate independent true/false applicability for each label.

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
        Input DataFrame with added columns: ``prob_true_{label}``,
        ``predicted_{label}`` (bool), and ``thinking_text_{label}``
        (when ``use_reasoning=True``) for each label.
    """
    texts = _get_column(df, column_name)
    n = len(texts)
    reasoning_active = use_reasoning and model.can_reason
    logger.info(f"label: {n} texts, {len(labels)} labels")

    builder = PromptBuilder(random_seed=random_seed)
    binary_labels = ["true", "false"]

    contexts = _resolve_contexts(context, n)
    columns: Dict[str, list] = {}

    for label_name, label_description in labels.items():
        prob_trues: List[float] = []
        thinking_texts: List[Optional[str]] = []
        logger.info(f"label: starting label {label_name!r}")
        system_msg = PromptBuilder.render_label_system(
            label=label_name,
            label_description=label_description,
            additional_instructions=additional_instructions,
        )
        computer, n_absorbed = _build_probability_computer(
            model,
            binary_labels,
            system_msg,
            PromptBuilder.render_label_user(
                "",
                context=_get_constant_context(contexts) if _is_context_constant(contexts) else None,
            ),
        )

        if reasoning_active:
            # --- COT path: uncached ---
            for i, text in enumerate(texts):
                _, user_msg = builder.render_label(
                    text=text,
                    label=label_name,
                    label_description=label_description,
                    context=contexts[i],
                    additional_instructions=additional_instructions,
                )
                prompt_tokens = model.tokenize_prompt(system_msg, user_msg)
                result = computer.compute_probabilities_with_cot(
                    prompt_tokens,
                    model.think_end_tokens,
                    max_thinking_tokens,
                    n_absorbed=n_absorbed,
                )
                prob_trues.append(result.probabilities["true"])
                thinking_texts.append(result.thinking_text)

                if (i + 1) % 100 == 0 or i == n - 1:
                    logger.info(
                        f"label [{label_name}]: processed {i + 1}/{n}"
                    )
        else:
            # --- Non-COT path: CascadingCache (one per label) ---
            ctx_is_constant = _is_context_constant(contexts)
            constant_ctx = (
                _get_constant_context(contexts) if ctx_is_constant else None
            )

            render_usr = PromptBuilder.render_label_user

            cascading = CascadingCache(
                backend=model.backend,
                model=model,
                render_system_fn=lambda _ordering: system_msg,
                render_user_fn=render_usr,
                labels=binary_labels,
                context_is_constant=ctx_is_constant,
                build_level0=False,
            )
            cascading.set_fixed_prefix(system_msg, constant_context=constant_ctx)

            for i, text in enumerate(texts):
                usr_msg = render_usr(text, context=contexts[i])
                prompt_tokens = model.tokenize_prompt(system_msg, usr_msg)
                direct_tokens = _direct_prompt_tokens(prompt_tokens, model)
                effective_tokens = _effective_prompt_tokens(direct_tokens, n_absorbed)

                logits, row_cache = cascading.forward_row(effective_tokens)
                probs = computer.compute_probabilities_cached(logits, row_cache)
                prob_trues.append(probs["true"])
                thinking_texts.append(None)

                if (i + 1) % 100 == 0 or i == n - 1:
                    logger.info(
                        f"label [{label_name}]: processed {i + 1}/{n}"
                    )

        columns[f"prob_true_{label_name}"] = prob_trues
        columns[f"predicted_{label_name}"] = [p > 0.5 for p in prob_trues]
        if reasoning_active:
            columns[f"thinking_text_{label_name}"] = thinking_texts

    return _add_columns(df, columns)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _direct_prompt_tokens(prompt_tokens: List[int], model: Model) -> List[int]:
    """Append think-end tokens to skip thinking phase on reasoning models.

    For models that support chain-of-thought, the generation prompt
    (e.g. ``<|start|>assistant``) puts the model into thinking mode.
    Appending the think-end sequence (e.g. ``<|channel|>final<|message|>``)
    tells the model to skip thinking and answer directly.

    For models without a think-end sequence, returns prompt_tokens unchanged.
    """
    if model.think_end_tokens:
        return prompt_tokens + model.think_end_tokens
    return prompt_tokens


def _build_probability_computer(
    model: Model,
    labels: List[str],
    system_message: str,
    user_message: str,
) -> Tuple[LabelProbabilityComputer, int]:
    """Create a probability computer and absorbed-token count for a prompt shape."""
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


def _effective_prompt_tokens(prompt_tokens: List[int], n_absorbed: int) -> List[int]:
    """Back up absorbed prompt tokens before probability extraction."""
    if n_absorbed <= 0:
        return prompt_tokens
    if n_absorbed >= len(prompt_tokens):
        raise ValueError(
            f"n_absorbed ({n_absorbed}) must be smaller than prompt length ({len(prompt_tokens)})"
        )
    return prompt_tokens[:-n_absorbed]


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


def _is_context_constant(contexts: List[Optional[str]]) -> bool:
    """Check whether all context values are identical."""
    if not contexts:
        return True
    first = contexts[0]
    return all(c == first for c in contexts)


def _get_constant_context(contexts: List[Optional[str]]) -> Optional[str]:
    """Return the constant context value (may be ``None``)."""
    if not contexts:
        return None
    return contexts[0]
