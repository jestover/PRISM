"""Public API wrappers around the task implementations."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from prism.model import Model
from prism.tasks import Classify, Label, Rate


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
):
    """Classify text into one of several mutually exclusive labels."""
    return Classify(
        df,
        column_name,
        labels,
        model,
        label_descriptions=label_descriptions,
        use_reasoning=use_reasoning,
        max_thinking_tokens=max_thinking_tokens,
        additional_instructions=additional_instructions,
        context=context,
        shuffle_labels=shuffle_labels,
        random_seed=random_seed,
        save_dir=save_dir,
    ).run()


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
    """Rate text on an attribute, returning a distribution over an integer scale."""
    return Rate(
        df,
        column_name,
        attribute,
        model,
        attribute_description=attribute_description,
        scale_min=scale_min,
        scale_max=scale_max,
        use_reasoning=use_reasoning,
        max_thinking_tokens=max_thinking_tokens,
        additional_instructions=additional_instructions,
        context=context,
        random_seed=random_seed,
        save_dir=save_dir,
    ).run()


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
    """Evaluate independent true/false applicability for each label."""
    return Label(
        df,
        column_name,
        labels,
        model,
        use_reasoning=use_reasoning,
        max_thinking_tokens=max_thinking_tokens,
        additional_instructions=additional_instructions,
        context=context,
        random_seed=random_seed,
        save_dir=save_dir,
    ).run()
