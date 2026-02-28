"""Prompt templates and the PromptBuilder.

Templates use ``.format()`` — no Jinja2 dependency.  They produce message
*content* only; the model's chat template handles structural formatting.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """\
You are classifying text. For the text provided, select the single most \
appropriate label from the list below.

{label_descriptions_block}\
{additional_instructions_block}\
Respond with ONLY the label name, exactly as written. No explanation."""

CLASSIFY_USER = """\
{context_block}\
Text: {text}

Label: """

# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------

RATE_SYSTEM = """\
You are rating text on a specific attribute. For the text provided, assign a \
single integer rating on a {scale_min}-{scale_max} scale.

Attribute: {attribute}
{attribute_description_block}\
Rating scale:
{scale_min} = absent or not at all
{scale_max} = extreme or overwhelmingly present
Use the full range. Do not round to multiples of 5 or 10.
Consider intermediate values (e.g., 19, 67, 32) to capture nuance.

{additional_instructions_block}\
Respond with ONLY the integer. No explanation."""

RATE_USER = """\
{context_block}\
Text: {text}

Rating: """

# ---------------------------------------------------------------------------
# Binary classify
# ---------------------------------------------------------------------------

BINARY_CLASSIFY_SYSTEM = """\
You are evaluating whether a label applies to the provided text.

Label: {label}
{label_description_block}\
{additional_instructions_block}\
Respond with ONLY "true" if the label applies, or "false" if it does not. \
No explanation."""

BINARY_CLASSIFY_USER = """\
{context_block}\
Text: {text}

Applies: """


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Build prompts for each PRISM task type.

    Handles label shuffling (to prevent position bias) and optional blocks
    for descriptions, context, and additional instructions.

    Args:
        random_seed: Seed for deterministic label shuffling.  ``None`` for
            non-deterministic.
    """

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = random.Random(random_seed)

    # -- public API --

    def render_classify(
        self,
        text: str,
        labels: List[str],
        label_descriptions: Optional[Dict[str, str]] = None,
        context: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        shuffle: bool = True,
    ) -> Tuple[str, str]:
        """Render a classify prompt.

        Args:
            text: The text to classify.
            labels: Possible labels.
            label_descriptions: Optional descriptions for each label.
            context: Optional context to include.
            additional_instructions: Extra instructions for the system prompt.
            shuffle: Whether to randomize label order.

        Returns:
            ``(system_message, user_message)``
        """
        ordered = list(labels)
        if shuffle:
            self.rng.shuffle(ordered)

        label_descriptions_block = self._label_descriptions_block(ordered, label_descriptions)
        additional_instructions_block = self._additional_instructions_block(additional_instructions)
        context_block = self._context_block(context)

        system = CLASSIFY_SYSTEM.format(
            label_descriptions_block=label_descriptions_block,
            additional_instructions_block=additional_instructions_block,
        )
        user = CLASSIFY_USER.format(context_block=context_block, text=text)

        return system, user

    def render_rate(
        self,
        text: str,
        attribute: str,
        attribute_description: Optional[str] = None,
        scale_min: int = 0,
        scale_max: int = 100,
        context: Optional[str] = None,
        additional_instructions: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Render a rate prompt.

        Returns:
            ``(system_message, user_message)``
        """
        attribute_description_block = (
            f"Description: {attribute_description}\n\n" if attribute_description else "\n"
        )
        additional_instructions_block = self._additional_instructions_block(additional_instructions)
        context_block = self._context_block(context)

        system = RATE_SYSTEM.format(
            attribute=attribute,
            attribute_description_block=attribute_description_block,
            scale_min=scale_min,
            scale_max=scale_max,
            additional_instructions_block=additional_instructions_block,
        )
        user = RATE_USER.format(context_block=context_block, text=text)

        return system, user

    def render_binary_classify(
        self,
        text: str,
        label: str,
        label_description: Optional[str] = None,
        context: Optional[str] = None,
        additional_instructions: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Render a binary classify prompt.

        Returns:
            ``(system_message, user_message)``
        """
        label_description_block = (
            f"Description: {label_description}\n\n" if label_description else "\n"
        )
        additional_instructions_block = self._additional_instructions_block(additional_instructions)
        context_block = self._context_block(context)

        system = BINARY_CLASSIFY_SYSTEM.format(
            label=label,
            label_description_block=label_description_block,
            additional_instructions_block=additional_instructions_block,
        )
        user = BINARY_CLASSIFY_USER.format(context_block=context_block, text=text)

        return system, user

    # -- helpers --

    @staticmethod
    def _label_descriptions_block(
        labels: List[str],
        descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        if descriptions:
            lines = []
            for label in labels:
                if label in descriptions:
                    lines.append(f"- {label}: {descriptions[label]}")
                else:
                    lines.append(f"- {label}")
            return "Labels:\n" + "\n".join(lines) + "\n\n"
        return "Labels: " + ", ".join(labels) + "\n\n"

    @staticmethod
    def _additional_instructions_block(instructions: Optional[str]) -> str:
        if instructions:
            return instructions.rstrip() + "\n\n"
        return ""

    @staticmethod
    def _context_block(context: Optional[str]) -> str:
        if context:
            return f"Context: {context}\n\n"
        return ""
