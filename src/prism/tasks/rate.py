"""Rate task implementation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

from prism.core.prompt_cache import CascadingCache
from prism.model import Model
from prism.prompts.templates import PromptBuilder
from prism.tasks.shared import (
    add_columns,
    build_probability_computer,
    direct_prompt_tokens,
    effective_prompt_tokens,
    entropy,
    get_column,
    get_constant_context,
    is_context_constant,
    normalize_named_spec,
    resolve_contexts,
)
from prism.utils import get_logger

logger = get_logger(__name__)


class Rate:
    """Execute the integer rating task."""

    def __init__(
        self,
        df,
        column_name: str,
        attributes: Union[str, List[str], Dict[str, Optional[str]]],
        model: Model,
        *,
        scale_min: int = 0,
        scale_max: int = 100,
        use_reasoning: bool = False,
        max_thinking_tokens: int = 2048,
        additional_instructions: Optional[str] = None,
        context: Optional[Union[str, List[Optional[str]]]] = None,
        random_seed: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        self.df = df
        self.column_name = column_name
        self.attributes, self.attribute_descriptions = normalize_named_spec(
            attributes,
            argument_name="attributes",
            allow_single_string=True,
        )
        self.model = model
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.use_reasoning = use_reasoning
        self.max_thinking_tokens = max_thinking_tokens
        self.additional_instructions = additional_instructions
        self.context = context
        self.random_seed = random_seed
        self.save_dir = save_dir
        self.builder = PromptBuilder(random_seed=random_seed)
        self.scale_labels = [str(value) for value in range(scale_min, scale_max + 1)]

    @property
    def reasoning_active(self) -> bool:
        """Return whether reasoning mode will be used for this run."""
        return self.use_reasoning and self.model.can_reason

    def run(self):
        """Run the task and return the input DataFrame with result columns."""
        texts = get_column(self.df, self.column_name)
        contexts = resolve_contexts(self.context, len(texts))
        logger.info(
            "rate: %s texts, %s attributes, scale=%s-%s",
            len(texts),
            len(self.attributes),
            self.scale_min,
            self.scale_max,
        )

        columns: Dict[str, list] = {}
        for attribute in self.attributes:
            attribute_description = None
            if self.attribute_descriptions is not None:
                attribute_description = self.attribute_descriptions.get(attribute)

            attribute_columns = self._run_attribute(
                texts,
                contexts,
                attribute,
                attribute_description,
            )

            if len(self.attributes) == 1:
                columns.update(attribute_columns)
            else:
                columns.update(self._prefix_attribute_columns(attribute, attribute_columns))

        return add_columns(self.df, columns)

    def _run_attribute(
        self,
        texts: List[str],
        contexts: List[Optional[str]],
        attribute: str,
        attribute_description: Optional[str],
    ) -> Dict[str, list]:
        system_msg = PromptBuilder.render_rate_system(
            attribute=attribute,
            attribute_description=attribute_description,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            additional_instructions=self.additional_instructions,
        )
        constant_context = get_constant_context(contexts) if is_context_constant(contexts) else None
        probe_user_msg = PromptBuilder.render_rate_user("", context=constant_context)
        computer, n_absorbed = build_probability_computer(
            self.model,
            self.scale_labels,
            system_msg,
            probe_user_msg,
        )

        all_probs: List[Optional[Dict[str, float]]] = [None] * len(texts)
        all_thinking: List[Optional[str]] = [None] * len(texts)

        if self.reasoning_active:
            self._run_reasoning(
                texts,
                contexts,
                attribute,
                attribute_description,
                system_msg,
                computer,
                n_absorbed,
                all_probs,
                all_thinking,
            )
        else:
            self._run_direct(texts, contexts, system_msg, computer, n_absorbed, all_probs)

        columns: Dict[str, list] = {}
        for label in self.scale_labels:
            columns[f"prob_{label}"] = [probabilities[label] for probabilities in all_probs]

        scale_values = list(range(self.scale_min, self.scale_max + 1))
        expected_values = []
        std_devs = []
        modes = []

        for probabilities in all_probs:
            expected_value = sum(value * probabilities[str(value)] for value in scale_values)
            variance = sum((value - expected_value) ** 2 * probabilities[str(value)] for value in scale_values)
            mode = max(scale_values, key=lambda value: probabilities[str(value)])
            expected_values.append(expected_value)
            std_devs.append(math.sqrt(variance))
            modes.append(mode)

        columns["expected_value"] = expected_values
        columns["std_dev"] = std_devs
        columns["mode"] = modes
        columns["entropy"] = [entropy(probabilities) for probabilities in all_probs]
        if self.reasoning_active:
            columns["thinking_text"] = all_thinking

        return columns

    def _run_reasoning(
        self,
        texts: List[str],
        contexts: List[Optional[str]],
        attribute: str,
        attribute_description: Optional[str],
        system_msg: str,
        computer,
        n_absorbed: int,
        all_probs: List[Optional[Dict[str, float]]],
        all_thinking: List[Optional[str]],
    ) -> None:
        for index, text in enumerate(texts):
            _, user_msg = self.builder.render_rate(
                text=text,
                attribute=attribute,
                attribute_description=attribute_description,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
                context=contexts[index],
                additional_instructions=self.additional_instructions,
            )
            prompt_tokens = self.model.tokenize_prompt(system_msg, user_msg)
            result = computer.compute_probabilities_with_cot(
                prompt_tokens,
                self.model.think_end_tokens,
                self.max_thinking_tokens,
                n_absorbed=n_absorbed,
            )
            all_probs[index] = result.probabilities
            all_thinking[index] = result.thinking_text

            if (index + 1) % 100 == 0 or index == len(texts) - 1:
                logger.info("rate: processed %s/%s", index + 1, len(texts))

    def _run_direct(
        self,
        texts: List[str],
        contexts: List[Optional[str]],
        system_msg: str,
        computer,
        n_absorbed: int,
        all_probs: List[Optional[Dict[str, float]]],
    ) -> None:
        context_is_constant = is_context_constant(contexts)
        constant_context = get_constant_context(contexts) if context_is_constant else None
        render_user = PromptBuilder.render_rate_user

        cache = CascadingCache(
            backend=self.model.backend,
            model=self.model,
            render_system_fn=lambda _ordering: system_msg,
            render_user_fn=render_user,
            labels=self.scale_labels,
            context_is_constant=context_is_constant,
            build_level0=False,
        )
        cache.set_fixed_prefix(system_msg, constant_context=constant_context)

        for index, text in enumerate(texts):
            user_msg = render_user(text, context=contexts[index])
            prompt_tokens = self.model.tokenize_prompt(system_msg, user_msg)
            prompt_tokens = direct_prompt_tokens(prompt_tokens, self.model)
            prompt_tokens = effective_prompt_tokens(prompt_tokens, n_absorbed)

            logits, row_cache = cache.forward_row(prompt_tokens)
            all_probs[index] = computer.compute_probabilities_cached(logits, row_cache)

            if (index + 1) % 100 == 0 or index == len(texts) - 1:
                logger.info("rate: processed %s/%s", index + 1, len(texts))

    @staticmethod
    def _prefix_attribute_columns(attribute: str, columns: Dict[str, list]) -> Dict[str, list]:
        """Prefix per-attribute outputs when multiple attributes are rated."""
        prefixed: Dict[str, list] = {}
        for name, values in columns.items():
            if name.startswith("prob_"):
                prefixed[f"prob_{attribute}_{name.removeprefix('prob_')}"] = values
            else:
                prefixed[f"{name}_{attribute}"] = values
        return prefixed
