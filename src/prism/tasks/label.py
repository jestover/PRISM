"""Label task implementation."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from prism.core.prompt_cache import CascadingCache
from prism.model import Model
from prism.prompts.templates import PromptBuilder
from prism.tasks.shared import (
    add_columns,
    build_probability_computer,
    direct_prompt_tokens,
    effective_prompt_tokens,
    get_column,
    get_constant_context,
    is_context_constant,
    normalize_named_spec,
    resolve_contexts,
)
from prism.utils import get_logger

logger = get_logger(__name__)


class Label:
    """Execute the independent true/false label task."""

    def __init__(
        self,
        df,
        column_name: str,
        labels: Union[str, List[str], Dict[str, Optional[str]]],
        model: Model,
        *,
        use_reasoning: bool = False,
        max_thinking_tokens: int = 2048,
        additional_instructions: Optional[str] = None,
        context: Optional[Union[str, List[Optional[str]]]] = None,
        random_seed: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        self.df = df
        self.column_name = column_name
        self.labels, self.label_descriptions = normalize_named_spec(
            labels,
            argument_name="labels",
            allow_single_string=True,
        )
        self.model = model
        self.use_reasoning = use_reasoning
        self.max_thinking_tokens = max_thinking_tokens
        self.additional_instructions = additional_instructions
        self.context = context
        self.random_seed = random_seed
        self.save_dir = save_dir
        self.builder = PromptBuilder(random_seed=random_seed)
        self.binary_labels = ["true", "false"]

    @property
    def reasoning_active(self) -> bool:
        """Return whether reasoning mode will be used for this run."""
        return self.use_reasoning and self.model.can_reason

    def run(self):
        """Run the task and return the input DataFrame with result columns."""
        texts = get_column(self.df, self.column_name)
        contexts = resolve_contexts(self.context, len(texts))
        logger.info("label: %s texts, %s labels", len(texts), len(self.labels))

        columns: Dict[str, list] = {}
        for label_name in self.labels:
            label_description = None
            if self.label_descriptions is not None:
                label_description = self.label_descriptions.get(label_name)
            probability_true: List[float] = []
            thinking_texts: List[Optional[str]] = []
            logger.info("label: starting label %r", label_name)

            system_msg = PromptBuilder.render_label_system(
                label=label_name,
                label_description=label_description,
                additional_instructions=self.additional_instructions,
            )
            constant_context = get_constant_context(contexts) if is_context_constant(contexts) else None
            probe_user_msg = PromptBuilder.render_label_user("", context=constant_context)
            computer, n_absorbed = build_probability_computer(
                self.model,
                self.binary_labels,
                system_msg,
                probe_user_msg,
            )

            if self.reasoning_active:
                self._run_reasoning(
                    texts,
                    contexts,
                    label_name,
                    label_description,
                    system_msg,
                    computer,
                    n_absorbed,
                    probability_true,
                    thinking_texts,
                )
            else:
                self._run_direct(
                    texts,
                    contexts,
                    label_name,
                    system_msg,
                    computer,
                    n_absorbed,
                    probability_true,
                )
                thinking_texts = [None] * len(texts)

            columns[f"prob_true_{label_name}"] = probability_true
            columns[f"predicted_{label_name}"] = [probability > 0.5 for probability in probability_true]
            if self.reasoning_active:
                columns[f"thinking_text_{label_name}"] = thinking_texts

        return add_columns(self.df, columns)

    def _run_reasoning(
        self,
        texts: List[str],
        contexts: List[Optional[str]],
        label_name: str,
        label_description: Optional[str],
        system_msg: str,
        computer,
        n_absorbed: int,
        probability_true: List[float],
        thinking_texts: List[Optional[str]],
    ) -> None:
        for index, text in enumerate(texts):
            _, user_msg = self.builder.render_label(
                text=text,
                label=label_name,
                label_description=label_description,
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
            probability_true.append(result.probabilities["true"])
            thinking_texts.append(result.thinking_text)

            if (index + 1) % 100 == 0 or index == len(texts) - 1:
                logger.info("label [%s]: processed %s/%s", label_name, index + 1, len(texts))

    def _run_direct(
        self,
        texts: List[str],
        contexts: List[Optional[str]],
        label_name: str,
        system_msg: str,
        computer,
        n_absorbed: int,
        probability_true: List[float],
    ) -> None:
        context_is_constant = is_context_constant(contexts)
        constant_context = get_constant_context(contexts) if context_is_constant else None
        render_user = PromptBuilder.render_label_user

        cache = CascadingCache(
            backend=self.model.backend,
            model=self.model,
            render_system_fn=lambda _ordering: system_msg,
            render_user_fn=render_user,
            labels=self.binary_labels,
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
            probabilities = computer.compute_probabilities_cached(logits, row_cache)
            probability_true.append(probabilities["true"])

            if (index + 1) % 100 == 0 or index == len(texts) - 1:
                logger.info("label [%s]: processed %s/%s", label_name, index + 1, len(texts))
