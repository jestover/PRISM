"""Classify task implementation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

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


class Classify:
    """Execute the mutually-exclusive classification task."""

    def __init__(
        self,
        df,
        column_name: str,
        labels: Union[List[str], Dict[str, Optional[str]]],
        model: Model,
        *,
        use_reasoning: bool = False,
        max_thinking_tokens: int = 2048,
        additional_instructions: Optional[str] = None,
        context: Optional[Union[str, List[Optional[str]]]] = None,
        shuffle_labels: bool = True,
        random_seed: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        self.df = df
        self.column_name = column_name
        self.labels, self.label_descriptions = normalize_named_spec(
            labels,
            argument_name="labels",
            allow_single_string=False,
        )
        self.model = model
        self.use_reasoning = use_reasoning
        self.max_thinking_tokens = max_thinking_tokens
        self.additional_instructions = additional_instructions
        self.context = context
        self.shuffle_labels = shuffle_labels
        self.random_seed = random_seed
        self.save_dir = save_dir
        self.builder = PromptBuilder(random_seed=random_seed)

    @property
    def reasoning_active(self) -> bool:
        """Return whether reasoning mode will be used for this run."""
        return self.use_reasoning and self.model.can_reason

    def run(self):
        """Run the task and return the input DataFrame with result columns."""
        texts = get_column(self.df, self.column_name)
        contexts = resolve_contexts(self.context, len(texts))
        logger.info(
            "classify: %s texts, %s labels, use_reasoning=%s",
            len(texts),
            len(self.labels),
            self.reasoning_active,
        )

        all_probs: List[Optional[Dict[str, float]]] = [None] * len(texts)
        all_thinking: List[Optional[str]] = [None] * len(texts)

        if self.reasoning_active:
            self._run_reasoning(texts, contexts, all_probs, all_thinking)
        else:
            self._run_direct(texts, contexts, all_probs)

        columns: Dict[str, list] = {}
        for label in self.labels:
            columns[f"prob_{label}"] = [probabilities[label] for probabilities in all_probs]

        columns["predicted_class"] = [max(probabilities, key=probabilities.get) for probabilities in all_probs]
        columns["max_prob"] = [max(probabilities.values()) for probabilities in all_probs]
        columns["entropy"] = [entropy(probabilities) for probabilities in all_probs]
        if self.reasoning_active:
            columns["thinking_text"] = all_thinking

        return add_columns(self.df, columns)

    def _run_reasoning(
        self,
        texts: List[str],
        contexts: List[Optional[str]],
        all_probs: List[Optional[Dict[str, float]]],
        all_thinking: List[Optional[str]],
    ) -> None:
        for index, text in enumerate(texts):
            system_msg, user_msg = self.builder.render_classify(
                text=text,
                labels=self.labels,
                label_descriptions=self.label_descriptions,
                context=contexts[index],
                additional_instructions=self.additional_instructions,
                shuffle=self.shuffle_labels,
            )
            computer, n_absorbed = build_probability_computer(
                self.model,
                self.labels,
                system_msg,
                user_msg,
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
                logger.info("classify: processed %s/%s", index + 1, len(texts))

    def _run_direct(
        self,
        texts: List[str],
        contexts: List[Optional[str]],
        all_probs: List[Optional[Dict[str, float]]],
    ) -> None:
        orderings = self._generate_orderings(len(texts))
        groups: Dict[Tuple[str, ...], List[int]] = {}
        for index, ordering in enumerate(orderings):
            groups.setdefault(ordering, []).append(index)

        logger.info("classify: %s ordering groups for %s rows", len(groups), len(texts))

        context_is_constant = is_context_constant(contexts)
        constant_context = get_constant_context(contexts) if context_is_constant else None

        def render_system(ordering: Tuple[str, ...]) -> str:
            return self.builder.render_classify_system(
                ordering,
                label_descriptions=self.label_descriptions,
                additional_instructions=self.additional_instructions,
            )

        render_user = PromptBuilder.render_classify_user
        cache = CascadingCache(
            backend=self.model.backend,
            model=self.model,
            render_system_fn=render_system,
            render_user_fn=render_user,
            labels=self.labels,
            context_is_constant=context_is_constant,
            build_level0=self.shuffle_labels,
        )

        processed = 0
        for ordering, row_indices in groups.items():
            cache.set_ordering(ordering, constant_context=constant_context)
            system_msg = render_system(ordering)
            probe_user_msg = render_user("", context=constant_context if context_is_constant else None)
            computer, n_absorbed = build_probability_computer(
                self.model,
                list(ordering),
                system_msg,
                probe_user_msg,
            )

            for row_index in row_indices:
                user_msg = render_user(texts[row_index], context=contexts[row_index])
                prompt_tokens = self.model.tokenize_prompt(system_msg, user_msg)
                prompt_tokens = direct_prompt_tokens(prompt_tokens, self.model)
                prompt_tokens = effective_prompt_tokens(prompt_tokens, n_absorbed)

                logits, row_cache = cache.forward_row(prompt_tokens)
                all_probs[row_index] = computer.compute_probabilities_cached(logits, row_cache)

                processed += 1
                if processed % 100 == 0 or processed == len(texts):
                    logger.info("classify: processed %s/%s", processed, len(texts))

    def _generate_orderings(self, n_rows: int) -> List[Tuple[str, ...]]:
        orderings: List[Tuple[str, ...]] = []
        for _ in range(n_rows):
            ordered = list(self.labels)
            if self.shuffle_labels:
                self.builder.rng.shuffle(ordered)
            orderings.append(tuple(ordered))
        return orderings
