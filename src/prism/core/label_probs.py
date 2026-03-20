"""Compute label probabilities only at branching points for efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from prism.core.token_trie import TERMINAL_TOKEN, LabelTokenTrie
from prism.utils import get_logger

if TYPE_CHECKING:
    from prism.backends.base import InferenceBackend

logger = get_logger(__name__)


@dataclass
class COTResult:
    """Result from chain-of-thought probability computation."""

    probabilities: Dict[str, float]
    thinking_text: str
    thinking_tokens: List[int]


class LabelProbabilityComputer:
    """Compute probability for each label by evaluating only at branch points.

    Args:
        label_token_sequences: Mapping of ``{label: [token_ids]}``.
        backend: Inference backend providing ``get_logits``, ``softmax``,
            and ``generate_until``.
        decode: Optional callable to decode token IDs to text (typically
            ``tokenizer.decode``). Required for chain-of-thought.
    """

    def __init__(
        self,
        label_token_sequences: Dict[str, List[int]],
        backend: InferenceBackend,
        decode: Optional[Callable[[List[int]], str]] = None,
    ):
        self.backend = backend
        self.decode = decode
        self.trie = LabelTokenTrie(label_token_sequences)
        logger.info(f"Initialized probability computer for {len(label_token_sequences)} labels")

    def _branch_probabilities(self, logits: Any, branches: Dict[int, List[str]]) -> Dict[int, float]:
        """Compute branch probabilities, including terminal mass when needed."""
        continuation_tokens = [token_id for token_id in branches if token_id != TERMINAL_TOKEN]

        if TERMINAL_TOKEN in branches:
            full_probs = self.backend.softmax(logits)
            continuation_probs: List[float] = []
            if continuation_tokens:
                continuation_probs = full_probs[continuation_tokens].tolist()
            continuation_mass = sum(continuation_probs)
            terminal_prob = max(0.0, 1.0 - continuation_mass)

            branch_probs = {
                token_id: prob
                for token_id, prob in zip(continuation_tokens, continuation_probs)
            }
            branch_probs[TERMINAL_TOKEN] = terminal_prob
            return branch_probs

        masked_logits = logits[continuation_tokens]
        probs = self.backend.softmax(masked_logits).tolist()
        return {
            token_id: prob
            for token_id, prob in zip(continuation_tokens, probs)
        }

    def compute_probabilities(self, prompt_tokens: List[int]) -> Dict[str, float]:
        """Compute P(label) for each label.

        Args:
            prompt_tokens: Tokenized prompt ending right before label generation.

        Returns:
            Mapping of ``{label: probability}``.
        """
        label_probs = {label: 1.0 for label in self.trie.label_sequences}

        for branch_point in self.trie.branch_points:
            current_tokens = prompt_tokens + branch_point.prefix
            logits = self.backend.get_logits(current_tokens)

            for token_id, prob in self._branch_probabilities(logits, branch_point.branches).items():
                for label in branch_point.branches[token_id]:
                    label_probs[label] *= prob

        return label_probs

    def compute_probabilities_cached(
        self,
        prompt_logits: Any,
        prompt_cache: Any,
    ) -> Dict[str, float]:
        """Compute P(label) using cached prompt state (Level 3).

        Instead of reprocessing the full prompt for each branch point
        (as :meth:`compute_probabilities` does), this method:

        1. Uses *prompt_logits* for the root branch point (prefix=[]).
        2. For deeper branch points, forks the nearest cached ancestor
           and forwards only the incremental prefix tokens.

        This is Level 3 of the cascading cache hierarchy.  The caller
        obtains ``prompt_logits`` and ``prompt_cache`` from
        :meth:`~prism.core.prompt_cache.CascadingCache.forward_row`.

        Args:
            prompt_logits: Logits at the final prompt position.
            prompt_cache: KV cache containing the full prompt's state.

        Returns:
            Mapping of ``{label: probability}``.
        """
        label_probs: Dict[str, float] = {
            label: 1.0 for label in self.trie.label_sequences
        }

        # Sort branch points by prefix length for correct tree traversal
        sorted_bps = sorted(
            self.trie.branch_points, key=lambda bp: len(bp.prefix)
        )

        # Cache state at each resolved prefix for parent look-up.
        # key: tuple(prefix), value: (logits, cache)
        prefix_states: Dict[Tuple[int, ...], Tuple[Any, Any]] = {
            (): (prompt_logits, prompt_cache)
        }

        for bp in sorted_bps:
            prefix_key = tuple(bp.prefix)

            if not bp.prefix:
                # Root branch point — use prompt state directly
                logits = prompt_logits
            else:
                # Find the longest cached ancestor prefix
                parent_key: Tuple[int, ...] = ()
                for length in range(len(prefix_key) - 1, -1, -1):
                    candidate = prefix_key[:length]
                    if candidate in prefix_states:
                        parent_key = candidate
                        break

                parent_logits, parent_cache = prefix_states[parent_key]
                incremental = list(prefix_key[len(parent_key) :])

                if incremental:
                    child_cache = self.backend.copy_cache(parent_cache)
                    logits, child_cache = self.backend.forward(
                        incremental, child_cache
                    )
                    prefix_states[prefix_key] = (logits, child_cache)
                else:
                    logits = parent_logits

            # Extract probabilities at this branch point
            for token_id, prob in self._branch_probabilities(logits, bp.branches).items():
                for label in bp.branches[token_id]:
                    label_probs[label] *= prob

        return label_probs

    def compute_probabilities_with_cot(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        max_thinking_tokens: int = 2048,
        use_cache: bool = True,
        n_absorbed: int = 0,
    ) -> COTResult:
        """Compute probabilities after allowing chain-of-thought reasoning.

        The model generates tokens freely until it produces the stop sequence
        (the think-end sequence), then label probabilities are computed
        from the post-thinking position.

        Args:
            prompt_tokens: Tokenized prompt that allows the model to think.
            stop_tokens: Token sequence marking end of thinking phase.
            max_thinking_tokens: Maximum tokens for the thinking phase.
            use_cache: Whether to use KV cache during generation.
            n_absorbed: Number of prompt tokens absorbed into the label
                continuation by in-context tokenization.

        Returns:
            :class:`COTResult` with probabilities, thinking text, and tokens.
        """
        if self.decode is None:
            raise ValueError("decode callable required for COT. Pass it during initialization.")

        thinking_tokens, full_tokens = self.backend.generate_until(
            prompt_tokens, stop_tokens, max_thinking_tokens, use_cache
        )
        thinking_text = self.decode(thinking_tokens)
        effective_tokens = full_tokens[:-n_absorbed] if n_absorbed else full_tokens
        probabilities = self.compute_probabilities(effective_tokens)

        return COTResult(
            probabilities=probabilities,
            thinking_text=thinking_text,
            thinking_tokens=thinking_tokens,
        )
