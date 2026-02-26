"""Compute label probabilities only at branching points for efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from prism.core.token_trie import LabelTokenTrie
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

            valid_tokens = list(branch_point.branches.keys())
            masked_logits = logits[valid_tokens]
            probs = self.backend.softmax(masked_logits)
            probs_list = probs.tolist()

            for token_id, prob in zip(valid_tokens, probs_list):
                for label in branch_point.branches[token_id]:
                    label_probs[label] *= prob

        return label_probs

    def compute_probabilities_with_cot(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        max_thinking_tokens: int = 2048,
        use_cache: bool = True,
    ) -> COTResult:
        """Compute probabilities after allowing chain-of-thought reasoning.

        The model generates tokens freely until it produces the stop sequence
        (typically the reasoning prefix), then label probabilities are computed
        from the post-thinking position.

        Args:
            prompt_tokens: Tokenized prompt that allows the model to think.
            stop_tokens: Token sequence marking end of thinking phase.
            max_thinking_tokens: Maximum tokens for the thinking phase.
            use_cache: Whether to use KV cache during generation.

        Returns:
            :class:`COTResult` with probabilities, thinking text, and tokens.
        """
        if self.decode is None:
            raise ValueError("decode callable required for COT. Pass it during initialization.")

        thinking_tokens, full_tokens = self.backend.generate_until(
            prompt_tokens, stop_tokens, max_thinking_tokens, use_cache
        )
        thinking_text = self.decode(thinking_tokens)
        probabilities = self.compute_probabilities(full_tokens)

        return COTResult(
            probabilities=probabilities,
            thinking_text=thinking_text,
            thinking_tokens=thinking_tokens,
        )
