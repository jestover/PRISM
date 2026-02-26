"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod
from typing import Any, List, Union


class InferenceBackend(ABC):
    """Abstract base class for inference backends.

    Implementors provide: model inference, softmax, and KV cache management.
    Must expose logit-level access — API-only backends that return text cannot
    be used (the logits are the whole point of PRISM).
    """

    # ---- Inference ----

    @abstractmethod
    def get_logits(self, tokens: List[int]) -> Any:
        """Get logits for next token position (uncached).

        Args:
            tokens: Full token sequence.

        Returns:
            Array of shape ``[vocab_size]`` with logits for the next token.
        """

    @abstractmethod
    def softmax(self, logits: Any) -> Any:
        """Apply softmax to logits.

        Args:
            logits: Raw logits (may be a subset of the vocabulary).

        Returns:
            Probabilities with same shape as input.
        """

    @abstractmethod
    def argmax(self, logits: Any) -> int:
        """Get index of the maximum logit.

        Args:
            logits: Raw logits array.

        Returns:
            Integer index of the largest value.
        """

    # ---- Cache Management ----

    @abstractmethod
    def create_cache(self) -> Any:
        """Create an empty KV cache."""

    @abstractmethod
    def forward(self, tokens: List[int], cache: Any = None) -> Union[Any, tuple]:
        """Forward pass with optional KV cache.

        Args:
            tokens: Token IDs to process.
            cache: If ``None``, return logits only. If provided, extend the
                cache and return ``(logits, updated_cache)``.

        Returns:
            Logits (no cache) or ``(logits, cache)`` tuple.
        """

    @abstractmethod
    def copy_cache(self, cache: Any) -> Any:
        """Deep-copy a KV cache for branching. Must produce an independent copy."""

    @abstractmethod
    def cache_memory_bytes(self, cache: Any) -> int:
        """Estimate memory usage of a cache in bytes."""

    @abstractmethod
    def cache_sequence_length(self, cache: Any) -> int:
        """Number of tokens stored in the cache."""

    # ---- Generation (for chain-of-thought) ----

    @abstractmethod
    def generate_until(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        max_tokens: int = 2048,
        use_cache: bool = True,
    ) -> tuple:
        """Generate tokens until a stop sequence is produced.

        Args:
            prompt_tokens: Initial prompt token IDs.
            stop_tokens: Token sequence that signals end of generation.
            max_tokens: Maximum tokens to generate.
            use_cache: Whether to use KV cache during generation.

        Returns:
            ``(generated_tokens, full_sequence_tokens)`` where
            ``full_sequence`` includes prompt + generated + stop tokens.
        """
