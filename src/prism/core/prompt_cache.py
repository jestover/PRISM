"""Cascading KV cache for prompt prefix reuse.

Implements a 4-level cache hierarchy that ensures no token is ever
forwarded through the model twice:

- **Level 0** — pre-labels prefix (built once per run)
- **Level 1** — per-ordering (built once per label-ordering group)
- **Level 2** — per-row (built once per row, via ``forward_row``)
- **Level 3** — per-branch point (handled by
  :meth:`~prism.core.label_probs.LabelProbabilityComputer.compute_probabilities_cached`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

from prism.utils import get_logger

if TYPE_CHECKING:
    from prism.backends.base import InferenceBackend
    from prism.model import Model

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Split-point detection
# ---------------------------------------------------------------------------


def find_split_point(
    tokens_a: List[int],
    tokens_b: List[int],
    *,
    bpe_guard: bool = False,
) -> int:
    """Find the index where two token sequences first diverge.

    Used to determine cache boundaries by comparing two prompts that
    differ only in the segment we want to isolate (e.g. different label
    orderings, or different sentinel texts).

    Args:
        tokens_a: First token sequence.
        tokens_b: Second token sequence.
        bpe_guard: If ``True``, back up 1 token from the raw divergence
            point.  BPE tokenizers can merge a delimiter (e.g. a space)
            with the first character of the following text, producing a
            different token depending on what that character is.
            Sentinel characters (``\\x00``, ``\\x01``) do NOT merge with
            the preceding space the way real text does, so the last
            "shared" token from sentinel probing may differ from the
            actual token at that position.  Subtracting 1 ensures the
            cached prefix contains only tokens that are truly stable.

    Returns:
        Index of the first diverging token.  All tokens before this
        index are shared and can be cached.
    """
    min_len = min(len(tokens_a), len(tokens_b))
    for i in range(min_len):
        if tokens_a[i] != tokens_b[i]:
            return max(i - 1, 0) if bpe_guard else i
    result = min_len
    return max(result - 1, 0) if bpe_guard else result


# ---------------------------------------------------------------------------
# CascadingCache
# ---------------------------------------------------------------------------


class CascadingCache:
    """4-level cascading KV cache hierarchy.

    Levels 0–1 are managed here.  Level 2 is produced by
    :meth:`forward_row`, and Level 3 is handled inside
    :class:`~prism.core.label_probs.LabelProbabilityComputer`.

    Args:
        backend: Inference backend with cache management.
        model: PRISM Model (for tokenization).
        render_system_fn: Callable that takes a label ordering (tuple of
            label strings) and returns the system message string.
        render_user_fn: Callable that takes ``(text, context)`` and returns
            the user message string.
        labels: The full list of labels (used to probe split points).
        label_descriptions: Optional label descriptions (needed for rendering).
        additional_instructions: Optional extra instructions.
        context_is_constant: Whether context is the same for every row.
            If ``True`` (or context is ``None``), the user-message prefix
            up to the text insertion point is included in the Level 1 cache.
            If ``False`` (per-row context), only the system portion is cached.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        model: Model,
        render_system_fn: Callable[..., str],
        render_user_fn: Callable[..., str],
        labels: List[str],
        *,
        context_is_constant: bool = True,
        build_level0: bool = True,
    ):
        self.backend = backend
        self.model = model
        self._render_system = render_system_fn
        self._render_user = render_user_fn
        self._labels = labels
        self._context_is_constant = context_is_constant

        # -- Level 0: find label split and build pre-labels cache --

        self._label_split = 0
        self._level0_cache: Optional[Any] = None

        if build_level0 and len(labels) >= 2:
            # Probe with two different orderings to find where labels diverge
            ordering_a = tuple(labels)
            ordering_b = tuple(reversed(labels))

            sys_a = render_system_fn(ordering_a)
            sys_b = render_system_fn(ordering_b)

            user_probe = render_user_fn("\x00", None)

            tokens_a = model.tokenize_prompt(sys_a, user_probe)
            tokens_b = model.tokenize_prompt(sys_b, user_probe)

            self._label_split = find_split_point(tokens_a, tokens_b)

            if self._label_split > 0:
                # Build Level 0 cache: forward pre-label tokens
                pre_label_tokens = tokens_a[: self._label_split]
                cache = self.backend.create_cache()
                _, self._level0_cache = self.backend.forward(
                    pre_label_tokens, cache
                )
                mem = self.backend.cache_memory_bytes(self._level0_cache)
                logger.info(
                    f"CascadingCache Level 0: cached {self._label_split} "
                    f"pre-label tokens ({mem / 1024:.1f} KB)"
                )
            else:
                logger.info("CascadingCache Level 0: labels at position 0")
        else:
            logger.info(
                "CascadingCache Level 0: skipped "
                f"({'build_level0=False' if not build_level0 else 'single label'})"
            )

        # Level 1 state (set by set_ordering or set_fixed_prefix)
        self._ordering_cache: Optional[Any] = None
        self._text_split: int = 0

    def set_ordering(
        self,
        ordering: Tuple[str, ...],
        constant_context: Optional[str] = None,
    ) -> None:
        """Build Level 1 cache for a specific label ordering.

        Copies the Level 0 cache and extends it with the ordering-specific
        tokens (label list + post-system + user prefix up to text).

        Args:
            ordering: The label ordering for this group.
            constant_context: If context is constant, include it in the
                cached prefix.  Pass ``None`` when context varies per row.
        """
        sys_msg = self._render_system(ordering)

        # Find text split for this ordering.
        # When context is constant, include it in the cached prefix so we
        # cache right up to the per-row text.  When context varies per row,
        # probe with raw sentinels to find the system/user-content boundary
        # (everything in the user message is per-row in that case).
        if self._context_is_constant:
            user_a = self._render_user("\x00", constant_context)
            user_b = self._render_user("\x01", constant_context)
        else:
            # Raw sentinels — finds boundary between chat template user
            # marker and actual user content.
            user_a = "\x00"
            user_b = "\x01"

        tokens_a = self.model.tokenize_prompt(sys_msg, user_a)
        tokens_b = self.model.tokenize_prompt(sys_msg, user_b)
        self._text_split = find_split_point(tokens_a, tokens_b, bpe_guard=True)

        # Build Level 1: extend Level 0 with ordering-specific tokens
        ordering_tokens = tokens_a[self._label_split : self._text_split]

        if self._level0_cache is not None and ordering_tokens:
            self._ordering_cache = self.backend.copy_cache(self._level0_cache)
            _, self._ordering_cache = self.backend.forward(
                ordering_tokens, self._ordering_cache
            )
        elif self._level0_cache is not None:
            # No ordering-specific tokens (unusual but handle gracefully)
            self._ordering_cache = self.backend.copy_cache(self._level0_cache)
        else:
            # No Level 0 — build Level 1 from scratch
            cache = self.backend.create_cache()
            prefix_tokens = tokens_a[: self._text_split]
            _, self._ordering_cache = self.backend.forward(
                prefix_tokens, cache
            )

        mem = self.backend.cache_memory_bytes(self._ordering_cache)
        logger.debug(
            f"CascadingCache Level 1: cached {self._text_split} tokens "
            f"for ordering ({mem / 1024:.1f} KB)"
        )

    def set_fixed_prefix(self, system_msg: str, constant_context: Optional[str] = None) -> None:
        """Build Level 1 cache for tasks without label shuffling.

        For ``rate()`` and ``label()`` (and ``classify()`` with
        ``shuffle_labels=False``), the system message is constant.
        Level 0 and Level 1 collapse into a single cache.

        Args:
            system_msg: The fixed system message.
            constant_context: If context is constant, include it.
                Pass ``None`` when context varies per row.
        """
        if self._context_is_constant:
            user_a = self._render_user("\x00", constant_context)
            user_b = self._render_user("\x01", constant_context)
        else:
            user_a = "\x00"
            user_b = "\x01"

        tokens_a = self.model.tokenize_prompt(system_msg, user_a)
        tokens_b = self.model.tokenize_prompt(system_msg, user_b)
        self._text_split = find_split_point(tokens_a, tokens_b, bpe_guard=True)
        self._label_split = self._text_split  # no separate label boundary

        prefix_tokens = tokens_a[: self._text_split]
        cache = self.backend.create_cache()
        _, self._ordering_cache = self.backend.forward(prefix_tokens, cache)

        mem = self.backend.cache_memory_bytes(self._ordering_cache)
        logger.info(
            f"CascadingCache fixed prefix: cached {self._text_split} tokens "
            f"({mem / 1024:.1f} KB)"
        )

    @property
    def text_split(self) -> int:
        """Token index where per-row text begins."""
        return self._text_split

    def forward_row(self, full_prompt_tokens: List[int]) -> Tuple[Any, Any]:
        """Build Level 2 cache for a specific row.

        Copies the Level 1 (ordering) cache and extends it with the
        row-specific tokens (text + suffix).

        Args:
            full_prompt_tokens: The complete tokenized prompt for this row.

        Returns:
            ``(logits, cache)`` — logits at the final prompt position, and
            the extended cache containing the full prompt's KV state.
        """
        if self._ordering_cache is None:
            raise RuntimeError(
                "Call set_ordering() or set_fixed_prefix() before forward_row()"
            )

        suffix_tokens = full_prompt_tokens[self._text_split:]
        row_cache = self.backend.copy_cache(self._ordering_cache)
        logits, row_cache = self.backend.forward(suffix_tokens, row_cache)
        return logits, row_cache
