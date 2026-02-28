"""Model loading and the Model class."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from prism.utils import get_logger

logger = get_logger(__name__)


class Model:
    """A loaded model ready for use with PRISM.

    Bundles an inference backend, tokenizer (with chat template), and
    reasoning configuration.  Created via :func:`load_model`.
    """

    def __init__(
        self,
        backend: Any,
        tokenizer: Any,
        reasoning_prefix: Optional[str] = None,
        reasoning_prefix_tokens: Optional[List[int]] = None,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.reasoning_prefix = reasoning_prefix
        self.reasoning_prefix_tokens = reasoning_prefix_tokens

    @property
    def can_reason(self) -> bool:
        """Whether the model supports chain-of-thought reasoning."""
        return self.reasoning_prefix is not None

    def tokenize_prompt(self, system_message: str, user_message: str) -> List[int]:
        """Build a prompt using the model's chat template and tokenize it.

        Args:
            system_message: System prompt content.
            user_message: User prompt content.

        Returns:
            Token IDs for the full formatted prompt.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer.encode(prompt)

    def tokenize_labels(self, labels: List[str]) -> Dict[str, List[int]]:
        """Tokenize each label string into its token ID sequence.

        Args:
            labels: Label strings (e.g. ``["positive", "negative"]``).

        Returns:
            Mapping of ``{label: [token_ids]}``.
        """
        return {
            label: self.tokenizer.encode(label, add_special_tokens=False)
            for label in labels
        }

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to a string."""
        return self.tokenizer.decode(tokens)


def _detect_backend() -> str:
    """Auto-detect the best available backend."""
    try:
        import mlx.core  # noqa: F401

        return "mlx"
    except ImportError:
        pass
    return "torch"


def load_model(
    model_path: str,
    backend: str = "auto",
    reasoning_prefix: Optional[str] = None,
    **backend_kwargs,
) -> Model:
    """Load a model for use with PRISM.

    Args:
        model_path: HuggingFace model ID or local path.
        backend: ``"mlx"``, ``"torch"``, or ``"auto"``
            (MLX on Apple Silicon, else PyTorch).
        reasoning_prefix: Token string marking end of reasoning phase.
            ``None`` if the model doesn't support chain-of-thought.
        **backend_kwargs: Passed to the backend constructor
            (e.g. ``device="cuda:0"`` for Torch).

    Returns:
        A :class:`Model` ready for :func:`~prism.classify`,
        :func:`~prism.rate`, etc.
    """
    if backend == "auto":
        backend = _detect_backend()
        logger.info(f"Auto-detected backend: {backend}")

    if backend == "mlx":
        from prism.backends.mlx import MLXBackend

        backend_instance = MLXBackend(model_path, **backend_kwargs)
    elif backend == "torch":
        from prism.backends.torch import TorchBackend

        backend_instance = TorchBackend(model_path, **backend_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'mlx', 'torch', or 'auto'.")

    tokenizer = backend_instance.tokenizer

    reasoning_prefix_tokens = None
    if reasoning_prefix is not None:
        reasoning_prefix_tokens = tokenizer.encode(reasoning_prefix, add_special_tokens=False)
        logger.info(
            f"Reasoning prefix: {reasoning_prefix!r} -> "
            f"{len(reasoning_prefix_tokens)} tokens"
        )

    logger.info(
        f"Model ready: {model_path} "
        f"(backend={backend}, can_reason={reasoning_prefix is not None})"
    )
    return Model(
        backend=backend_instance,
        tokenizer=tokenizer,
        reasoning_prefix=reasoning_prefix,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
    )
