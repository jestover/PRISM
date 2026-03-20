"""Model loading and the Model class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from prism.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LabelTokenization:
    """Result of in-context label tokenization."""

    label_tokens: Dict[str, List[int]]
    prompt_length: int
    raw_prompt_length: int

    @property
    def n_absorbed(self) -> int:
        """Number of prompt tokens absorbed into the label continuation."""
        return self.raw_prompt_length - self.prompt_length


class Model:
    """A loaded model ready for use with PRISM.

    Bundles an inference backend, tokenizer (with chat template), and
    reasoning configuration.  Created via :func:`load_model`.
    """

    def __init__(
        self,
        backend: Any,
        tokenizer: Any,
        think_end: Optional[str] = None,
        think_end_tokens: Optional[List[int]] = None,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.think_end = think_end
        self.think_end_tokens = think_end_tokens

    @property
    def can_reason(self) -> bool:
        """Whether the model supports chain-of-thought reasoning."""
        return self.think_end is not None

    def format_prompt(self, system_message: str, user_message: str) -> str:
        """Build a prompt string using the model's chat template."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def tokenize_prompt(self, system_message: str, user_message: str) -> List[int]:
        """Build a prompt using the model's chat template and tokenize it.

        Args:
            system_message: System prompt content.
            user_message: User prompt content.

        Returns:
            Token IDs for the full formatted prompt.
        """
        return self.tokenizer.encode(self.format_prompt(system_message, user_message))

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

    def tokenize_labels_in_context(
        self, labels: List[str], prompt_text: str
    ) -> LabelTokenization:
        """Tokenize labels as continuations of the given prompt text.

        BPE tokenizers can absorb the final prompt token into the first label
        token. This method finds the earliest stable prompt boundary shared
        across all candidate labels and returns label token sequences from that
        boundary onward.
        """
        prompt_tokens = self.tokenizer.encode(prompt_text)
        raw_prompt_length = len(prompt_tokens)

        combined_tokens: Dict[str, List[int]] = {}
        divergence_points: Dict[str, int] = {}

        for label in labels:
            combined = self.tokenizer.encode(prompt_text + label)
            combined_tokens[label] = combined

            diverge = raw_prompt_length
            for i in range(min(raw_prompt_length, len(combined))):
                if combined[i] != prompt_tokens[i]:
                    diverge = i
                    break
            divergence_points[label] = diverge

        prompt_length = min(divergence_points.values())

        label_tokens: Dict[str, List[int]] = {}
        for label in labels:
            combined = combined_tokens[label]
            for i in range(prompt_length):
                if combined[i] != prompt_tokens[i]:
                    raise ValueError(
                        f"Unexpected token mismatch at position {i} for label {label!r}"
                    )

            continuation = combined[prompt_length:]
            if not continuation:
                logger.warning(
                    "Label %r produced an empty in-context continuation; "
                    "falling back to isolated tokenization",
                    label,
                )
                continuation = self.tokenizer.encode(label, add_special_tokens=False)

            label_tokens[label] = continuation

        if prompt_length < raw_prompt_length:
            logger.info(
                "BPE boundary: prompt cutoff backed up from %s to %s (%s absorbed)",
                raw_prompt_length,
                prompt_length,
                raw_prompt_length - prompt_length,
            )

        return LabelTokenization(
            label_tokens=label_tokens,
            prompt_length=prompt_length,
            raw_prompt_length=raw_prompt_length,
        )

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to a string."""
        return self.tokenizer.decode(tokens)


def detect_think_end(tokenizer) -> Optional[str]:
    """Auto-detect the think-end token sequence from a model's chat template.

    Inspects the Jinja2 chat template for known reasoning model patterns:

    - ``<think>``/``</think>`` — QwQ, DeepSeek-R1, Qwen3, Phi-4
    - ``<|channel|>`` — OpenAI gpt-oss (Harmony format)
    - ``[THINK]``/``[/THINK]`` — Mistral Magistral

    Args:
        tokenizer: A HuggingFace tokenizer with a ``chat_template`` attribute.

    Returns:
        The think-end token string, or ``None`` if the model does not appear
        to be a reasoning model.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None

    # gpt-oss / Harmony format: uses channel tokens for reasoning
    if "<|channel|>" in template:
        return "<|channel|>final<|message|>"

    # QwQ, DeepSeek-R1, Qwen3, Phi-4: <think>...</think> pattern
    if "<think>" in template or "enable_thinking" in template:
        return "</think>"

    # Mistral Magistral: [THINK]...[/THINK] pattern
    if "[THINK]" in template:
        return "[/THINK]"

    return None


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
    think_end: Optional[str] = None,
    **backend_kwargs,
) -> Model:
    """Load a model for use with PRISM.

    Args:
        model_path: HuggingFace model ID or local path.
        backend: ``"mlx"``, ``"torch"``, or ``"auto"``
            (MLX on Apple Silicon, else PyTorch).
        think_end: Token string marking the end of the thinking phase
            (e.g. ``"</think>"`` or ``"<|channel|>final<|message|>"``).
            If ``None``, PRISM auto-detects from the model's chat template.
            Pass explicitly to override auto-detection.
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

    # Auto-detect think_end from chat template if not provided
    if think_end is None:
        think_end = detect_think_end(tokenizer)
        if think_end is not None:
            logger.info(f"Auto-detected reasoning model (think_end={think_end!r})")

    think_end_tokens = None
    if think_end is not None:
        think_end_tokens = tokenizer.encode(think_end, add_special_tokens=False)
        logger.info(
            f"Think-end sequence: {think_end!r} -> "
            f"{len(think_end_tokens)} tokens"
        )

    logger.info(
        f"Model ready: {model_path} "
        f"(backend={backend}, can_reason={think_end is not None})"
    )
    return Model(
        backend=backend_instance,
        tokenizer=tokenizer,
        think_end=think_end,
        think_end_tokens=think_end_tokens,
    )
