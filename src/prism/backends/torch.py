"""PyTorch inference backend for CUDA / CPU."""

import copy
from typing import Any, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prism.backends.base import InferenceBackend
from prism.utils import get_logger

logger = get_logger(__name__)


class TorchBackend(InferenceBackend):
    """PyTorch inference backend using ``transformers``.

    Args:
        model_path: HuggingFace model ID or local path.
        device: Device string (``"cuda"``, ``"mps"``, ``"cpu"``).
            Auto-detected if not provided.
        use_half_precision: Load model in float16 for faster inference.
        **kwargs: Passed to ``AutoModelForCausalLM.from_pretrained``.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        use_half_precision: bool = True,
        **kwargs,
    ):
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        logger.info(f"Loading Torch model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        dtype = torch.float16 if use_half_precision else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, dtype=dtype, **kwargs
        )
        self.model.eval()

        vocab_size = len(self.tokenizer.get_vocab())
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Torch model loaded: vocab_size={vocab_size:,}, "
            f"parameters={param_count:,}, dtype={dtype}"
        )

    # ---- Inference ----

    def get_logits(self, tokens: List[int]) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(torch.tensor(tokens, device=self.device).unsqueeze(0))
        return output.logits[0, -1, :]

    def softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(logits, dim=-1)

    def argmax(self, logits: torch.Tensor) -> int:
        return int(torch.argmax(logits))

    # ---- Cache Management ----

    def create_cache(self) -> Any:
        return None  # PyTorch DynamicCache created on first forward pass

    def forward(self, tokens: List[int], cache: Any = None) -> Union[torch.Tensor, tuple]:
        token_tensor = torch.tensor(tokens, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if cache is None:
                output = self.model(token_tensor)
                return output.logits[0, -1, :]

            output = self.model(token_tensor, past_key_values=cache, use_cache=True)
            return output.logits[0, -1, :], output.past_key_values

    def copy_cache(self, cache: Any) -> Any:
        if cache is None:
            return None
        return copy.deepcopy(cache)

    def cache_memory_bytes(self, cache: Any) -> int:
        if cache is None:
            return 0
        total = 0
        for layer_kv in cache:
            for tensor in layer_kv:
                total += tensor.nelement() * tensor.element_size()
        return total

    def cache_sequence_length(self, cache: Any) -> int:
        if cache is None:
            return 0
        # First layer, key tensor, sequence length dimension
        return cache[0][0].shape[2]

    # ---- Generation ----

    def generate_until(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        max_tokens: int = 2048,
        use_cache: bool = True,
    ) -> tuple[List[int], List[int]]:
        if use_cache:
            return self._generate_with_cache(prompt_tokens, stop_tokens, max_tokens)
        else:
            return self._generate_without_cache(prompt_tokens, stop_tokens, max_tokens)

    def _generate_without_cache(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        max_tokens: int,
    ) -> tuple[List[int], List[int]]:
        current_tokens = list(prompt_tokens)
        generated_tokens: List[int] = []

        for _ in range(max_tokens):
            logits = self.get_logits(current_tokens)
            next_token = self.argmax(logits)
            current_tokens.append(next_token)
            generated_tokens.append(next_token)

            if len(generated_tokens) >= len(stop_tokens):
                if generated_tokens[-len(stop_tokens):] == stop_tokens:
                    generated_tokens = generated_tokens[:-len(stop_tokens)]
                    break

        return generated_tokens, current_tokens

    def _generate_with_cache(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        max_tokens: int,
    ) -> tuple[List[int], List[int]]:
        # Process prompt and build initial cache
        token_tensor = torch.tensor(prompt_tokens, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.model(token_tensor, use_cache=True)
        cache = output.past_key_values

        current_tokens = list(prompt_tokens)
        generated_tokens: List[int] = []

        next_token = int(torch.argmax(output.logits[0, -1, :]))
        current_tokens.append(next_token)
        generated_tokens.append(next_token)

        for _ in range(max_tokens - 1):
            if len(generated_tokens) >= len(stop_tokens):
                if generated_tokens[-len(stop_tokens):] == stop_tokens:
                    generated_tokens = generated_tokens[:-len(stop_tokens)]
                    break

            token_tensor = torch.tensor([[next_token]], device=self.device)
            with torch.no_grad():
                output = self.model(token_tensor, past_key_values=cache, use_cache=True)
            cache = output.past_key_values

            next_token = int(torch.argmax(output.logits[0, -1, :]))
            current_tokens.append(next_token)
            generated_tokens.append(next_token)

        return generated_tokens, current_tokens
