"""MLX inference backend for Apple Silicon."""

import copy
from typing import Any, List, Union

import mlx.core as mx
from mlx_lm import load

from prism.backends.base import InferenceBackend
from prism.utils import get_logger

logger = get_logger(__name__)

# Type alias for MLX cache (list of layer caches)
Cache = List[Any]


class MLXBackend(InferenceBackend):
    """MLX inference backend using ``mlx-lm``.

    Args:
        model_path: HuggingFace model ID or local path to an MLX model.
    """

    def __init__(self, model_path: str):
        logger.info(f"Loading MLX model from {model_path}")
        self.model, self.tokenizer = load(model_path)
        self._num_layers = len(self.model.layers)
        self._hidden_size = self.model.args.hidden_size

        vocab_size = len(self.tokenizer.get_vocab())
        logger.info(
            f"Model loaded: vocab_size={vocab_size:,}, "
            f"layers={self._num_layers}, hidden_size={self._hidden_size}"
        )

    # ---- Inference ----

    def get_logits(self, tokens: List[int]) -> mx.array:
        logits = self.model(mx.array(tokens)[None, :])
        return logits[0, -1, :]

    def softmax(self, logits: mx.array) -> mx.array:
        return mx.softmax(logits)

    def argmax(self, logits: mx.array) -> int:
        return int(mx.argmax(logits))

    # ---- Cache Management ----

    def create_cache(self) -> Cache:
        return self.model.make_cache()

    def forward(self, tokens: List[int], cache: Cache = None) -> Union[mx.array, tuple[mx.array, Cache]]:
        token_array = mx.array(tokens)[None, :]

        if cache is None:
            logits = self.model(token_array)
            return logits[0, -1, :]

        logits = self.model(token_array, cache=cache)
        mx.eval(logits, *[c.state for c in cache])
        return logits[0, -1, :], cache

    def copy_cache(self, cache: Cache) -> Cache:
        return copy.deepcopy(cache)

    def cache_sequence_length(self, cache: Cache) -> int:
        if not cache or len(cache) == 0:
            return 0
        return cache[0].offset

    def cache_memory_bytes(self, cache: Cache) -> int:
        if not cache or len(cache) == 0:
            return 0
        total_bytes = 0
        for layer_cache in cache:
            keys, values = layer_cache.state
            if keys is not None:
                total_bytes += keys.size * 2  # float16
            if values is not None:
                total_bytes += values.size * 2
        return total_bytes

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
        cache = self.model.make_cache()

        prompt_array = mx.array(prompt_tokens)[None, :]
        logits = self.model(prompt_array, cache=cache)
        mx.eval(logits, *[c.state for c in cache])

        current_tokens = list(prompt_tokens)
        generated_tokens: List[int] = []

        next_token = int(mx.argmax(logits[0, -1, :]))
        current_tokens.append(next_token)
        generated_tokens.append(next_token)

        for _ in range(max_tokens - 1):
            if len(generated_tokens) >= len(stop_tokens):
                if generated_tokens[-len(stop_tokens):] == stop_tokens:
                    generated_tokens = generated_tokens[:-len(stop_tokens)]
                    break

            token_array = mx.array([[next_token]])
            logits = self.model(token_array, cache=cache)
            mx.eval(logits, *[c.state for c in cache])

            next_token = int(mx.argmax(logits[0, -1, :]))
            current_tokens.append(next_token)
            generated_tokens.append(next_token)

        return generated_tokens, current_tokens
