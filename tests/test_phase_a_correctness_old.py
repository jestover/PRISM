"""Regression tests for Phase A correctness work."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import polars as pl
import pytest

import prism
from prism.core.label_probs import LabelProbabilityComputer
from prism.core.token_trie import TERMINAL_TOKEN
from prism.model import Model
from prism.prompts.templates import PromptBuilder


class FakeTensor:
    """Minimal tensor wrapper for backend-free probability tests."""

    def __init__(self, values: List[float]):
        self.values = list(values)

    def __getitem__(self, index):
        if isinstance(index, list):
            return FakeTensor([self.values[i] for i in index])
        return self.values[index]

    def tolist(self) -> List[float]:
        return list(self.values)


class FakeTokenizer:
    """Tokenizer with deterministic context-sensitive boundary absorption."""

    THINK_END_TEXT = "<THINK_END>"
    THINK_END_TOKEN = 91
    BOUNDARY_TOKEN = 90
    TEXT_MARKER = 50

    TASK_TOKENS = {
        "classify": 10,
        "rate": 20,
        "label": 30,
    }

    ISOLATED_LABEL_TOKENS = {
        "alpha": [300],
        "beta": [301],
        "true": [400],
        "false": [401],
    }

    CONTEXT_LABEL_TOKENS = {
        "alpha": [30],
        "beta": [31],
        "true": [40],
        "false": [41],
    }

    REASONING_LABEL_TOKENS = {
        "alpha": [32],
        "beta": [33],
        "true": [42],
        "false": [43],
    }

    def __init__(self):
        self.chat_template = "fake-template"
        self._text_tokens: Dict[str, int] = {}
        self._next_text_token = 100

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        user_message = messages[1]["content"]
        if "Label: " in user_message:
            task = "classify"
            suffix = "\n\nLabel: "
        elif "Rating: " in user_message:
            task = "rate"
            suffix = "\n\nRating: "
        elif "Applies: " in user_message:
            task = "label"
            suffix = "\n\nApplies: "
        else:
            raise ValueError(f"Unsupported user message: {user_message!r}")

        text = user_message.split("Text: ", 1)[1].split(suffix, 1)[0]
        return f"PROMPT|task={task}|text={text}|gen="

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens

        if text == self.THINK_END_TEXT:
            return [self.THINK_END_TOKEN]

        if text in self.ISOLATED_LABEL_TOKENS:
            return list(self.ISOLATED_LABEL_TOKENS[text])

        if not text.startswith("PROMPT|"):
            raise ValueError(f"Unsupported text for fake tokenizer: {text!r}")

        prefix, continuation = text.split("|gen=", 1)
        fields = dict(part.split("=", 1) for part in prefix.split("|")[1:])
        task = fields["task"]
        prompt_tokens = self.base_tokens(task, fields["text"]) + [self.BOUNDARY_TOKEN]

        if continuation == "":
            return prompt_tokens
        if continuation == self.THINK_END_TEXT:
            return prompt_tokens + [self.THINK_END_TOKEN]

        if continuation.startswith(self.THINK_END_TEXT):
            label = continuation[len(self.THINK_END_TEXT):]
            return prompt_tokens + list(self.REASONING_LABEL_TOKENS[label])

        return self.base_tokens(task, fields["text"]) + list(self.CONTEXT_LABEL_TOKENS[continuation])

    def decode(self, tokens: List[int]) -> str:
        if tokens == [700]:
            return "thinking"
        return " ".join(str(token) for token in tokens)

    def base_tokens(self, task: str, text: str) -> List[int]:
        tokens = [self.TASK_TOKENS[task], self.TEXT_MARKER]
        if text:
            tokens.append(self._text_token(text))
        return tokens

    def _text_token(self, text: str) -> int:
        if text not in self._text_tokens:
            self._text_tokens[text] = self._next_text_token
            self._next_text_token += 1
        return self._text_tokens[text]


class FakeBackend:
    """Backend driven by explicit per-prefix token distributions."""

    def __init__(
        self,
        distributions: Dict[Tuple[int, ...], Dict[int, float]],
        *,
        vocab_size: int = 800,
        thinking_tokens: List[int] | None = None,
    ):
        self.distributions = distributions
        self.vocab_size = vocab_size
        self.thinking_tokens = thinking_tokens or [700]
        self.tokenizer = None

    def get_logits(self, tokens: List[int]) -> FakeTensor:
        return self._logits_for(tuple(tokens))

    def softmax(self, logits: FakeTensor) -> FakeTensor:
        max_value = max(logits.values)
        exp_values = [0.0 if value < -1e8 else math.exp(value - max_value) for value in logits.values]
        total = sum(exp_values)
        return FakeTensor([value / total for value in exp_values])

    def argmax(self, logits: FakeTensor) -> int:
        return max(range(len(logits.values)), key=lambda index: logits.values[index])

    def create_cache(self):
        return ()

    def forward(self, tokens: List[int], cache=None):
        cache = tuple(cache or ())
        new_cache = cache + tuple(tokens)
        return self._logits_for(new_cache), new_cache

    def copy_cache(self, cache):
        return tuple(cache or ())

    def cache_memory_bytes(self, cache) -> int:
        return len(cache or ()) * 8

    def cache_sequence_length(self, cache) -> int:
        return len(cache or ())

    def generate_until(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        max_tokens: int = 2048,
        use_cache: bool = True,
    ):
        del max_tokens, use_cache
        generated = list(self.thinking_tokens) + list(stop_tokens)
        return list(self.thinking_tokens), list(prompt_tokens) + generated

    def _logits_for(self, state: Tuple[int, ...]) -> FakeTensor:
        if state not in self.distributions:
            raise KeyError(f"No fake distribution defined for state {state!r}")

        logits = [-1e9] * self.vocab_size
        for token_id, prob in self.distributions[state].items():
            logits[token_id] = math.log(prob)
        return FakeTensor(logits)


def _make_model(
    backend: FakeBackend,
    tokenizer: FakeTokenizer,
    *,
    reasoning: bool = False,
) -> Model:
    backend.tokenizer = tokenizer
    if reasoning:
        return Model(
            backend=backend,
            tokenizer=tokenizer,
            think_end=tokenizer.THINK_END_TEXT,
            think_end_tokens=tokenizer.encode(tokenizer.THINK_END_TEXT),
        )
    return Model(backend=backend, tokenizer=tokenizer)


def test_tokenize_labels_in_context_detects_absorbed_prompt_tokens():
    tokenizer = FakeTokenizer()
    model = _make_model(FakeBackend({}), tokenizer)
    builder = PromptBuilder(random_seed=0)
    system_msg, user_msg = builder.render_classify(
        text="",
        labels=["alpha", "beta"],
        shuffle=False,
    )

    prompt_text = model.format_prompt(system_msg, user_msg)
    isolated = model.tokenize_labels(["alpha", "beta"])
    in_context = model.tokenize_labels_in_context(["alpha", "beta"], prompt_text)

    assert isolated == {"alpha": [300], "beta": [301]}
    assert in_context.label_tokens == {"alpha": [30], "beta": [31]}
    assert in_context.prompt_length == in_context.raw_prompt_length - 1
    assert in_context.n_absorbed == 1


def test_prefix_overlap_labels_receive_terminal_probability_mass():
    backend = FakeBackend(
        {
            (5,): {6: 0.3, 499: 0.7},
        }
    )
    computer = LabelProbabilityComputer(
        {"pos": [5], "positive": [5, 6]},
        backend,
    )

    probs = computer.compute_probabilities([])

    assert TERMINAL_TOKEN in computer.trie.branch_points[0].branches
    assert probs["pos"] == pytest.approx(0.7, abs=1e-9)
    assert probs["positive"] == pytest.approx(0.3, abs=1e-9)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)


def test_numeric_prefix_labels_sum_to_one():
    backend = FakeBackend(
        {
            (): {1: 1.0},
            (1,): {0: 0.4, 499: 0.6},
            (1, 0): {0: 0.5, 499: 0.5},
        }
    )
    computer = LabelProbabilityComputer(
        {"1": [1], "10": [1, 0], "100": [1, 0, 0]},
        backend,
    )

    probs = computer.compute_probabilities([])

    assert probs == pytest.approx({"1": 0.6, "10": 0.2, "100": 0.2}, abs=1e-9)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)


def test_cached_and_uncached_probability_paths_match():
    backend = FakeBackend(
        {
            (): {1: 1.0},
            (1,): {0: 0.4, 499: 0.6},
            (1, 0): {0: 0.5, 499: 0.5},
        }
    )
    computer = LabelProbabilityComputer(
        {"1": [1], "10": [1, 0], "100": [1, 0, 0]},
        backend,
    )

    uncached = computer.compute_probabilities([])
    cached = computer.compute_probabilities_cached(
        backend.get_logits([]),
        backend.create_cache(),
    )

    assert cached == pytest.approx(uncached, abs=1e-9)


def test_classify_uses_in_context_label_tokenization():
    tokenizer = FakeTokenizer()
    text = "great"
    base_state = tuple(tokenizer.base_tokens("classify", text))
    backend = FakeBackend(
        {
            (10,): {0: 1.0},
            base_state: {
                30: 0.45,
                31: 0.05,
                300: 0.05,
                301: 0.45,
            }
        }
    )
    model = _make_model(backend, tokenizer)
    df = pl.DataFrame({"text": [text]})

    result = prism.classify(
        df,
        column_name="text",
        labels=["alpha", "beta"],
        model=model,
        shuffle_labels=False,
    )

    assert result["predicted_class"][0] == "alpha"
    assert result["prob_alpha"][0] == pytest.approx(0.9, abs=1e-9)
    assert result["prob_beta"][0] == pytest.approx(0.1, abs=1e-9)


def test_label_returns_independent_true_false_probabilities():
    tokenizer = FakeTokenizer()
    text = "great"
    base_state = tuple(tokenizer.base_tokens("label", text))
    backend = FakeBackend(
        {
            (30,): {0: 1.0},
            base_state: {
                40: 0.55,
                41: 0.05,
                400: 0.05,
                401: 0.35,
            }
        }
    )
    model = _make_model(backend, tokenizer)
    df = pl.DataFrame({"text": [text]})

    result = prism.label(
        df,
        column_name="text",
        labels={"applies": "Applies to the text"},
        model=model,
    )

    assert result["prob_true_applies"][0] == pytest.approx(0.9166666667, abs=1e-9)
    assert result["predicted_applies"][0] is True


def test_reasoning_and_direct_paths_share_distribution_contract():
    tokenizer = FakeTokenizer()
    text = "reasoned"
    direct_state = tuple(tokenizer.base_tokens("classify", text) + [tokenizer.BOUNDARY_TOKEN])
    reasoning_state = tuple(
        tokenizer.base_tokens("classify", text) + [tokenizer.BOUNDARY_TOKEN, 700]
    )
    backend = FakeBackend(
        {
            (10,): {0: 1.0},
            direct_state: {32: 0.75, 33: 0.25},
            reasoning_state: {32: 0.75, 33: 0.25},
        }
    )
    model = _make_model(backend, tokenizer, reasoning=True)
    df = pl.DataFrame({"text": [text]})

    direct = prism.classify(
        df,
        column_name="text",
        labels=["alpha", "beta"],
        model=model,
        shuffle_labels=False,
    )
    reasoned = prism.classify(
        df,
        column_name="text",
        labels=["alpha", "beta"],
        model=model,
        use_reasoning=True,
        shuffle_labels=False,
    )

    direct_total = direct["prob_alpha"][0] + direct["prob_beta"][0]
    reasoned_total = reasoned["prob_alpha"][0] + reasoned["prob_beta"][0]

    assert direct_total == pytest.approx(1.0, abs=1e-9)
    assert reasoned_total == pytest.approx(1.0, abs=1e-9)
    assert direct["prob_alpha"][0] == pytest.approx(reasoned["prob_alpha"][0], abs=1e-9)
    assert reasoned["thinking_text"][0] == "thinking"
