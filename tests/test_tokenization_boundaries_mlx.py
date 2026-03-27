"""Opt-in tokenizer boundary checks for blessed MLX model families.

Run manually with:
    PRISM_RUN_MLX_TOKENIZER_TESTS=1 uv run --extra mlx --extra dev \
        pytest tests/test_tokenization_boundaries_mlx.py -v
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

from prism.model import Model, detect_think_end
from prism.prompts.templates import PromptBuilder
from prism.tasks.shared import build_probability_computer

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("PRISM_RUN_MLX_TOKENIZER_TESTS") != "1",
        reason="Set PRISM_RUN_MLX_TOKENIZER_TESTS=1 to run blessed MLX tokenizer checks.",
    ),
    pytest.mark.filterwarnings(
        "ignore:builtin type SwigPyPacked has no __module__ attribute:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:builtin type SwigPyObject has no __module__ attribute:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:builtin type swigvarlink has no __module__ attribute:DeprecationWarning"
    ),
]

BLESSED_MODELS_PATH = Path(__file__).with_name("blessed_models.toml")


class _DummyBackend:
    """Placeholder backend for tokenizer-only probability-computer construction."""


@dataclass(frozen=True)
class BlessedModelSpec:
    """Blessed-model metadata used by opt-in verification tests."""

    id: str
    backend: str
    family: str
    model_path: str
    expected_think_end: str


@dataclass(frozen=True)
class LoadedBlessedModel:
    """Loaded tokenizer/model wrapper paired with its blessed-model spec."""

    spec: BlessedModelSpec
    model: Model


def _load_reasoning_mlx_models() -> list[BlessedModelSpec]:
    with BLESSED_MODELS_PATH.open("rb") as handle:
        config = tomllib.load(handle)

    models = []
    for raw_model in config["models"]:
        if raw_model["backend"] != "mlx" or not raw_model["reasoning"]:
            continue
        models.append(
            BlessedModelSpec(
                id=raw_model["id"],
                backend=raw_model["backend"],
                family=raw_model["family"],
                model_path=raw_model["model_path"],
                expected_think_end=raw_model["expected_think_end"],
            )
        )
    return models


@pytest.fixture(
    scope="module", params=_load_reasoning_mlx_models(), ids=lambda spec: spec.id
)
def reasoning_model(request):
    transformers = pytest.importorskip("transformers")
    spec: BlessedModelSpec = request.param

    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.model_path)
    think_end = detect_think_end(tokenizer)
    assert think_end is not None, (
        f"{spec.id}: blessed tokenizer must expose a reasoning boundary"
    )
    assert think_end == spec.expected_think_end, (
        f"{spec.id}: detected think_end {think_end!r} does not match blessed expectation "
        f"{spec.expected_think_end!r}"
    )

    print(f"\nBlessed tokenizer model: {spec.model_path}")
    print(f"Detected think_end: {think_end!r}")

    return LoadedBlessedModel(
        spec=spec,
        model=Model(
            backend=_DummyBackend(),
            tokenizer=tokenizer,
            think_end=think_end,
            think_end_tokens=tokenizer.encode(think_end, add_special_tokens=False),
        ),
    )


@pytest.mark.parametrize(
    ("task_name", "candidates", "render"),
    [
        (
            "classify",
            ["positive", "negative", "neutral"],
            lambda builder: builder.render_classify(
                text="The author praises the proposal but notes some uncertainty.",
                labels=["positive", "negative", "neutral"],
                shuffle=False,
            ),
        ),
        (
            "rate",
            ["0", "1", "2", "3"],
            lambda builder: builder.render_rate(
                text="The speech uses moderately populist rhetoric.",
                attribute="populism",
                scale_min=0,
                scale_max=3,
            ),
        ),
        (
            "label",
            ["true", "false"],
            lambda builder: builder.render_label(
                text="The message is sarcastic but not overtly hostile.",
                label="sarcastic",
                label_description="Uses sarcasm or irony",
            ),
        ),
    ],
)
def test_reasoning_boundary_tokenization_matches_post_think_end_context(
    reasoning_model: LoadedBlessedModel,
    task_name: str,
    candidates: list[str],
    render,
):
    builder = PromptBuilder(random_seed=0)
    system_message, user_message = render(builder)
    model = reasoning_model.model
    model_id = reasoning_model.spec.id

    prompt_without_boundary = model.format_prompt(system_message, user_message)
    prompt_with_boundary = prompt_without_boundary + model.think_end

    expected = model.tokenize_labels_in_context(candidates, prompt_with_boundary)
    computer, n_absorbed = build_probability_computer(
        model,
        candidates,
        system_message,
        user_message,
    )

    prompt_tokens = model.tokenizer.encode(prompt_with_boundary)

    assert computer.trie.label_sequences == expected.label_tokens, (
        f"{model_id} {task_name}: build_probability_computer produced candidate token sequences "
        "that do not match tokenize_labels_in_context(..., prompt + think_end)"
    )
    assert n_absorbed == expected.n_absorbed, (
        f"{model_id} {task_name}: absorbed-token count from build_probability_computer does not "
        "match tokenize_labels_in_context"
    )
    assert expected.prompt_length <= expected.raw_prompt_length, (
        f"{model_id} {task_name}: prompt_length should never exceed raw_prompt_length"
    )
    assert expected.raw_prompt_length == len(prompt_tokens), (
        f"{model_id} {task_name}: raw_prompt_length should equal the encoded length of "
        "prompt + think_end"
    )
    assert expected.prompt_length == len(prompt_tokens) - n_absorbed, (
        f"{model_id} {task_name}: prompt_length should equal len(prompt + think_end) - n_absorbed"
    )

    for candidate in candidates:
        combined = model.tokenizer.encode(prompt_with_boundary + candidate)
        assert (
            combined[: expected.prompt_length]
            == prompt_tokens[: expected.prompt_length]
        ), (
            f"{model_id} {task_name} candidate={candidate!r}: encoded prompt prefix diverged before "
            "the stable prompt boundary"
        )
        assert combined[expected.prompt_length :] == expected.label_tokens[candidate], (
            f"{model_id} {task_name} candidate={candidate!r}: stored continuation tokens do not "
            "match the encoded suffix after the stable prompt boundary"
        )
