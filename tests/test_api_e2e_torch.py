"""End-to-end API tests using gpt-oss-20b on PyTorch.

Mirrors test_api_e2e.py but uses the Torch backend to verify
identical behavior across backends.

NOTE: gpt-oss-20b uses Mixture of Experts (MoE) which requires torch.histc(),
not supported on MPS. Tests force device="cpu". This is slow (~minutes per
forward pass for a 20B model) but verifies correctness.

On CUDA machines, remove device="cpu" for much faster execution.

Requires the model to be downloaded. Run with:
    uv run --extra torch --extra dev pytest tests/test_api_e2e_torch.py -v -s
"""

import pytest
import polars as pl

import prism


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_PATH = "openai/gpt-oss-20b"
THINK_END = "<|channel|>final<|message|>"


@pytest.fixture(scope="module")
def model():
    """Model with think-end sequence (used for both direct and COT modes).

    The think-end sequence is needed even for direct mode on reasoning models —
    it tells the model to skip thinking and answer immediately.

    Uses device="cpu" because gpt-oss-20b's MoE layers use torch.histc()
    which is not implemented for Int on MPS. On CUDA, remove device="cpu".
    """
    prism.set_log_level("info")
    return prism.load_model(
        MODEL_PATH, backend="torch", think_end=THINK_END,
        device="cpu", use_half_precision=False,
    )


@pytest.fixture
def sample_df():
    """Small DataFrame with obvious sentiment texts."""
    return pl.DataFrame(
        {
            "text": [
                "I absolutely love this product, it's amazing and wonderful!",
                "This is terrible, worst experience I've ever had.",
                "The meeting is scheduled for 3pm on Tuesday.",
            ]
        }
    )


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_direct(self, model, sample_df):
        result = prism.classify(
            sample_df,
            column_name="text",
            labels=["positive", "negative", "neutral"],
            model=model,
            shuffle_labels=False,
        )

        # Check columns exist
        assert "prob_positive" in result.columns
        assert "prob_negative" in result.columns
        assert "prob_neutral" in result.columns
        assert "predicted_class" in result.columns
        assert "max_prob" in result.columns
        assert "entropy" in result.columns

        # Check shape
        assert result.shape[0] == 3

        # Probabilities should sum to ~1
        for i in range(3):
            total = (
                result["prob_positive"][i]
                + result["prob_negative"][i]
                + result["prob_neutral"][i]
            )
            assert abs(total - 1.0) < 0.01, f"Row {i}: probs sum to {total}"

        # Check predictions make sense
        predictions = result["predicted_class"].to_list()
        print(f"\nClassify (direct, torch) predictions: {predictions}")
        print(f"  Row 0 (positive text): {predictions[0]}, max_prob={result['max_prob'][0]:.3f}")
        print(f"  Row 1 (negative text): {predictions[1]}, max_prob={result['max_prob'][1]:.3f}")
        print(f"  Row 2 (neutral text):  {predictions[2]}, max_prob={result['max_prob'][2]:.3f}")

        assert predictions[0] == "positive"
        assert predictions[1] == "negative"

    def test_with_reasoning(self, model, sample_df):
        result = prism.classify(
            sample_df,
            column_name="text",
            labels=["positive", "negative", "neutral"],
            model=model,
            use_reasoning=True,
            shuffle_labels=False,
        )

        predictions = result["predicted_class"].to_list()
        print(f"\nClassify (reasoning, torch) predictions: {predictions}")
        for i in range(3):
            print(f"  Row {i}: {predictions[i]}, max_prob={result['max_prob'][i]:.3f}")

        assert predictions[0] == "positive"
        assert predictions[1] == "negative"

        # Thinking text should be present
        assert "thinking_text" in result.columns
        for i in range(3):
            assert result["thinking_text"][i] is not None
            assert len(result["thinking_text"][i]) > 0
            print(f"  Row {i} thinking: {result['thinking_text'][i][:100]}...")


# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------


class TestRate:
    def test_direct(self, model, sample_df):
        result = prism.rate(
            sample_df,
            column_name="text",
            attributes="sentiment",
            model=model,
            scale_min=0,
            scale_max=10,  # small scale for speed
        )

        assert "expected_value" in result.columns
        assert "std_dev" in result.columns
        assert "mode" in result.columns
        assert "entropy" in result.columns
        assert result.shape[0] == 3

        print("\nRate (direct, torch):")
        for i in range(3):
            print(
                f"  Row {i}: expected={result['expected_value'][i]:.1f}, "
                f"mode={result['mode'][i]}, std={result['std_dev'][i]:.2f}"
            )

        # Positive text should rate higher than negative text
        assert result["expected_value"][0] > result["expected_value"][1]


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


class TestLabel:
    def test_direct(self, model, sample_df):
        result = prism.label(
            sample_df,
            column_name="text",
            labels={
                "positive_sentiment": "Expresses positive feelings or enthusiasm",
                "factual_statement": "States a fact without expressing opinion",
            },
            model=model,
        )

        assert "prob_true_positive_sentiment" in result.columns
        assert "predicted_positive_sentiment" in result.columns
        assert "prob_true_factual_statement" in result.columns
        assert "predicted_factual_statement" in result.columns

        print("\nLabel (direct, torch):")
        for i in range(3):
            print(
                f"  Row {i}: positive_sentiment={result['prob_true_positive_sentiment'][i]:.3f}, "
                f"factual={result['prob_true_factual_statement'][i]:.3f}"
            )

        # "I absolutely love this product" should be positive sentiment
        assert result["predicted_positive_sentiment"][0] is True
        # "The meeting is scheduled for 3pm" should be factual
        assert result["predicted_factual_statement"][2] is True
