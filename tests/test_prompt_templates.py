"""Tests for prompt template structure and answer boundaries."""

from prism.prompts.templates import PromptBuilder


def test_classify_template_has_exclusive_guidance_and_label_boundary():
    builder = PromptBuilder(random_seed=0)

    system_msg, user_msg = builder.render_classify(
        text="Example text",
        labels=["alpha", "beta"],
        label_descriptions={"alpha": "First", "beta": "Second"},
        shuffle=False,
    )

    assert "mutually exclusive" in system_msg
    assert "single best-fitting label" in system_msg
    assert user_msg.endswith("\n\nLabel: ")


def test_rate_template_has_direct_signal_guidance_and_rating_boundary():
    builder = PromptBuilder()

    system_msg, user_msg = builder.render_rate(
        text="Example text",
        attribute="populism",
        attribute_description="Strength of populist rhetoric",
    )

    assert "direct signal of this attribute" in system_msg
    assert "Extremes should be rare" in system_msg
    assert user_msg.endswith("\n\nRating: ")


def test_label_template_has_independent_judgment_and_false_default_boundary():
    builder = PromptBuilder()

    system_msg, user_msg = builder.render_label(
        text="Example text",
        label="toxic",
        label_description="Contains toxic language",
    )

    assert "Judge this label independently" in system_msg
    assert "respond false" in system_msg
    assert user_msg.endswith("\n\nApplies: ")
