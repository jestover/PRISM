# PRISM Prompt Alignment Notes

Active as of March 26, 2026.

This document compares PRISM's current prompt templates to GABRIEL's current prompt files and identifies the highest-value alignment work before beta.

## Sources Reviewed

- PRISM templates in [src/prism/prompts/templates.py](/Users/jes0129/code/prism/src/prism/prompts/templates.py)
- GABRIEL classification prompt: [classification_prompt.jinja2](https://github.com/openai/GABRIEL/blob/main/src/gabriel/prompts/classification_prompt.jinja2)
- GABRIEL ratings prompt: [ratings_prompt.jinja2](https://github.com/openai/GABRIEL/blob/main/src/gabriel/prompts/ratings_prompt.jinja2)

## Overall Takeaway

PRISM's current templates are clean and extraction-friendly, but they are much thinner than GABRIEL's prompt language. That simplicity is good for debugging, but before beta PRISM should add a bit more judgment guidance so researchers get prompts that feel closer to GABRIEL's workflow without weakening answer-boundary stability.

The main gap is not structure. The main gap is calibration language.

## Current PRISM Template Shape

### `rate`

Current strengths:

- Clear single-attribute framing
- Explicit integer-only response boundary
- Good existing calibration on using the full range and not rounding to 5s or 10s

Current limitations:

- Less explicit than GABRIEL about judging only the direct signal of the attribute
- Less explicit about avoiding indirect inference
- Less explicit about extremes being rare and deserving extra caution
- Less explicit about high-precision, holistic judgment expectations

### `label`

Current strengths:

- Very clean true/false extraction boundary
- Straightforward single-label evaluation setup
- Works naturally with PRISM's one-label-at-a-time probability extraction

Current limitations:

- Much less guidance than GABRIEL on what counts as applying
- No explicit "judge this label independently" language
- No explicit caution against cross-label inference or weak implication-based inference
- No explicit default-to-false-unless-supported guidance

### `classify`

Current strengths:

- Clear mutually exclusive task framing
- Stable answer boundary for direct label extraction
- Label descriptions already supported

Current limitations:

- Sparse compared with GABRIEL's more judgment-oriented style
- Does not explicitly tell the model to use label definitions as anchors when present
- Does not explicitly tell the model how to handle borderline cases among several plausible labels
- Does not emphasize choosing the single best-fitting label based on direct evidence in the text

## What GABRIEL Is Doing That PRISM Should Borrow

### From `ratings_prompt.jinja2`

Useful ideas to borrow:

- Judge each attribute independently
- Avoid indirect inference from related attributes
- Measure only the direct signal in the content
- Use the full range and intermediate values
- Treat extremes as rare and deserving extra caution
- Frame the task as requiring careful, holistic judgment

What PRISM should not copy literally:

- Multi-attribute JSON output instructions
- Repeated reminders about assessing every attribute, because PRISM currently rates one attribute per prompt

### From `classification_prompt.jinja2`

Useful ideas to borrow:

- Use definitions to anchor judgment when provided
- Judge each label on its own evidence
- Avoid cross-label inference
- Default to false unless the evidence for true is actually present
- Preserve label names exactly as written

What PRISM should not copy literally:

- JSON output instructions
- "Assess every label" language, because PRISM's `label()` already evaluates one label per prompt
- Pairwise differentiation logic, which is specific to one GABRIEL workflow

## Recommended Template Changes

### `rate`

Recommended changes:

- Add one sentence saying the rating should measure only the direct signal of the named attribute in the text.
- Add one sentence saying not to infer the rating from related traits or broad impressions.
- Add one sentence saying extremes should be used sparingly and double-checked.
- Keep the current numeric answer boundary exactly as-is: `Rating: `

Suggested direction:

PRISM should end up roughly as strict as GABRIEL on direct-signal measurement, but remain shorter because it only handles one attribute per prompt.

### `label`

Recommended changes:

- Add wording that the model should decide whether this specific label applies based on direct evidence in the text.
- Add wording that the label should be judged independently of other possible labels.
- Add wording that false is the correct answer when evidence is absent or too weak.
- Keep the current true/false answer boundary exactly as-is: `Applies: `

Suggested direction:

`label()` should feel like the single-label, extraction-safe analogue of GABRIEL's classify prompt.

### `classify`

Recommended changes:

- Add wording that label definitions should anchor judgment when provided.
- Add wording that the choice should be based on the text itself, not inferred from loosely related traits.
- Add wording that if several labels seem plausible, the model should choose the single best-fitting one.
- Keep the current exact-label answer boundary exactly as-is: `Label: `

Suggested direction:

`classify()` should not try to mimic GABRIEL's multi-label classify prompt too literally. It should instead borrow GABRIEL's calibration style while staying explicit that this is an exclusive one-label choice.

## Recommended Order Of Work

1. Update `label` first, because it is currently the sparsest relative to GABRIEL's classify guidance.
2. Update `rate` second, because the current template is already fairly strong and just needs sharper calibration language.
3. Update `classify` third, because it needs the smallest wording change but should still be brought into the same family.
4. Add prompt-focused tests that assert the stable answer boundaries remain `Label: `, `Rating: `, and `Applies: `.
