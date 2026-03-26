# PRISM Beta Task List

Active as of March 26, 2026.

This document turns the current roadmap into a prioritized, commit-sized task list for moving PRISM from alpha to a trustworthy beta. The ordering is intentional: correctness and honest scope come first, usability and polish come after that.

## Priority 0: Correctness And Scope

### Goal 1: Lock the actual PRISM contract before adding more surface area

Why this comes first:
PRISM should only promise what the code and tests can defend. The first step toward beta is tightening the spec and removing ambiguity about what is implemented, what is deferred, and what is merely aspirational.

1. Tighten `spec.md` so the implemented API, deferred features, and planned beta changes are clearly separated.
2. Remove or clearly mark unsupported claims around `rate_multiple`, checkpointing, scheduling, and other deferred features in public docs.
3. Add a single "current behavior vs beta target" API table covering `classify`, `rate`, and `binary_classify` -> `label`.
4. Add a dedicated "supported-model matrix" doc section with the currently verified MLX and Torch paths, even if the matrix is initially small.

### Goal 2: Freeze the beta API around `classify`, `rate`, and `label`

Why this comes second:
Users cannot build trust if the top-level verbs keep shifting. PRISM should settle the public task names before beta and only expand after those names are stable and well tested.

5. Add `prism.label()` as the intended public name for independent yes/no labeling.
6. Rename the public API from `binary_classify()` to `label()` and update imports, docs, and examples in the same change.
7. Define the intended `label()` behavior and output contract directly in tests, without treating `binary_classify()` as the reference implementation.
8. Remove `binary_classify()` from the documented public surface before beta so the top-level trio is just `classify`, `rate`, and `label`.

### Goal 3: Align the core prompt templates with GABRIEL where it helps users

Why this comes third:
Prompt wording is part of the user-facing contract. Before beta, PRISM should intentionally decide how close its templates should be to GABRIEL's `rate` and `classify` prompts, while preserving PRISM-specific answer boundaries for probability extraction.

9. Review the current `rate`, `binary_classify`/`label`, and `classify` templates against GABRIEL's task prompts and document the meaningful similarities and differences.
10. Update the `rate` template so it is roughly equivalent in intent and guidance to GABRIEL's `rate` template, while still ending in a stable numeric answer slot.
11. Update the `label` template so it is roughly equivalent in intent and guidance to GABRIEL's `classify` template, while still ending in a stable true/false answer slot.
12. Update the PRISM `classify` template so it is clearly analogous to GABRIEL's task style but explicitly tailored for mutually exclusive label distributions.
13. Add prompt-focused tests covering rendered structure, required answer boundaries, and any new template invariants.

### Goal 4: Expand the correctness harness with property-based tests

Why this comes fourth:
The core risk is still distribution correctness at the prompt/token boundary. Property-based tests are a good fit because they can exercise many trie, cache, and token-boundary edge cases without depending on heavyweight models.

14. Add `hypothesis` to dev dependencies and create a new `tests/test_properties_*.py` entry point for core invariants.
15. Add property tests that generated trie label sets always produce normalized probabilities that sum to 1.
16. Add property tests that cached and uncached probability extraction agree for generated branch structures.
17. Add property tests for prefix-overlap labels so shorter labels always receive the correct terminal probability mass.
18. Add property tests for `find_split_point()` and BPE-guard behavior so cached prefixes never include unstable boundary tokens.
19. Add property tests for in-context tokenization showing that recovered label continuations are consistent with `prompt + label` tokenization.
20. Add property tests showing that label shuffling changes prompt order but not the final remapped distribution contract.

## Priority 1: Guardrails And Verified Paths

### Goal 5: Add minimal CI so the repo rejects obvious regressions automatically

Why this matters:
With agents in the loop, PRISM needs an automatic gatekeeper. The CI should stay small and cheap, but it should always answer whether a change broke the core contract.

21. Add a GitHub Actions workflow that runs `ruff` and the cheap correctness suite on every push and pull request.
22. Make the CI job use the canonical local commands from `AGENTS.md` so local and remote checks stay aligned.
23. Add a short contributor note documenting exactly what CI covers and what still requires manual opt-in verification.

### Goal 6: Define a blessed beta support matrix, with MLX first

Why this matters:
PRISM should make honest support claims. Beta should mean "these exact model/backend paths have been checked" rather than "any HuggingFace-compatible model probably works."

24. Pick the initial blessed MLX beta model and record it in the supported-model matrix.
25. Pick the initial blessed Torch compatibility model and record its caveats in the same matrix.
26. Add a small MLX smoke regression suite for the blessed model that checks `classify`, `rate`, and `label`.
27. Add or tighten the Torch smoke suite so it mirrors the MLX surface while remaining explicitly secondary.
28. Update the README language from broad compatibility claims to the verified support matrix plus "best effort" wording for unverified models.

## Priority 2: Researcher Workflow And GABRIEL-Style Ergonomics

### Goal 7: Implement a real run-directory contract (`save_dir` and `reset_files`)

Why this matters:
One of GABRIEL's best ideas is treating LLM work as a resumable, inspectable measurement run. PRISM should do the same, but with artifacts that make sense for local probability extraction.

29. Design and document the on-disk run layout for `save_dir`, including manifest, config, outputs, and raw artifacts.
30. Implement a manifest file that records model ID, backend, prompt settings, label definitions, and PRISM version for each run.
31. Implement persisted result outputs for `classify`, `rate`, and `label`.
32. Persist audit artifacts that help users stand behind results: rendered prompt text, label order, in-context label tokenization, and reasoning text when enabled.
33. Add `reset_files` support that safely clears or overwrites prior run artifacts.
34. Add resume/loading behavior so partially completed runs can restart without recomputing completed rows.

### Goal 8: Add operational tests, not just algorithm tests

Why this matters:
Once PRISM starts saving run artifacts, those workflows need tests of their own. GABRIEL is strong here, and PRISM should adopt the same mindset.

35. Add tests covering first-run save behavior for `classify`, `rate`, and `label`.
36. Add tests covering cached loading/resume behavior when result files already exist.
37. Add tests covering `reset_files=True` and `reset_files=False`.
38. Add tests covering manifest compatibility checks when a saved run conflicts with a changed model, backend, or task definition.

### Goal 9: Finalize GABRIEL-compatible parameter vocabulary where it helps users

Why this matters:
Compatibility should reduce friction, not force PRISM into an API that does not fit probability outputs. Reusing familiar parameter names is useful when semantics match.

39. Audit current top-level parameters and keep GABRIEL-aligned names where semantics already match, especially `column_name`, `additional_instructions`, and `save_dir`.
40. Introduce `reset_files` on the public task API once the run-directory contract exists.
41. Document which parameters are intentionally shared with GABRIEL and which differ because PRISM returns distributions instead of point estimates.

## Priority 3: Safe Flexibility And User Onboarding

### Goal 10: Add lightweight prompt customization with guardrails

Why this matters:
GABRIEL benefits from customizable templates. PRISM should support this carefully, because prompt-boundary correctness is part of its core contract.

42. Design a minimal template override API for `classify`, `rate`, and `label` that preserves required answer boundaries.
43. Add validation that custom templates still expose a stable answer slot for label/rating extraction.
44. Add tests for template override success and failure cases.
45. Document a small set of safe customization patterns built around `additional_instructions` first and template overrides second.

### Goal 11: Build a tutorial-first onboarding path and GABRIEL migration guide

Why this matters:
PRISM is aimed at researchers, not just engineers. The package should teach the happy path clearly and show familiar users how PRISM maps onto GABRIEL concepts.

46. Add a tutorial notebook that walks through `classify`, `rate`, and `label` on the blessed MLX model.
47. Add a short "GABRIEL -> PRISM" guide mapping task names, parameter names, and output differences.
48. Add an auditability walkthrough showing where to inspect saved prompts, tokenizations, model metadata, and output distributions.

## Explicitly Deferred Until After The Core Is Stable

These are worth revisiting later, but they should not interrupt the beta path above:

- New top-level task types beyond `classify`, `rate`, and `label`
- Rich result-object redesigns
- Cluster scheduling and distributed execution
- Broad reasoning-analysis surfaces beyond the current experimental path
- Large prompt-template systems beyond a tightly constrained override mechanism
