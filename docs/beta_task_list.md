# PRISM Beta Task List

Active as of March 20, 2026.

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
7. Add tests proving the renamed `label()` path preserves the current binary-labeling behavior and output contract.
8. Remove `binary_classify()` from the documented public surface before beta so the top-level trio is just `classify`, `rate`, and `label`.

### Goal 3: Expand the correctness harness with property-based tests

Why this comes third:
The core risk is still distribution correctness at the prompt/token boundary. Property-based tests are a good fit because they can exercise many trie, cache, and token-boundary edge cases without depending on heavyweight models.

9. Add `hypothesis` to dev dependencies and create a new `tests/test_properties_*.py` entry point for core invariants.
10. Add property tests that generated trie label sets always produce normalized probabilities that sum to 1.
11. Add property tests that cached and uncached probability extraction agree for generated branch structures.
12. Add property tests for prefix-overlap labels so shorter labels always receive the correct terminal probability mass.
13. Add property tests for `find_split_point()` and BPE-guard behavior so cached prefixes never include unstable boundary tokens.
14. Add property tests for in-context tokenization showing that recovered label continuations are consistent with `prompt + label` tokenization.
15. Add property tests showing that label shuffling changes prompt order but not the final remapped distribution contract.

## Priority 1: Guardrails And Verified Paths

### Goal 4: Add minimal CI so the repo rejects obvious regressions automatically

Why this matters:
With agents in the loop, PRISM needs an automatic gatekeeper. The CI should stay small and cheap, but it should always answer whether a change broke the core contract.

16. Add a GitHub Actions workflow that runs `ruff` and the cheap correctness suite on every push and pull request.
17. Make the CI job use the canonical local commands from `AGENTS.md` so local and remote checks stay aligned.
18. Add a short contributor note documenting exactly what CI covers and what still requires manual opt-in verification.

### Goal 5: Define a blessed beta support matrix, with MLX first

Why this matters:
PRISM should make honest support claims. Beta should mean "these exact model/backend paths have been checked" rather than "any HuggingFace-compatible model probably works."

19. Pick the initial blessed MLX beta model and record it in the supported-model matrix.
20. Pick the initial blessed Torch compatibility model and record its caveats in the same matrix.
21. Add a small MLX smoke regression suite for the blessed model that checks `classify`, `rate`, and `label`.
22. Add or tighten the Torch smoke suite so it mirrors the MLX surface while remaining explicitly secondary.
23. Update the README language from broad compatibility claims to the verified support matrix plus "best effort" wording for unverified models.

## Priority 2: Researcher Workflow And GABRIEL-Style Ergonomics

### Goal 6: Implement a real run-directory contract (`save_dir` and `reset_files`)

Why this matters:
One of GABRIEL's best ideas is treating LLM work as a resumable, inspectable measurement run. PRISM should do the same, but with artifacts that make sense for local probability extraction.

24. Design and document the on-disk run layout for `save_dir`, including manifest, config, outputs, and raw artifacts.
25. Implement a manifest file that records model ID, backend, prompt settings, label definitions, and PRISM version for each run.
26. Implement persisted result outputs for `classify`, `rate`, and `label`.
27. Persist audit artifacts that help users stand behind results: rendered prompt text, label order, in-context label tokenization, and reasoning text when enabled.
28. Add `reset_files` support that safely clears or overwrites prior run artifacts.
29. Add resume/loading behavior so partially completed runs can restart without recomputing completed rows.

### Goal 7: Add operational tests, not just algorithm tests

Why this matters:
Once PRISM starts saving run artifacts, those workflows need tests of their own. GABRIEL is strong here, and PRISM should adopt the same mindset.

30. Add tests covering first-run save behavior for `classify`, `rate`, and `label`.
31. Add tests covering cached loading/resume behavior when result files already exist.
32. Add tests covering `reset_files=True` and `reset_files=False`.
33. Add tests covering manifest compatibility checks when a saved run conflicts with a changed model, backend, or task definition.

### Goal 8: Finalize GABRIEL-compatible parameter vocabulary where it helps users

Why this matters:
Compatibility should reduce friction, not force PRISM into an API that does not fit probability outputs. Reusing familiar parameter names is useful when semantics match.

34. Audit current top-level parameters and keep GABRIEL-aligned names where semantics already match, especially `column_name`, `additional_instructions`, and `save_dir`.
35. Introduce `reset_files` on the public task API once the run-directory contract exists.
36. Document which parameters are intentionally shared with GABRIEL and which differ because PRISM returns distributions instead of point estimates.

## Priority 3: Safe Flexibility And User Onboarding

### Goal 9: Add lightweight prompt customization with guardrails

Why this matters:
GABRIEL benefits from customizable templates. PRISM should support this carefully, because prompt-boundary correctness is part of its core contract.

37. Design a minimal template override API for `classify`, `rate`, and `label` that preserves required answer boundaries.
38. Add validation that custom templates still expose a stable answer slot for label/rating extraction.
39. Add tests for template override success and failure cases.
40. Document a small set of safe customization patterns built around `additional_instructions` first and template overrides second.

### Goal 10: Build a tutorial-first onboarding path and GABRIEL migration guide

Why this matters:
PRISM is aimed at researchers, not just engineers. The package should teach the happy path clearly and show familiar users how PRISM maps onto GABRIEL concepts.

41. Add a tutorial notebook that walks through `classify`, `rate`, and `label` on the blessed MLX model.
42. Add a short "GABRIEL -> PRISM" guide mapping task names, parameter names, and output differences.
43. Add an auditability walkthrough showing where to inspect saved prompts, tokenizations, model metadata, and output distributions.

## Explicitly Deferred Until After The Core Is Stable

These are worth revisiting later, but they should not interrupt the beta path above:

- New top-level task types beyond `classify`, `rate`, and `label`
- Rich result-object redesigns
- Cluster scheduling and distributed execution
- Broad reasoning-analysis surfaces beyond the current experimental path
- Large prompt-template systems beyond a tightly constrained override mechanism
