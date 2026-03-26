# PRISM Realignment

## Status

Active as of March 26, 2026. Phase A is the only active implementation target. Phase B remains deferred until correctness is stable.

## Why This Exists

The current project risk is correctness at the prompt/token boundary, not missing infrastructure. The realignment work focuses on making label probabilities trustworthy before expanding the API surface or distributed execution story.

The prioritized beta work list now lives in [`docs/beta_task_list.md`](/Users/jes0129/code/prism/docs/beta_task_list.md). That document orders the concrete next steps from "must get right first" to "nice to have later."

## Phase A: Correctness

Implemented in the current branch:

- In-context continuation token derivation in `Model`
- Prompt-boundary absorption handling across uncached, cached, and reasoning paths
- Terminal trie branches for prefix-overlap labels
- Shared branch-probability logic for cached and uncached traversal
- Lightweight fake-backend/tokenizer regression tests

Still worth expanding before Phase B:

- More real-model regression cases across MLX and Torch
- More edge-case coverage for tokenizer-specific continuation behavior
- Broader validation on representative research tasks
- Prompt-template alignment so `rate` and `label` are meaningfully comparable to GABRIEL's task prompts and `classify` is analogous but PRISM-specific

## Phase B: Deferred

Do not start these until Phase A remains stable under regression coverage:

- Task-class API redesign
- `PrismResult` result objects
- Broader reasoning-mode analysis surfaces
- Scheduling, checkpointing, and resume infrastructure

## Testing Expectations

- Keep `tests/test_phase_a_correctness.py` and `tests/test_token_trie.py` as the cheapest correctness signal.
- Keep MLX/Torch e2e suites opt-in because they depend on heavyweight models.
- Add regression tests before refactoring probability extraction internals.
