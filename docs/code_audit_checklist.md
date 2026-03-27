# PRISM Code Audit Checklist

This checklist tracks the collaborative beta audit. The rule for every section is:

1. Read the current code together.
2. Explain what it is doing in plain language.
3. Confirm or challenge the current behavior.
4. Make only the agreed change.
5. Add or replace tests only after discussing the exact test purpose.
6. Update docs once the contract for that section is settled.

## Current State

- [x] Thin public API wrappers now delegate to task classes.
- [x] Legacy correctness/prompt suites renamed with `_old` suffixes.
- [x] One opt-in MLX tokenizer-boundary regression test exists for the current blessed reasoning model.
- [ ] Docs aligned to the new task-class structure.
- [x] New audit-era tests added.

## Future Ideas Parking Lot

These are intentionally not part of the beta roadmap. They are ideas worth revisiting later and may never be implemented.

- A helper such as `build_context_window(...)` for constructing rolling sentence/section context outside the core task API
- An `execution_mode` argument for `rate()` and `label()` to support grouped prompt strategies beyond today's independent-per-item execution
- If grouped execution is ever added, consider matching shuffle controls for grouped labels/attributes

## Audit Order

### 1. Public API Surface

- [x] Review [`src/prism/api.py`](/Users/jes0129/code/prism/src/prism/api.py)
- [x] Review [`src/prism/__init__.py`](/Users/jes0129/code/prism/src/prism/__init__.py)
- [x] Confirm the public contract we want to preserve during the audit

### 2. Shared Task Helpers

- [x] Review [`src/prism/tasks/shared.py`](/Users/jes0129/code/prism/src/prism/tasks/shared.py)
- [x] Confirm DataFrame handling, context normalization, prompt-boundary helpers, and probability-computer construction
- [x] Decide whether any helper should move or be simplified before deeper task audits
- [ ] Add direct tests for `normalize_named_spec()` covering list/dict/string normalization and validation failures
- [ ] Add direct tests for `resolve_contexts()`, `is_context_constant()`, and `get_constant_context()`
- [ ] Add direct tests for `effective_prompt_tokens()` edge cases
- [ ] Complete the shared-helper follow-up tests before starting the `rate.py` audit

### 3. Classify Task

- [ ] Review [`src/prism/tasks/classify.py`](/Users/jes0129/code/prism/src/prism/tasks/classify.py)
- [ ] Trace direct-mode flow end to end
- [ ] Trace reasoning-mode flow end to end
- [ ] Confirm output-column contract
- [ ] Decide the first classify-specific test replacement

### 4. Prompt Construction

- [ ] Review [`src/prism/prompts/templates.py`](/Users/jes0129/code/prism/src/prism/prompts/templates.py)
- [ ] Confirm classify prompt shape and answer boundary
- [ ] Confirm rate prompt shape and answer boundary
- [ ] Confirm label prompt shape and answer boundary
- [ ] Decide prompt test migration order

### 5. Model Boundary Logic

- [ ] Review [`src/prism/model.py`](/Users/jes0129/code/prism/src/prism/model.py)
- [ ] Confirm prompt formatting behavior
- [ ] Confirm isolated vs. in-context tokenization behavior
- [ ] Confirm reasoning-model detection behavior

### 6. Prompt Cache

- [ ] Review [`src/prism/core/prompt_cache.py`](/Users/jes0129/code/prism/src/prism/core/prompt_cache.py)
- [ ] Confirm split-point logic
- [ ] Confirm Level 0/1/2 cache responsibilities
- [ ] Decide cache-specific test migration order

### 7. Trie Structure

- [ ] Review [`src/prism/core/token_trie.py`](/Users/jes0129/code/prism/src/prism/core/token_trie.py)
- [ ] Confirm branch-point construction
- [ ] Confirm terminal-branch behavior for prefix-overlap labels

### 8. Probability Engine

- [ ] Review [`src/prism/core/label_probs.py`](/Users/jes0129/code/prism/src/prism/core/label_probs.py)
- [ ] Confirm uncached probability traversal
- [ ] Confirm cached probability traversal
- [ ] Confirm chain-of-thought extraction path
- [ ] Decide probability-engine test migration order

### 9. Rate Task

- [ ] Review [`src/prism/tasks/rate.py`](/Users/jes0129/code/prism/src/prism/tasks/rate.py)
- [ ] Trace direct-mode flow end to end
- [ ] Trace reasoning-mode flow end to end
- [ ] Confirm summary-statistics contract

### 10. Label Task

- [ ] Review [`src/prism/tasks/label.py`](/Users/jes0129/code/prism/src/prism/tasks/label.py)
- [ ] Trace direct-mode flow end to end
- [ ] Trace reasoning-mode flow end to end
- [ ] Confirm independent true/false contract

### 11. Backend Interface

- [ ] Review [`src/prism/backends/base.py`](/Users/jes0129/code/prism/src/prism/backends/base.py)
- [ ] Confirm which backend guarantees are required by the core engine
- [ ] Confirm MLX-first support scope

### 12. MLX Backend

- [ ] Review [`src/prism/backends/mlx.py`](/Users/jes0129/code/prism/src/prism/backends/mlx.py)
- [ ] Confirm prompt forward path
- [ ] Confirm cache-copy behavior
- [ ] Confirm generation behavior for reasoning mode

### 13. Torch Status

- [ ] Review [`src/prism/backends/torch.py`](/Users/jes0129/code/prism/src/prism/backends/torch.py)
- [ ] Decide what stays in tree but is marked unsupported for this beta audit

### 14. Test Migration

- [ ] Review [`tests/test_phase_a_correctness_old.py`](/Users/jes0129/code/prism/tests/test_phase_a_correctness_old.py)
- [ ] Review [`tests/test_prompt_templates_old.py`](/Users/jes0129/code/prism/tests/test_prompt_templates_old.py)
- [ ] Review [`tests/test_token_trie.py`](/Users/jes0129/code/prism/tests/test_token_trie.py)
- [ ] Replace old suites one agreed test at a time

### 15. Docs Alignment

- [ ] Update [`docs/realignment.md`](/Users/jes0129/code/prism/docs/realignment.md)
- [ ] Update [`docs/overview.md`](/Users/jes0129/code/prism/docs/overview.md)
- [ ] Update [`README.md`](/Users/jes0129/code/prism/README.md)
- [ ] Update [`spec.md`](/Users/jes0129/code/prism/spec.md)

## Session Notes

- 2026-03-27: Public API and shared-task-helper audit sections are complete.
- 2026-03-27: `classify(labels=...)` now accepts list or dict. `label(labels=...)` now accepts string, list, or dict. `rate(attributes=...)` now accepts string, list, or dict and currently handles multiple attributes by running independent single-attribute prompts.
- 2026-03-27: Multi-attribute `rate()` keeps unprefixed columns for a single attribute and prefixes columns by attribute name only when multiple attributes are requested.
- 2026-03-27: Added [`tests/blessed_models.toml`](/Users/jes0129/code/prism/tests/blessed_models.toml) as the blessed-model registry for opt-in verification tests.
- 2026-03-27: Added [`tests/test_tokenization_boundaries_mlx.py`](/Users/jes0129/code/prism/tests/test_tokenization_boundaries_mlx.py) to verify reasoning-boundary tokenization against the blessed MLX tokenizer with no model forward pass.
- 2026-03-27: The `SwigPyPacked` / `SwigPyObject` / `swigvarlink` deprecation warnings come from the `sentencepiece` dependency on Python 3.12, not from PRISM. They are now suppressed narrowly in the tokenizer-boundary test module.
- Next resume point: finish the remaining [`src/prism/tasks/shared.py`](/Users/jes0129/code/prism/src/prism/tasks/shared.py) helper tests listed above, then audit [`src/prism/tasks/rate.py`](/Users/jes0129/code/prism/src/prism/tasks/rate.py) line by line, especially `_run_attribute()` and the multi-attribute output naming contract.
