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
- [ ] Docs aligned to the new task-class structure.
- [ ] New audit-era tests added.

## Audit Order

### 1. Public API Surface

- [ ] Review [`src/prism/api.py`](/Users/jes0129/code/prism/src/prism/api.py)
- [ ] Review [`src/prism/__init__.py`](/Users/jes0129/code/prism/src/prism/__init__.py)
- [ ] Confirm the public contract we want to preserve during the audit

### 2. Shared Task Helpers

- [ ] Review [`src/prism/tasks/shared.py`](/Users/jes0129/code/prism/src/prism/tasks/shared.py)
- [ ] Confirm DataFrame handling, context normalization, prompt-boundary helpers, and probability-computer construction
- [ ] Decide whether any helper should move or be simplified before deeper task audits

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
