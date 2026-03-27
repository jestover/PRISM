# PRISM Agents

## Defaults

- Treat PRISM as an alpha, MLX-first research tool.
- Prioritize correctness over infrastructure.
- Treat Torch as supported but secondary to MLX.
- Treat reasoning as experimental; direct-mode distribution correctness comes first.
- Breaking changes are acceptable at this stage if they improve the long-term design.

## Source Of Truth

- Code and tests define actual behavior.
- [`docs/realignment.md`](/Users/jes0129/code/prism/docs/realignment.md) is the active roadmap.
- [`docs/overview.md`](/Users/jes0129/code/prism/docs/overview.md) is the agent-neutral project overview.
- [`README.md`](/Users/jes0129/code/prism/README.md) is the public summary.
- [`spec.md`](/Users/jes0129/code/prism/spec.md) is the detailed reference and must stay aligned with the code.
- `.claude/` is scratch space, not a canonical source of truth.

## Working Rules

- If behavior or API changes, update the relevant docs in the same change.
- Prefer lightweight fake-backend tests for core probability/tokenization logic.
- Keep large-model MLX/Torch tests as opt-in integration coverage.
- Do not start the Phase B API redesign until Phase A correctness work is stable and covered by regression tests.

## Audit Workflow

- The current beta audit proceeds section by section through the codebase with the user.
- Before adding new code, tests, or docs beyond already-approved structural work, explain the exact purpose of the next change and wait for user feedback.
- Add tests one at a time during the audit. For each proposed test, explain what contract it covers, why that structure is appropriate, and let the user react before implementing it.
- Keep legacy suites with an `_old` suffix until their replacements are reviewed and accepted.
- Keep the session checklist in [`docs/code_audit_checklist.md`](/Users/jes0129/code/prism/docs/code_audit_checklist.md) updated so progress survives across sessions.

## Canonical Commands

- `uv sync --extra dev`
- `uv run --extra dev pytest tests/test_token_trie.py tests/test_phase_a_correctness_old.py -q`
- `uv run --extra mlx --extra dev pytest tests/test_api_e2e.py -v -s`
- `uv run --extra torch --extra dev pytest tests/test_api_e2e_torch.py -v -s`
