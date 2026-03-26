# PRISM Overview

## What PRISM Is

PRISM is a Python package for extracting full probability distributions from local LLMs over discrete label sets. Instead of returning a single class or score, it returns the probability assigned to every candidate label so downstream analysis can use uncertainty directly.

## Relationship To GABRIEL

GABRIEL wraps hosted LLM APIs and returns point estimates. PRISM is intentionally similar in spirit and surface area, but it runs local models and exposes distributional outputs.

Use GABRIEL when you want the simplest hosted workflow or unconstrained free-form answers.
Use PRISM when you need label probabilities, uncertainty metrics, or local/open-weight inference.

## Design Principles

- GABRIEL-like top-level API: `classify`, `rate`, and `label`
- Local inference only: no API-only backends
- Backend abstraction for MLX and PyTorch
- Efficiency through trie-based branch evaluation and prompt-cache reuse
- Minimal prompt templating and minimal dependencies

## Current Architecture

1. Load a HuggingFace-compatible model through MLX or PyTorch.
2. Render task prompts with `PromptBuilder`.
3. Tokenize labels as continuations of the actual prompt context.
4. Reuse prompt state through `CascadingCache` where possible.
5. Traverse only trie branch points to recover a label distribution.
6. Return DataFrame outputs with probability columns and summary statistics.

## Current Status

- Single-machine MLX and PyTorch support exists.
- Multi-level prompt caching is implemented.
- Phase A correctness work is the active priority: in-context label tokenization, prompt-boundary absorption, prefix-overlap label handling, and parity between cached and uncached extraction.
- Distributed scheduling, checkpoint/resume, and broader API redesign are deferred until correctness is stable.

## API Contract

| Function | Task shape | Core input | Core outputs | Summary columns |
|----------|------------|------------|--------------|-----------------|
| `classify` | Mutually exclusive labels | `labels=[...]` | `prob_{label}` | `predicted_class`, `max_prob`, `entropy` |
| `rate` | Integer scale distribution | `attribute=...`, `scale_min`, `scale_max` | `prob_{i}` | `expected_value`, `std_dev`, `mode`, `entropy` |
| `label` | Independent true/false applicability | `labels={label: description}` | `prob_true_{label}` | `predicted_{label}` |

When `use_reasoning=True`, `classify` and `rate` add `thinking_text`, and `label` adds `thinking_text_{label}`.

## Key Docs

- [`docs/realignment.md`](/Users/jes0129/code/prism/docs/realignment.md): active roadmap
- [`docs/beta_task_list.md`](/Users/jes0129/code/prism/docs/beta_task_list.md): prioritized beta task list
- [`docs/prompt_alignment.md`](/Users/jes0129/code/prism/docs/prompt_alignment.md): prompt-template comparison against GABRIEL and current alignment notes
- [`spec.md`](/Users/jes0129/code/prism/spec.md): detailed reference
- [`README.md`](/Users/jes0129/code/prism/README.md): public-facing summary
