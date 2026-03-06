"""Benchmark: Cache reuse in compute_probabilities().

Current approach: `get_logits()` reprocesses the ENTIRE token sequence
(prompt + branch prefix) from scratch for every branch point.

Proposed approach: Process the shared prompt tokens once into a KV cache,
then fork the cache for each branch point and process only the short prefix.

This measures:
  1. Correctness: Do both approaches produce identical label probabilities?
  2. Speedup: How much faster is cache reuse?
  3. Scaling: How does speedup change with prompt length and label count?

Usage:
    uv run --extra mlx python tests/bench_cache_reuse.py
"""

import time
from typing import Dict, List, Tuple

import mlx.core as mx
from mlx_lm import load

from prism.core.token_trie import LabelTokenTrie


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

MODEL_PATH = "mlx-community/gpt-oss-20b-MXFP4-Q8"


def load_model():
    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Current approach (no cache — mirrors LabelProbabilityComputer)
# ---------------------------------------------------------------------------


def compute_probs_no_cache(
    model, prompt_tokens: List[int], trie: LabelTokenTrie
) -> Dict[str, float]:
    """Current approach: full reprocessing per branch point."""
    label_probs = {label: 1.0 for label in trie.label_sequences}

    for bp in trie.branch_points:
        current_tokens = prompt_tokens + bp.prefix
        token_array = mx.array(current_tokens)[None, :]
        logits = model(token_array)
        logits = logits[0, -1, :]

        valid_tokens = list(bp.branches.keys())
        masked = logits[valid_tokens]
        probs = mx.softmax(masked)
        mx.eval(probs)
        probs_list = probs.tolist()

        for token_id, prob in zip(valid_tokens, probs_list):
            for label in bp.branches[token_id]:
                label_probs[label] *= prob

    return label_probs


# ---------------------------------------------------------------------------
# Proposed approach (cache reuse)
# ---------------------------------------------------------------------------


def compute_probs_cached(
    model, prompt_tokens: List[int], trie: LabelTokenTrie
) -> Dict[str, float]:
    """Proposed approach: process prompt once, fork cache per branch point."""
    label_probs = {label: 1.0 for label in trie.label_sequences}

    # Step 1: Process the shared prompt tokens once
    prompt_array = mx.array(prompt_tokens)[None, :]
    base_cache = model.make_cache()
    base_logits = model(prompt_array, cache=base_cache)
    mx.eval(base_logits, *[c.state for c in base_cache])

    for bp in trie.branch_points:
        if len(bp.prefix) == 0:
            # No prefix — use logits from the prompt processing directly
            logits = base_logits[0, -1, :]
        else:
            # Fork the cache and process only the prefix tokens
            import copy
            branch_cache = copy.deepcopy(base_cache)
            prefix_array = mx.array(bp.prefix)[None, :]
            logits = model(prefix_array, cache=branch_cache)
            mx.eval(logits, *[c.state for c in branch_cache])
            logits = logits[0, -1, :]

        valid_tokens = list(bp.branches.keys())
        masked = logits[valid_tokens]
        probs = mx.softmax(masked)
        mx.eval(probs)
        probs_list = probs.tolist()

        for token_id, prob in zip(valid_tokens, probs_list):
            for label in bp.branches[token_id]:
                label_probs[label] *= prob

    return label_probs


# ---------------------------------------------------------------------------
# Tree-structured cache reuse (most efficient)
# ---------------------------------------------------------------------------


def compute_probs_tree_cached(
    model, prompt_tokens: List[int], trie: LabelTokenTrie
) -> Dict[str, float]:
    """Tree-structured cache: reuse cache along shared prefix paths.

    Instead of always forking from the base prompt cache, build caches
    incrementally along the trie paths. E.g. if branch points have
    prefixes [], [2], [2,4,3], we process:
      - prompt → cache_base  (used for prefix=[])
      - cache_base + [2] → cache_2  (used for prefix=[2])
      - cache_2 + [4,3] → cache_2_4_3  (used for prefix=[2,4,3])
    """
    import copy

    label_probs = {label: 1.0 for label in trie.label_sequences}

    # Process prompt once
    prompt_array = mx.array(prompt_tokens)[None, :]
    base_cache = model.make_cache()
    base_logits = model(prompt_array, cache=base_cache)
    mx.eval(base_logits, *[c.state for c in base_cache])

    # Build a mapping of prefix → (logits, cache) for incremental reuse
    # Key: tuple(prefix) for hashability
    prefix_cache = {(): (base_logits[0, -1, :], base_cache)}

    # Sort branch points by prefix length to process parents before children
    sorted_bps = sorted(trie.branch_points, key=lambda bp: len(bp.prefix))

    for bp in sorted_bps:
        prefix_tuple = tuple(bp.prefix)

        if prefix_tuple in prefix_cache:
            logits, _ = prefix_cache[prefix_tuple]
        else:
            # Find the longest cached prefix that is a parent of this prefix
            best_parent = ()
            for cached_prefix in prefix_cache:
                if (prefix_tuple[:len(cached_prefix)] == cached_prefix
                        and len(cached_prefix) > len(best_parent)):
                    best_parent = cached_prefix

            _, parent_cache = prefix_cache[best_parent]
            remaining = list(prefix_tuple[len(best_parent):])

            branch_cache = copy.deepcopy(parent_cache)
            remaining_array = mx.array(remaining)[None, :]
            logits_out = model(remaining_array, cache=branch_cache)
            mx.eval(logits_out, *[c.state for c in branch_cache])
            logits = logits_out[0, -1, :]

            prefix_cache[prefix_tuple] = (logits, branch_cache)

        valid_tokens = list(bp.branches.keys())
        masked = logits[valid_tokens]
        probs = mx.softmax(masked)
        mx.eval(probs)
        probs_list = probs.tolist()

        for token_id, prob in zip(valid_tokens, probs_list):
            for label in bp.branches[token_id]:
                label_probs[label] *= prob

    return label_probs


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def bench_scenario(
    model, tokenizer, name: str,
    system_text: str, user_text: str,
    labels: List[str],
    n_iters: int = 5,
):
    """Run one benchmark scenario."""
    # Build prompt tokens
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt)

    # Build trie
    label_seqs = {
        label: tokenizer.encode(label, add_special_tokens=False)
        for label in labels
    }
    trie = LabelTokenTrie(label_seqs)

    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"  prompt_tokens={len(prompt_tokens)}, labels={len(labels)}, "
          f"branch_points={len(trie.branch_points)}")
    print(f"  branch prefixes: {[len(bp.prefix) for bp in trie.branch_points]}")
    print(f"{'=' * 65}")

    # --- Correctness ---
    probs_base = compute_probs_no_cache(model, prompt_tokens, trie)
    probs_cached = compute_probs_cached(model, prompt_tokens, trie)
    probs_tree = compute_probs_tree_cached(model, prompt_tokens, trie)

    max_diff_cached = max(
        abs(probs_base[l] - probs_cached[l]) for l in probs_base
    )
    max_diff_tree = max(
        abs(probs_base[l] - probs_tree[l]) for l in probs_base
    )

    print(f"\n  Probabilities (no_cache):  {probs_base}")
    print(f"  Probabilities (cached):   {probs_cached}")
    print(f"  Probabilities (tree):     {probs_tree}")
    print(f"  Max |diff| cached:        {max_diff_cached:.2e}")
    print(f"  Max |diff| tree:          {max_diff_tree:.2e}")

    # --- Timing ---
    # Warm up
    for _ in range(2):
        compute_probs_no_cache(model, prompt_tokens, trie)
        compute_probs_cached(model, prompt_tokens, trie)
        compute_probs_tree_cached(model, prompt_tokens, trie)

    def time_fn(fn, n):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return sum(times) / len(times), times

    base_mean, base_times = time_fn(
        lambda: compute_probs_no_cache(model, prompt_tokens, trie), n_iters
    )
    cached_mean, cached_times = time_fn(
        lambda: compute_probs_cached(model, prompt_tokens, trie), n_iters
    )
    tree_mean, tree_times = time_fn(
        lambda: compute_probs_tree_cached(model, prompt_tokens, trie), n_iters
    )

    print(f"\n  No-cache mean:   {base_mean * 1000:.1f}ms  "
          f"[{min(base_times)*1000:.1f}, {max(base_times)*1000:.1f}]")
    print(f"  Cached mean:     {cached_mean * 1000:.1f}ms  "
          f"[{min(cached_times)*1000:.1f}, {max(cached_times)*1000:.1f}]")
    print(f"  Tree mean:       {tree_mean * 1000:.1f}ms  "
          f"[{min(tree_times)*1000:.1f}, {max(tree_times)*1000:.1f}]")
    print(f"  Speedup (cached vs no-cache): {base_mean / cached_mean:.2f}x")
    print(f"  Speedup (tree vs no-cache):   {base_mean / tree_mean:.2f}x")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    model, tokenizer = load_model()

    # --- Scenario 1: Classify (3 labels, short prompt) ---
    bench_scenario(
        model, tokenizer,
        name="Classify 3 labels — short prompt",
        system_text=(
            "You are classifying text. Select the single most appropriate label. "
            "Respond with ONLY the label name."
        ),
        user_text="I absolutely love this product, it's amazing!",
        labels=["positive", "negative", "neutral"],
    )

    # --- Scenario 2: Classify (3 labels, long prompt) ---
    bench_scenario(
        model, tokenizer,
        name="Classify 3 labels — long prompt",
        system_text=(
            "You are classifying text. Select the single most appropriate label "
            "from the list below. Consider the tone, context, and overall "
            "sentiment of the text carefully before making your decision. "
            "Think about whether the text expresses approval, disapproval, "
            "or is simply stating facts without emotional valence. "
            "Respond with ONLY the label name, exactly as written."
        ),
        user_text=(
            "The quarterly earnings report showed a mixed picture for the "
            "technology sector. While several major companies exceeded analyst "
            "expectations for revenue growth, concerns about rising costs and "
            "increased competition led to cautious forward guidance. Investors "
            "reacted with measured optimism, pushing indices higher in early "
            "trading before settling back to more modest gains by market close."
        ),
        labels=["positive", "negative", "neutral"],
    )

    # --- Scenario 3: Rate (0-10 scale = 11 labels) ---
    bench_scenario(
        model, tokenizer,
        name="Rate 0-10 scale — 11 labels",
        system_text=(
            "You are rating text on a specific attribute. Assign a single "
            "integer rating on a 0-10 scale. Respond with ONLY the number."
        ),
        user_text="This product is decent but could be better in some areas.",
        labels=[str(i) for i in range(11)],
    )

    # --- Scenario 4: Rate (0-100 scale = 101 labels, many branch points) ---
    bench_scenario(
        model, tokenizer,
        name="Rate 0-100 scale — 101 labels",
        system_text=(
            "You are rating text on a specific attribute. Assign a single "
            "integer rating on a 0-100 scale. Respond with ONLY the number."
        ),
        user_text="This product is decent but could be better in some areas.",
        labels=[str(i) for i in range(101)],
    )

    # --- Scenario 5: Multi-token labels with shared prefixes ---
    # These labels are multi-token and share prefixes, creating deep tries
    bench_scenario(
        model, tokenizer,
        name="Multi-token labels — shared prefixes",
        system_text=(
            "You are classifying text. Select the single most appropriate label. "
            "Respond with ONLY the label, exactly as written."
        ),
        user_text="I absolutely love this product, it's amazing and wonderful!",
        labels=[
            "strongly positive",
            "somewhat positive",
            "strongly negative",
            "somewhat negative",
            "strongly neutral",
            "somewhat neutral",
        ],
    )

    # --- Scenario 6: Many multi-token labels (stress test) ---
    bench_scenario(
        model, tokenizer,
        name="Multi-token labels — 12 labels, deep trie",
        system_text=(
            "You are classifying the political ideology of a text. "
            "Select the most appropriate label. "
            "Respond with ONLY the label, exactly as written."
        ),
        user_text=(
            "The government should invest heavily in public infrastructure "
            "and universal healthcare while ensuring fiscal responsibility "
            "and maintaining a strong national defense."
        ),
        labels=[
            "far left progressive",
            "center left progressive",
            "far left socialist",
            "center left liberal",
            "center right conservative",
            "far right conservative",
            "center right libertarian",
            "far right nationalist",
            "center moderate",
            "center left moderate",
            "center right moderate",
            "far left anarchist",
        ],
    )

    # --- Scenario 7: Synthetic worst case (force multi-token via raw IDs) ---
    # Manually construct label sequences that share long prefixes
    # to isolate the cache-reuse benefit independent of tokenization
    print(f"\n{'=' * 65}")
    print(f"  Synthetic: forced multi-token labels (bypassing tokenizer)")
    print(f"{'=' * 65}")

    # Use some arbitrary token IDs that would form shared prefixes
    # Simulate: 4 labels sharing a 3-token prefix, diverging at token 4
    vocab = tokenizer.get_vocab()
    # Pick some common token IDs
    common_tokens = list(range(1000, 1010))
    synthetic_labels = {
        "label_a": [common_tokens[0], common_tokens[1], common_tokens[2], common_tokens[3]],
        "label_b": [common_tokens[0], common_tokens[1], common_tokens[2], common_tokens[4]],
        "label_c": [common_tokens[0], common_tokens[1], common_tokens[5], common_tokens[6]],
        "label_d": [common_tokens[0], common_tokens[7], common_tokens[8], common_tokens[9]],
        "label_e": [common_tokens[5], common_tokens[6], common_tokens[7], common_tokens[8]],
    }
    trie = LabelTokenTrie(synthetic_labels)

    # Build a prompt
    messages = [
        {"role": "system", "content": "Classify the text."},
        {"role": "user", "content": "Test input text."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt)

    print(f"  prompt_tokens={len(prompt_tokens)}, labels=5, "
          f"branch_points={len(trie.branch_points)}")
    print(f"  branch prefixes: {[len(bp.prefix) for bp in trie.branch_points]}")

    n_iters = 5

    # Correctness
    p_base = compute_probs_no_cache(model, prompt_tokens, trie)
    p_cached = compute_probs_cached(model, prompt_tokens, trie)
    p_tree = compute_probs_tree_cached(model, prompt_tokens, trie)

    max_diff_c = max(abs(p_base[l] - p_cached[l]) for l in p_base)
    max_diff_t = max(abs(p_base[l] - p_tree[l]) for l in p_base)
    print(f"\n  Max |diff| cached: {max_diff_c:.2e}")
    print(f"  Max |diff| tree:   {max_diff_t:.2e}")

    # Timing (with warmup)
    for _ in range(2):
        compute_probs_no_cache(model, prompt_tokens, trie)
        compute_probs_cached(model, prompt_tokens, trie)
        compute_probs_tree_cached(model, prompt_tokens, trie)

    def time_fn_simple(fn, n):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return sum(times) / len(times), times

    bm, bt = time_fn_simple(lambda: compute_probs_no_cache(model, prompt_tokens, trie), n_iters)
    cm, ct = time_fn_simple(lambda: compute_probs_cached(model, prompt_tokens, trie), n_iters)
    tm, tt = time_fn_simple(lambda: compute_probs_tree_cached(model, prompt_tokens, trie), n_iters)

    print(f"\n  No-cache mean:   {bm * 1000:.1f}ms  [{min(bt)*1000:.1f}, {max(bt)*1000:.1f}]")
    print(f"  Cached mean:     {cm * 1000:.1f}ms  [{min(ct)*1000:.1f}, {max(ct)*1000:.1f}]")
    print(f"  Tree mean:       {tm * 1000:.1f}ms  [{min(tt)*1000:.1f}, {max(tt)*1000:.1f}]")
    print(f"  Speedup (cached vs no-cache): {bm / cm:.2f}x")
    print(f"  Speedup (tree vs no-cache):   {bm / tm:.2f}x")
    print()
