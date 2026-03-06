"""Benchmark: mx.compile() and torch.compile() correctness and speedup.

Standalone script — does NOT modify the PRISM backends.
Tests compilation wrappers directly on the raw model objects.

Usage:
    # MLX only (Apple Silicon)
    uv run --extra mlx python tests/bench_compile.py --backend mlx

    # PyTorch only (MPS / CUDA / CPU)
    uv run --extra torch python tests/bench_compile.py --backend torch

    # Both
    uv run --extra mlx --extra torch python tests/bench_compile.py
"""

import argparse
import time
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def time_fn(fn: Callable, n: int, label: str) -> Tuple[float, List[float]]:
    """Call *fn* n times, return (mean_seconds, all_times)."""
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean = sum(times) / len(times)
    return mean, times


def report(name: str, base_mean: float, compiled_mean: float,
           base_times: List[float], compiled_times: List[float],
           max_diff: float, cold_start: float):
    """Print a formatted benchmark report."""
    speedup = base_mean / compiled_mean if compiled_mean > 0 else float("inf")
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Correctness (max |diff|):  {max_diff:.2e}")
    print(f"  Cold start (1st compiled): {cold_start:.3f}s")
    print(f"  Base mean:                 {base_mean * 1000:.1f}ms")
    print(f"  Compiled mean:             {compiled_mean * 1000:.1f}ms")
    print(f"  Speedup:                   {speedup:.2f}x")
    print(f"  Base range:                [{min(base_times)*1000:.1f}, {max(base_times)*1000:.1f}]ms")
    print(f"  Compiled range:            [{min(compiled_times)*1000:.1f}, {max(compiled_times)*1000:.1f}]ms")
    print()


# ---------------------------------------------------------------------------
# MLX benchmark
# ---------------------------------------------------------------------------

MLX_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"

def bench_mlx(n_iters: int = 10):
    """Benchmark mx.compile() on the MLX model."""
    import mlx.core as mx
    from mlx_lm import load

    print(f"Loading MLX model: {MLX_MODEL}")
    model, tokenizer = load(MLX_MODEL)

    # Create compiled version
    compiled_model = mx.compile(model)

    # Test inputs — short and medium
    short_text = "The economy is doing well."
    medium_text = (
        "The stock market experienced significant volatility today as investors "
        "reacted to the latest economic data. Several major technology companies "
        "reported earnings that exceeded expectations, while energy stocks declined "
        "amid falling oil prices."
    )

    for label, text in [("short (32 tok)", short_text), ("medium (~80 tok)", medium_text)]:
        tokens = tokenizer.encode(text)
        token_array = mx.array(tokens)[None, :]

        # --- Correctness ---
        base_logits = model(token_array)
        mx.eval(base_logits)

        compiled_logits = compiled_model(token_array)
        mx.eval(compiled_logits)

        diff = mx.abs(base_logits - compiled_logits)
        max_diff = float(mx.max(diff))

        # --- Cold start (already done above, but let's be explicit) ---
        # The first compiled call above WAS the cold start; measure a fresh one
        compiled_model_fresh = mx.compile(model)
        t0 = time.perf_counter()
        out = compiled_model_fresh(token_array)
        mx.eval(out)
        cold_start = time.perf_counter() - t0

        # --- Warm throughput ---
        # Warm up both paths
        for _ in range(3):
            out = model(token_array); mx.eval(out)
            out = compiled_model(token_array); mx.eval(out)

        def run_base():
            out = model(token_array)
            mx.eval(out)

        def run_compiled():
            out = compiled_model(token_array)
            mx.eval(out)

        base_mean, base_times = time_fn(run_base, n_iters, "base")
        compiled_mean, compiled_times = time_fn(run_compiled, n_iters, "compiled")

        report(f"MLX mx.compile — {label}", base_mean, compiled_mean,
               base_times, compiled_times, max_diff, cold_start)


# ---------------------------------------------------------------------------
# PyTorch benchmark
# ---------------------------------------------------------------------------

TORCH_MODEL = "microsoft/Phi-3-mini-4k-instruct"

def bench_torch(n_iters: int = 10):
    """Benchmark torch.compile() on the PyTorch model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading Torch model: {TORCH_MODEL} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(TORCH_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TORCH_MODEL, device_map=device, torch_dtype=torch.float16
    )
    model.eval()

    # Check torch.compile availability
    if not hasattr(torch, "compile"):
        print("torch.compile not available (requires PyTorch 2.0+), skipping")
        return

    compiled_model = torch.compile(model, mode="reduce-overhead")

    short_text = "The economy is doing well."
    medium_text = (
        "The stock market experienced significant volatility today as investors "
        "reacted to the latest economic data. Several major technology companies "
        "reported earnings that exceeded expectations, while energy stocks declined "
        "amid falling oil prices."
    )

    for label, text in [("short (32 tok)", short_text), ("medium (~80 tok)", medium_text)]:
        tokens = tokenizer.encode(text)
        token_tensor = torch.tensor(tokens, device=device).unsqueeze(0)

        # --- Correctness ---
        with torch.inference_mode():
            base_logits = model(token_tensor).logits[0, -1, :]
            compiled_logits = compiled_model(token_tensor).logits[0, -1, :]

        diff = torch.abs(base_logits.float() - compiled_logits.float())
        max_diff = float(diff.max())

        # --- Cold start ---
        compiled_fresh = torch.compile(model, mode="reduce-overhead")
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = compiled_fresh(token_tensor).logits
        cold_start = time.perf_counter() - t0

        # --- Warm throughput ---
        # Warm up
        with torch.inference_mode():
            for _ in range(3):
                _ = model(token_tensor).logits
                _ = compiled_model(token_tensor).logits

        def run_base():
            with torch.inference_mode():
                _ = model(token_tensor).logits

        def run_compiled():
            with torch.inference_mode():
                _ = compiled_model(token_tensor).logits

        base_mean, base_times = time_fn(run_base, n_iters, "base")
        compiled_mean, compiled_times = time_fn(run_compiled, n_iters, "compiled")

        report(f"Torch torch.compile — {label}", base_mean, compiled_mean,
               base_times, compiled_times, max_diff, cold_start)


# ---------------------------------------------------------------------------
# Also benchmark torch.inference_mode vs torch.no_grad
# ---------------------------------------------------------------------------

def bench_torch_inference_mode(n_iters: int = 30):
    """Compare torch.no_grad vs torch.inference_mode."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"\n--- inference_mode vs no_grad (device={device}) ---")
    tokenizer = AutoTokenizer.from_pretrained(TORCH_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TORCH_MODEL, device_map=device, torch_dtype=torch.float16
    )
    model.eval()

    tokens = tokenizer.encode("The economy is doing well today and the markets are up.")
    token_tensor = torch.tensor(tokens, device=device).unsqueeze(0)

    # Warm up
    with torch.no_grad():
        for _ in range(5):
            _ = model(token_tensor).logits

    def run_no_grad():
        with torch.no_grad():
            _ = model(token_tensor).logits

    def run_inference_mode():
        with torch.inference_mode():
            _ = model(token_tensor).logits

    no_grad_mean, no_grad_times = time_fn(run_no_grad, n_iters, "no_grad")
    inf_mode_mean, inf_mode_times = time_fn(run_inference_mode, n_iters, "inference_mode")

    speedup = no_grad_mean / inf_mode_mean if inf_mode_mean > 0 else float("inf")
    print(f"  no_grad mean:        {no_grad_mean * 1000:.1f}ms")
    print(f"  inference_mode mean: {inf_mode_mean * 1000:.1f}ms")
    print(f"  Speedup:             {speedup:.3f}x")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark compile optimizations")
    parser.add_argument(
        "--backend", choices=["mlx", "torch", "both"], default="both",
        help="Which backend to benchmark"
    )
    parser.add_argument("--iters", type=int, default=10, help="Iterations per benchmark")
    args = parser.parse_args()

    if args.backend in ("mlx", "both"):
        try:
            bench_mlx(args.iters)
        except ImportError:
            print("MLX not available, skipping")

    if args.backend in ("torch", "both"):
        try:
            bench_torch(args.iters)
            bench_torch_inference_mode(args.iters)
        except ImportError:
            print("PyTorch/transformers not available, skipping")
