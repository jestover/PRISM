# PRISM — Detailed Specification

**PRobabilistic Inference for Structured Measurement**

**Latest Update**: March 26, 2026

## Table of Contents

1. Overview and Goals
2. Relationship to GABRIEL
3. Package API Surface
4. Core Engine
5. Model Loading
6. Inference Backends
7. Prompt Templates
8. Caching Strategy
9. Distributed Computing
10. Output Format
11. Implementation Phases
12. File Structure
13. Dependencies
14. Implementation Status

---

## 1. Overview and Goals

### What PRISM Does

PRISM takes qualitative text data, runs it through a local LLM, and returns probability distributions over structured label sets. It supports three task types:

- **Classify**: Probability distribution over mutually exclusive labels (e.g., sentiment categories)
- **Rate**: Probability distribution over an integer scale (e.g., 0-100)
- **Label**: Independent P(true)/P(false) for each of multiple labels (e.g., topic tags)

### Why Probability Distributions Matter

API-based tools like GABRIEL return point estimates — a single label or number. PRISM returns the full distribution, which provides:

- **Uncertainty quantification**: Is the model confident (sharp peak) or uncertain (flat distribution)?
- **Richer downstream analysis**: Weight observations by model certainty; detect ambiguous cases
- **Distribution shape**: Two texts rated "50" may have very different distributions — one sharply peaked, one bimodal
- **No information loss**: The point estimate can always be recovered (mode or expected value), but the reverse is impossible

The analogy: GABRIEL runs a regression and gives you the coefficient. PRISM gives you the coefficient and the standard errors — and actually the full posterior distribution.

### Design Philosophy

- **GABRIEL-compatible**: Familiar API for researchers who know GABRIEL
- **Model-agnostic prompt handling**: Uses the model's tokenizer/chat template rather than hardcoded prompt adapters
- **Efficient**: Trie-based computation evaluates only at branch points, not every token
- **Single-machine first**: MLX and Torch are the active targets; distributed execution remains deferred
- **Simple**: `.format()` templates, clean abstractions, minimal dependencies

---

## 2. Relationship to GABRIEL

GABRIEL (Asirvatham, Mokski, & Shleifer 2026, "GPT as a Measurement Tool") is a Python library that wraps the OpenAI API to classify and rate qualitative data at scale. Key findings from their paper relevant to PRISM:

1. **Prompts barely matter**: 100 wildly different prompt variants produce near-identical ratings (r = 0.84-0.98). We don't need to perfectly replicate GABRIEL's Jinja2 templates.

2. **Attribute definitions barely matter**: Even undefined attributes produce similar ratings. GPT understands concepts like "populist" without explicit definitions.

3. **Small models ≈ large models**: gpt-5-mini nearly matches gpt-5 on 1000+ tasks. Reasoning capability matters more than size. This validates using local open-source models.

4. **No contamination bias**: Performance is identical pre- and post-training cutoff across hundreds of datasets.

5. **LLM labels ≈ human labels**: Model-Human correlation matches Human-Human correlation on subjective tasks.

### What PRISM Adds

GABRIEL calls the API and gets: `"42"`.
PRISM runs the model locally and gets: `{0: 0.001, ..., 42: 0.08, 43: 0.07, ..., 100: 0.001}`.

### Compatibility Goals

- Comparable task vocabulary: `rate()` should stay close to GABRIEL's `rate()`, and PRISM's planned `label()` corresponds most closely to GABRIEL's `classify()`
- Same input format: DataFrame with text column + label/attribute definitions
- Same output base: DataFrame with result columns (plus additional probability columns)
- Similar prompt structure: Based on GABRIEL's validated templates, adapted for direct label generation
- Documented differences: Clear examples showing GABRIEL vs. PRISM usage side by side

### GABRIEL Integration Path (Future)

GABRIEL's `get_all_responses()` accepts a `response_fn` parameter for dependency injection. A future adapter could let GABRIEL users swap in PRISM as a backend. This is not a priority but the architecture should not preclude it.

---

## 3. Package API Surface

Note on status:
The current implemented public API is `classify()`, `rate()`, and `label()`.

### 3.0 API Contract Summary

| Function | Task shape | Core input | Probability outputs | Summary outputs |
|----------|------------|------------|---------------------|-----------------|
| `classify()` | Mutually exclusive labels | `labels=[...]` | `prob_{label}` | `predicted_class`, `max_prob`, `entropy`, `thinking_text` |
| `rate()` | Integer scale distribution | `attribute=...`, `scale_min`, `scale_max` | `prob_{i}` | `expected_value`, `std_dev`, `mode`, `entropy`, `thinking_text` |
| `label()` | Independent true/false applicability | `labels={label: description}` | `prob_true_{label}` | `predicted_{label}`, `thinking_text_{label}` |

Notes:
- `thinking_text` columns are present only when `use_reasoning=True`.
- `label()` evaluates each label independently, so multiple labels can be true for the same row.

### 3.1 Model Loading

```python
import prism

model = prism.load_model(
    model_path="mlx-community/gpt-oss-20b-MXFP4-Q8",
    backend="mlx",              # "mlx", "torch", or "auto"
    think_end=None,             # token string marking end of reasoning, or None
    **backend_kwargs,           # passed to backend constructor
)
```

Returns a `Model` object bundling the inference backend, tokenizer (with chat template), and reasoning configuration. The model's built-in chat template handles all prompt structure — PRISM's templates only provide message content.

### 3.2 Classify

```python
result = prism.classify(
    df,                                     # Polars or Pandas DataFrame
    column_name="text",                     # column containing text to classify
    labels=["negative", "neutral", "positive"],
    label_descriptions=None,                # optional {label: description}
    model=model,
    use_reasoning=False,                    # chain-of-thought before answering
    max_thinking_tokens=2048,
    additional_instructions=None,           # appended to prompt (GABRIEL-compatible)
    context=None,                           # str (all rows) or list of per-row strings
    shuffle_labels=True,                    # randomize label order (debias)
    random_seed=None,                       # seed for label shuffling
    save_dir=None,                          # enable checkpointing (not yet implemented)
)
```

**Returns** DataFrame with original columns plus:
- `prob_{label}` for each label (float, sums to 1.0)
- `predicted_class` (str — argmax label)
- `max_prob` (float — probability of predicted class)
- `entropy` (float — Shannon entropy in bits)
- `thinking_text` (str — model's reasoning, only when `use_reasoning=True`)

### 3.3 Rate

```python
result = prism.rate(
    df,
    column_name="text",
    attribute="populism",
    attribute_description=None,
    scale_min=0,
    scale_max=100,
    model=model,
    use_reasoning=False,
    max_thinking_tokens=2048,
    additional_instructions=None,
    context=None,                           # str (all rows) or list of per-row strings
    random_seed=None,
    save_dir=None,                          # not yet implemented
)
```

**Returns** DataFrame with original columns plus:
- `prob_{i}` for each integer i in [scale_min, scale_max]
- `expected_value` (float — probability-weighted mean)
- `std_dev` (float — standard deviation)
- `mode` (int — most probable rating)
- `entropy` (float)
- `thinking_text` (str — model's reasoning, only when `use_reasoning=True`)

### 3.4 Label

```python
result = prism.label(
    df,
    column_name="text",
    labels={
        "toxic": "Contains toxic language directed at other users",
        "sarcastic": "Uses sarcasm or irony",
    },
    model=model,
    use_reasoning=False,
    max_thinking_tokens=2048,
    additional_instructions=None,
    context=None,                           # str (all rows) or list of per-row strings
    random_seed=None,
    save_dir=None,                          # not yet implemented
)
```

**Returns** DataFrame with original columns plus:
- `prob_true_{label}` for each label (float)
- `predicted_{label}` (bool — P(true) > 0.5)
- `thinking_text_{label}` (str — per-label reasoning, only when `use_reasoning=True`)

Each label is evaluated independently — multiple labels can be true simultaneously.

### 3.5 Multi-Attribute Rating

Status: deferred and not implemented in the current branch.

```python
results = prism.rate_multiple(
    df,
    column_name="text",
    attributes={
        "populism": "How populist is the rhetoric?",
        "optimism": "How optimistic about the future?",
    },
    model=model,
)
```

Each attribute rated independently with its own prompt and distribution.

---

## 4. Core Engine

### 4.1 How Probability Extraction Works

Given a prompt ending where the model would generate a label (e.g., `"...Sentiment: "`), PRISM computes the probability of each possible label by examining logits at that position.

PRISM's trie-based approach only evaluates at **branch points** where label token sequences diverge — far fewer model calls than the naive one-per-label approach.

### 4.2 Token Trie

**Location**: `prism/core/token_trie.py`

```python
@dataclass
class BranchPoint:
    prefix: List[int]                  # shared token prefix
    branches: Dict[int, List[str]]     # {next_token: [labels continuing with this token]}

@dataclass
class LabelTokenTrie:
    label_sequences: Dict[str, List[int]]
    branch_points: List[BranchPoint]

    def __init__(self, label_token_sequences: Dict[str, List[int]]):
        self.label_sequences = label_token_sequences
        self.branch_points = self._find_branch_points()
```

**Example** — labels tokenized as:
```
"positive": [1, 5, 9]
"negative": [1, 5, 10]
"neutral":  [2, 6, 8]
```

Branch points:
1. `prefix=[], branches={1: ["positive", "negative"], 2: ["neutral"]}`
2. `prefix=[1, 5], branches={9: ["positive"], 10: ["negative"]}`

**2 model calls instead of 3.**

For `rate` with scale 0-100: trie is built over 101 labels ("0" through "100"). Numbers sharing digit prefixes (e.g., "40"-"49") are handled efficiently.

### 4.3 Label Probability Computer

**Location**: `prism/core/label_probs.py`

```python
@dataclass
class COTResult:
    probabilities: Dict[str, float]
    thinking_text: str
    thinking_tokens: List[int]

class LabelProbabilityComputer:
    def __init__(
        self,
        label_token_sequences: Dict[str, List[int]],
        backend: InferenceBackend,
        decode: Optional[Callable] = None,
    ):
        self.backend = backend
        self.decode = decode
        self.trie = LabelTokenTrie(label_token_sequences)

    def compute_probabilities(self, prompt_tokens: List[int]) -> Dict[str, float]:
        """Compute P(label) for each label at branch points."""
        label_probs = {label: 1.0 for label in self.trie.label_sequences}

        for branch_point in self.trie.branch_points:
            current_tokens = prompt_tokens + branch_point.prefix
            logits = self.backend.get_logits(current_tokens)

            valid_tokens = list(branch_point.branches.keys())
            masked_logits = logits[valid_tokens]
            probs = self.backend.softmax(masked_logits)

            for token_id, prob in zip(valid_tokens, probs.tolist()):
                for label in branch_point.branches[token_id]:
                    label_probs[label] *= prob

        return label_probs

    def compute_probabilities_with_cot(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        max_thinking_tokens: int = 2048,
        use_cache: bool = True,
    ) -> COTResult:
        """Let model reason, then compute probabilities from post-thinking position."""
        thinking_tokens, full_tokens = self.backend.generate_until(
            prompt_tokens, stop_tokens, max_thinking_tokens, use_cache
        )
        thinking_text = self.decode(thinking_tokens)
        probabilities = self.compute_probabilities(full_tokens)
        return COTResult(probabilities, thinking_text, thinking_tokens)
```

The `LabelProbabilityComputer` takes an `InferenceBackend` instance (rather than individual callables), keeping the dependency clean. The optional `decode` callable (typically `tokenizer.decode`) is only needed for chain-of-thought mode.

### 4.4 Chain of Thought (COT)

For models that support reasoning:

1. Prompt formatted WITHOUT reasoning prefix → model thinks freely
2. Model generates tokens until it produces the reasoning prefix
3. Cache checkpointed at that position
4. Probability extraction runs from post-thinking position

When `use_reasoning=True`, the model's thinking text is captured and returned in a `thinking_text` column. Note that chain-of-thought often compresses the probability distribution, since the model frequently decides on a label during the thinking phase.

**Direct mode on reasoning models**: On reasoning-capable models, `apply_chat_template(add_generation_prompt=True)` puts the model into thinking mode. The API function `_direct_prompt_tokens()` appends the reasoning prefix tokens to the prompt, telling the model to skip thinking and answer directly. Without this, logits would be read from the thinking position and produce meaningless results.

---

## 5. Model Loading

### 5.1 The Model Class

The `Model` class bundles backend, tokenizer, and reasoning configuration as direct attributes (no intermediate `ModelSpec` wrapper):

```python
class Model:
    def __init__(self, backend, tokenizer, think_end=None,
                 think_end_tokens=None):
        self.backend = backend
        self.tokenizer = tokenizer
        self.think_end = think_end
        self.think_end_tokens = think_end_tokens

    @property
    def can_reason(self) -> bool:
        return self.think_end is not None

    def tokenize_prompt(self, system_message: str, user_message: str) -> List[int]:
        """Build prompt using model's chat template and tokenize."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer.encode(prompt)

    def tokenize_labels(self, labels: List[str]) -> Dict[str, List[int]]:
        """Tokenize each label into its token sequence."""
        return {
            label: self.tokenizer.encode(label, add_special_tokens=False)
            for label in labels
        }

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to a string."""
        return self.tokenizer.decode(tokens)
```

### 5.2 The load_model Function

```python
def load_model(
    model_path: str,
    backend: str = "auto",
    think_end: Optional[str] = None,
    **backend_kwargs,
) -> Model:
    """
    Load a model for use with PRISM.

    Args:
        model_path: HuggingFace model ID or local path.
        backend: "mlx", "torch", or "auto" (MLX on Apple Silicon, else Torch).
        think_end: Token string marking end of reasoning phase.
            None if model doesn't support chain-of-thought.
        **backend_kwargs: Passed to backend (e.g., device="cuda:0").
    """
```

### 5.3 Backend Auto-Detection

```python
def _detect_backend() -> str:
    try:
        import mlx.core
        return "mlx"
    except ImportError:
        pass
    return "torch"
```

### 5.4 No Hardcoded Model Configs

The model's built-in chat template (from HuggingFace tokenizer config) handles prompt formatting. Many models can work without adding code, but verified support should be documented separately for explicitly tested model/backend combinations. An optional small registry of well-tested models with their reasoning prefixes may be provided as a convenience.

---

## 6. Inference Backends

### 6.1 InferenceBackend ABC

```python
from abc import ABC, abstractmethod
from typing import Any, List, Union

class InferenceBackend(ABC):
    """Abstract base class for inference backends.

    Implementors provide: model inference, softmax, and KV cache management.
    Must expose logit-level access — API-only backends that return text cannot
    be used (the logits are the whole point of PRISM).
    """

    # ---- Inference ----

    @abstractmethod
    def get_logits(self, tokens: List[int]) -> Any:
        """Get logits for next token position (uncached)."""

    @abstractmethod
    def softmax(self, logits: Any) -> Any:
        """Apply softmax to logits."""

    @abstractmethod
    def argmax(self, logits: Any) -> int:
        """Get index of maximum logit."""

    # ---- Cache Management ----

    @abstractmethod
    def create_cache(self) -> Any:
        """Create empty KV cache."""

    @abstractmethod
    def forward(self, tokens: List[int], cache: Any = None) -> Union[Any, tuple]:
        """Forward pass. No cache → logits only. With cache → (logits, updated_cache)."""

    @abstractmethod
    def copy_cache(self, cache: Any) -> Any:
        """Deep copy cache for branching. Must produce independent copy."""

    @abstractmethod
    def cache_memory_bytes(self, cache: Any) -> int:
        """Estimate memory usage of cache in bytes."""

    @abstractmethod
    def cache_sequence_length(self, cache: Any) -> int:
        """Number of tokens in cache."""

    # ---- Generation (for COT) ----

    @abstractmethod
    def generate_until(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        max_tokens: int = 2048,
        use_cache: bool = True,
    ) -> tuple:
        """Generate tokens until stop sequence.
        Returns: (generated_tokens, full_sequence_tokens)
        """
```

### 6.2 Provided Backends

- **MLX** (`prism/backends/mlx.py`): Apple Silicon via `mlx-lm`. Existing code from the original project.
- **PyTorch** (`prism/backends/torch.py`): CUDA/CPU via `transformers`. Existing code, needs cache API.

### 6.3 Performance Notes

**`mx.compile()` (MLX)**: Benchmarked March 2026 on M-series Apple Silicon with `gpt-oss-20b-MXFP4-Q8`. Provides bit-identical outputs but negligible speedup (~1.01-1.03x). MLX's lazy evaluation already performs graph fusion internally — explicit compilation adds nothing meaningful. Not worth the complexity.

**`torch.compile()` (PyTorch)**: Benchmarked March 2026 on MPS (Apple Silicon) with `Phi-3-mini-4k-instruct`. Provides 2.7-3.2x speedup with `mode="reduce-overhead"`, but introduces small float16 rounding differences (max |diff| ~0.03 in logits). Cold start is negligible on MPS (~10ms). Needs re-benchmarking on NVIDIA CUDA GPUs where the gains may differ. Not yet implemented — waiting for CUDA benchmarks before committing.

**`torch.inference_mode()` vs `torch.no_grad()`**: Benchmarked March 2026 on MPS. `inference_mode` was ~9% *slower* than `no_grad` on MPS, likely due to MPS-specific overhead. Keeping `no_grad()` for now. Should re-test on CUDA.

### 6.4 Adding New Backends

Subclass `InferenceBackend`, implement all abstract methods, pass instance to `Model`. The backend must provide logit-level access.

---

## 7. Prompt Templates

### 7.1 Design Principles

- **`.format()` based** — no Jinja2 dependency
- **GABRIEL-compatible structure** — modeled on their validated templates
- **Label shuffling** — randomize label order per prompt to prevent position bias
- **Context-ready** — accept optional context block (used by applications like sentiment analysis)
- **Chat-template agnostic** — produce message *content* only; model's chat template handles formatting

### 7.2 Classify Template

```python
CLASSIFY_SYSTEM = """\
You are classifying text. For the text provided, select the single most \
appropriate label from the list below.

{label_descriptions_block}\
Treat the labels as mutually exclusive for this task. Base your judgment on \
the text itself. If label definitions are provided, use them to anchor your \
decision. If several labels seem plausible, choose the single best-fitting \
label.

{additional_instructions_block}\
Respond with ONLY the label name, exactly as written. No explanation."""

CLASSIFY_USER = """\
{context_block}\
Text: {text}

Label: """
```

Prompt ends with `"Label: "` — model's next token is the label.

### 7.3 Rate Template

```python
RATE_SYSTEM = """\
You are rating text on a specific attribute. For the text provided, assign a \
single integer rating on a {scale_min}-{scale_max} scale.

Attribute: {attribute}
{attribute_description_block}\

Rating scale:
{scale_min} = absent or not at all
{scale_max} = extreme or overwhelmingly present
Use the full range. Do not round to multiples of 5 or 10.
Consider intermediate values (e.g., 19, 67, 32) to capture nuance.
Measure only the direct signal of this attribute in the text itself. Do not \
infer the rating from related traits, broad impressions, or other implied \
attributes.
Extremes should be rare. Double-check before using values near the ends of the \
scale.

{additional_instructions_block}\
Respond with ONLY the integer. No explanation."""

RATE_USER = """\
{context_block}\
Text: {text}

Rating: """
```

### 7.4 Label Template

```python
LABEL_SYSTEM = """\
You are evaluating whether a specific label applies to the provided text.

Label: {label}
{label_description_block}\
Judge this label independently based only on direct evidence in the text. Do \
not infer it from other possible labels, related traits, or broad impressions.
If the evidence is absent, ambiguous, or too weak, respond false.

{additional_instructions_block}\
Respond with ONLY "true" if the label clearly applies, or "false" if it does \
not. \
No explanation."""

LABEL_USER = """\
{context_block}\
Text: {text}

Applies: """
```

### 7.5 PromptBuilder

```python
class PromptBuilder:
    def __init__(self, random_seed: Optional[int] = None):
        self.rng = random.Random(random_seed)

    def render_classify(self, text, labels, label_descriptions=None,
                        context=None, additional_instructions=None,
                        shuffle=True):
        """Returns (system_message, user_message).
        shuffle=True randomizes label order to prevent position bias.
        shuffle=False preserves the caller's label order.
        """
        ordered = list(labels)
        if shuffle:
            self.rng.shuffle(ordered)
        # ... format templates ...

    def render_rate(self, text, attribute, attribute_description=None,
                    scale_min=0, scale_max=100, context=None,
                    additional_instructions=None):
        """Returns (system_message, user_message)."""

    def render_label(self, text, label, label_description=None,
                               context=None, additional_instructions=None):
        """Returns (system_message, user_message)."""
```

### 7.6 Prompt → Token Pipeline

```
1. PromptBuilder.render_classify(text, labels)
   → (system_message, user_message)

2. model.tokenize_prompt(system_message, user_message)
   → applies chat template → tokenizes → prompt_tokens

3. model.tokenize_labels(labels)
   → {label: [token_ids]}

4. LabelProbabilityComputer(label_token_sequences, ...)
   → builds trie, finds branch points

5. computer.compute_probabilities(prompt_tokens)
   → {label: probability}
```

Step 2 is where model-specific formatting happens automatically. Everything else is model-agnostic.

---

## 8. Caching Strategy

### 8.1 Cache Levels

1. **Level 0**: pre-label prefix cache reused across label-order groups
2. **Level 1**: ordering-specific or fixed-prefix cache reused across rows
3. **Level 2**: per-row cache built from the row-specific text suffix
4. **Level 3**: branch-point cache reuse inside trie traversal

### 8.2 InferenceSession

```python
@dataclass
class CacheCheckpoint:
    cache: Any
    sequence_length: int

class InferenceSession:
    def __init__(self, backend: InferenceBackend, initial_cache=None):
        self.backend = backend
        self.cache = initial_cache or backend.create_cache()
        self._checkpoints: Dict[str, CacheCheckpoint] = {}

    def process(self, tokens: List[int]) -> Any:
        """Process tokens, extend cache, return logits."""
        logits, self.cache = self.backend.forward(tokens, self.cache)
        return logits

    def checkpoint(self, name: str):
        """Save current cache state."""
        self._checkpoints[name] = CacheCheckpoint(
            cache=self.backend.copy_cache(self.cache),
            sequence_length=self.backend.cache_sequence_length(self.cache),
        )

    def restore(self, name: str):
        """Restore to saved checkpoint."""
        self.cache = self.backend.copy_cache(self._checkpoints[name].cache)

    def fork(self) -> 'InferenceSession':
        """Create independent copy for branching."""
        return InferenceSession(self.backend, self.backend.copy_cache(self.cache))

    @property
    def memory_bytes(self) -> int:
        total = self.backend.cache_memory_bytes(self.cache)
        for cp in self._checkpoints.values():
            total += self.backend.cache_memory_bytes(cp.cache)
        return total
```

### 8.3 CascadingCache

```python
class CascadingCache:
    """
    Current implementation in src/prism/core/prompt_cache.py.

    Builds a 4-level cache hierarchy:
    - Level 0: pre-label prefix
    - Level 1: ordering-specific or fixed prefix
    - Level 2: row-specific suffix
    - Level 3: trie branch prefixes
    """
```

### 8.4 Memory Estimates

```
Per token ≈ 2 × num_layers × hidden_dim × 2 bytes (fp16)
20B model ≈ 2 × 40 × 6144 × 2 ≈ 1MB per token
500-token system prompt ≈ 500MB
```

---

## 9. Distributed Computing

### 9.1 JobScheduler ABC

```python
@dataclass
class JobConfig:
    job_name: str
    n_tasks: int
    work_dir: Path
    output_dir: Path
    model_path: str
    backend: str

    # Resources
    gpus_per_task: int = 1
    cpus_per_task: int = 4
    memory_gb: int = 32
    time_limit: str = "4:00:00"
    partition: Optional[str] = None

    # Environment
    conda_env: Optional[str] = None
    module_loads: List[str] = None
    setup_commands: List[str] = None

class JobScheduler(ABC):
    @abstractmethod
    def generate_script(self, config: JobConfig, task_index: int) -> str:
        """Generate submission script for one task."""

    @abstractmethod
    def submit(self, script_path: Path) -> str:
        """Submit job. Returns job ID."""

    @abstractmethod
    def status(self, job_id: str) -> str:
        """Returns 'pending', 'running', 'completed', 'failed'."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel job."""

    def submit_array(self, config: JobConfig) -> List[str]:
        """Submit all tasks. Returns list of job IDs."""
```

### 9.2 Provided Schedulers

- **SLURMScheduler**: `sbatch`/`squeue`/`scancel`
- **GridEngineScheduler**: `qsub`/`qstat`/`qdel` (for WRDS)
- **LocalScheduler**: Sequential subprocess execution

### 9.3 Distribution Strategy

1. Coordinator splits DataFrame into N shards
2. Each task loads model once, caches system prompt, processes its shard
3. Results written to per-task Parquet files
4. Coordinator merges shards after completion

```python
result = prism.classify(
    df,
    column_name="text",
    labels=["positive", "negative", "neutral"],
    model_path="openai/gpt-oss-20b",
    backend="torch",
    scheduler=SLURMScheduler(),
    scheduler_config=JobConfig(
        job_name="classify_job",
        n_tasks=8,
        gpus_per_task=1,
        partition="gpu",
    ),
)
```

When no scheduler is provided, execution is local and single-threaded.

### 9.4 Adding New Schedulers

Subclass `JobScheduler`, implement the four abstract methods, pass instance via `scheduler` parameter.

---

## 10. Output Format

### 10.1 DataFrame Columns

**Classify**:

| Column | Type | Description |
|--------|------|-------------|
| `prob_{label}` | float | Probability per label |
| `predicted_class` | str | Argmax label |
| `max_prob` | float | P(predicted_class) |
| `entropy` | float | Shannon entropy (bits) |
| `thinking_text` | str | Model's reasoning (COT only) |

**Rate**:

| Column | Type | Description |
|--------|------|-------------|
| `prob_{i}` | float | Probability per integer |
| `expected_value` | float | Probability-weighted mean |
| `std_dev` | float | Standard deviation |
| `mode` | int | Most probable rating |
| `entropy` | float | Shannon entropy (bits) |
| `thinking_text` | str | Model's reasoning (COT only) |

**Label**:

| Column | Type | Description |
|--------|------|-------------|
| `prob_true_{label}` | float | P(true) per label |
| `predicted_{label}` | bool | P(true) > 0.5 |
| `thinking_text_{label}` | str | Per-label reasoning (COT only) |

### 10.2 Checkpointing (Deferred Design)

Intended behavior when `save_dir` is implemented:
- Results saved incrementally to `{save_dir}/results.parquet`
- Processed rows identified by hash, skipped on resume
- Metadata file tracks progress, model info, config

### 10.3 GABRIEL Comparison

| GABRIEL column | PRISM equivalent | Additional PRISM columns |
|----------------|-----------------|--------------------------|
| `predicted_classes` | `predicted_class` | `prob_{label}`, `entropy`, `thinking_text` |
| `{attribute}` (0-100) | `expected_value` | `prob_{i}`, `std_dev`, `thinking_text` |
| `{label}` (bool) | `predicted_{label}` | `prob_true_{label}`, `thinking_text_{label}` |

---

## 11. Implementation Phases

### Phase 1: Core Engine Refactor ✓
- [x] Define `InferenceBackend` ABC in `prism/backends/base.py`
- [x] Migrate and refactor MLX backend to implement ABC
- [x] Migrate and refactor Torch backend to implement ABC (add cache API)
- [x] Implement `load_model()` with chat template support
- [x] Implement `Model` class (flattened — no ModelSpec wrapper)
- [x] Migrate `token_trie.py` and `label_probs.py` to `prism/core/`
- [x] Refactor `LabelProbabilityComputer` to accept `InferenceBackend` instead of individual callables
- [x] Verify tests pass with new structure
- [x] Packaging: pyproject.toml, pip-installable from GitHub

### Phase 2: Prompt Templates ✓
- [x] Write classify/rate/label templates
- [x] Implement `PromptBuilder` with label shuffling (controlled by `shuffle` parameter)
- [x] Test prompt rendering with gpt-oss-20b chat template
- [ ] Test prompt rendering with additional models' chat templates
- [x] Document comparison with GABRIEL's templates
- [x] Add prompt-focused tests for stable answer boundaries and calibration language

### Phase 3: High-Level API ✓
- [x] Implement `prism.classify()`, `prism.rate()`, `prism.label()`
- [x] DataFrame in → probability extraction → DataFrame out
- [x] Summary statistics (entropy, expected value, std dev)
- [x] Support both Polars and Pandas
- [x] Chain-of-thought support with `thinking_text` output columns
- [x] Direct mode on reasoning models (`_direct_prompt_tokens` pattern)
- [x] End-to-end tests passing (MLX backend, gpt-oss-20b)
- [ ] Checkpointing and resume

### Phase 4: Caching Infrastructure
- [x] Implement `CascadingCache`
- [x] Integrate cache-aware probability extraction
- [ ] Implement `InferenceSession`
- [ ] Benchmark/extend cache reuse

### Phase 5: Distributed Computing
- [ ] Implement `JobScheduler` ABC
- [ ] Implement `SLURMScheduler`
- [ ] Implement `GridEngineScheduler`
- [ ] Implement `LocalScheduler`
- [ ] Job splitting, submission, monitoring, merging

### Phase 6: Packaging and Documentation
- [x] pyproject.toml, pip-installable
- [ ] API documentation
- [ ] GABRIEL comparison examples
- [ ] Guide: adding models, backends, schedulers
- [ ] Tutorial notebook

---

## 12. File Structure

```
prism/
├── src/prism/
│   ├── __init__.py                     # Public API: classify, rate, label, load_model
│   ├── api.py                          # classify(), rate(), label()
│   ├── model.py                        # load_model(), Model
│   ├── backends/
│   │   ├── __init__.py                 # Exports InferenceBackend
│   │   ├── base.py                     # InferenceBackend ABC
│   │   ├── mlx.py                      # MLX backend (Apple Silicon)
│   │   └── torch.py                    # PyTorch backend (CUDA / CPU)
│   ├── core/
│   │   ├── __init__.py                 # Exports LabelTokenTrie, LabelProbabilityComputer, etc.
│   │   ├── token_trie.py               # LabelTokenTrie, BranchPoint
│   │   ├── label_probs.py              # LabelProbabilityComputer, COTResult
│   │   └── prompt_cache.py             # CascadingCache, split-point detection
│   ├── prompts/
│   │   ├── __init__.py                 # Exports PromptBuilder
│   │   └── templates.py                # Templates + PromptBuilder
│   ├── scheduling/
│   │   └── __init__.py                 # Placeholder (not yet implemented)
│   └── utils.py                        # Logging (GABRIEL-style), helpers
├── tests/
│   ├── test_token_trie.py              # Trie construction tests
│   ├── test_phase_a_correctness.py     # Lightweight fake-backend correctness tests
│   ├── test_api_e2e.py                 # End-to-end API tests (MLX)
│   └── test_api_e2e_torch.py           # End-to-end API tests (Torch/CPU)
├── claude.md                           # Project context (not in git)
├── spec.md                             # Detailed specification (not in git)
├── pyproject.toml
├── README.md
└── LICENSE
```

**Planned files** (not yet created):
- `src/prism/core/session.py` — InferenceSession, CacheCheckpoint
- `src/prism/scheduling/base.py` — JobScheduler ABC, JobConfig
- `src/prism/scheduling/slurm.py` — SLURM scheduler
- `src/prism/scheduling/grid_engine.py` — Grid Engine scheduler
- `src/prism/scheduling/local.py` — Local execution

---

## 13. Dependencies

### Core

```
polars          # DataFrame operations (primary)
pandas          # DataFrame compatibility
```

### Backends (user installs what they need)

```
# MLX (Apple Silicon)
mlx
mlx-lm

# PyTorch (CUDA / CPU)
torch
transformers
accelerate          # required for device_map support
```

**Note on MPS (Apple Silicon via PyTorch):** Models using Mixture of Experts (MoE) like gpt-oss-20b require `torch.histc()` which is not implemented for integer types on MPS. For these models, use `device="cpu"` (slow but works) or a CUDA GPU. The MLX backend is the recommended choice on Apple Silicon.

### Scheduling

No additional dependencies — uses subprocess for `sbatch`/`qsub`.

### Development

```
pytest
ruff
uv
```

---

## 14. Implementation Status

**Last Updated**: March 26, 2026

### Completed (Phases 1–3)

All code migrated from the sentiment analysis project, refactored, and working:

| File | Status | Notes |
|------|--------|-------|
| `src/prism/__init__.py` | ✓ | Public API: classify, rate, label, load_model, set_log_level |
| `src/prism/api.py` | ✓ | All three API functions, Polars + Pandas support, COT thinking_text |
| `src/prism/model.py` | ✓ | Flattened Model class, load_model with auto-detection |
| `src/prism/backends/base.py` | ✓ | InferenceBackend ABC |
| `src/prism/backends/mlx.py` | ✓ | MLX backend implementing ABC |
| `src/prism/backends/torch.py` | ✓ | PyTorch backend implementing ABC |
| `src/prism/core/token_trie.py` | ✓ | LabelTokenTrie with branch point detection |
| `src/prism/core/label_probs.py` | ✓ | LabelProbabilityComputer (takes InferenceBackend) |
| `src/prism/core/prompt_cache.py` | ✓ | CascadingCache for multi-level prompt reuse |
| `src/prism/prompts/templates.py` | ✓ | All templates + PromptBuilder with label shuffling |
| `src/prism/utils.py` | ✓ | GABRIEL-style console logging, PRISM_LOG_LEVEL env var |
| `tests/test_token_trie.py` | ✓ | 4 tests for trie construction |
| `tests/test_phase_a_correctness.py` | ✓ | Fake-backend tests for boundary absorption, terminal branches, and cache parity |
| `tests/test_api_e2e.py` | ✓ | 8 end-to-end tests (MLX, gpt-oss-20b) |
| `tests/test_api_e2e_torch.py` | ✓ | 3 direct-mode tests (Torch/CPU, gpt-oss-20b) |
| `pyproject.toml` | ✓ | pip-installable from GitHub |

### Key Design Decisions Made During Implementation

1. **Flattened Model class**: `ModelSpec` wrapper removed — `Model` holds `backend`, `tokenizer`, `think_end`, `think_end_tokens` directly.

2. **LabelProbabilityComputer takes InferenceBackend**: Instead of 4 individual callables (`get_logits`, `softmax`, `generate_until`, `decode`), the constructor takes a `backend: InferenceBackend` instance and an optional `decode` callable.

3. **Direct mode on reasoning models**: `_direct_prompt_tokens()` appends `think_end_tokens` to the prompt so reasoning models skip thinking and answer directly. Without this, `apply_chat_template(add_generation_prompt=True)` puts the model into thinking mode, producing meaningless logits.

4. **GABRIEL-style logging**: Console-only, controlled by `PRISM_LOG_LEVEL` env var (default: "warning") or `prism.set_log_level()`. No Rich dependency. Downstream projects add their own handlers to `logging.getLogger("prism")`.

5. **Polars/Pandas support**: Top-level `try/import` with `_is_polars()`/`_is_pandas()` helpers for consistent handling throughout.

6. **Phase A correctness contract**: label continuations are now derived in context, prompt-boundary absorption is handled consistently across direct/cached/reasoning paths, and trie terminal branches allocate probability mass to shorter prefix-overlap labels.

### Not Yet Implemented
- InferenceSession (checkpoint/restore/fork)
- Checkpointing and resume for long runs
- Job scheduling (SLURM, Grid Engine, Local)
- Multi-attribute rating (`rate_multiple`)
- Torch backend COT test (direct mode verified — classify, rate, label all pass on CPU)
- API documentation
- Tutorial notebook
