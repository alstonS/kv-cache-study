# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project benchmarks KV-cache memory behavior for long-context LLM inference using HuggingFace Transformers. It measures peak GPU memory, generation time, and decode throughput across varying input lengths and configurations.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the baseline benchmark:**
```bash
python scripts/run_baseline.py
```

**Run KV quantization benchmark:**
```bash
python scripts/run_quant.py
```

**Run a parameter sweep:**
```bash
python scripts/run_sweep.py
```

**Plot results (after running a benchmark):**
```bash
python scripts/plot_results.py
```

All scripts must be run from the repository root (they use `sys.path.append(os.path.abspath("."))` to resolve `src/`).

## Architecture

The pipeline flows as:

1. **Config** (`configs/baseline.yaml`) — defines model, device, dtype, input lengths, token budget, trial counts, and output path.
2. **Script** (`scripts/run_baseline.py`) — loads config, instantiates model/tokenizer, runs warmup trials, then timed trials, and appends each row to a CSV.
3. **`src/` library modules** (imported by scripts):
   - `prompts.py` — `build_prompt_to_length(tokenizer, target_tokens)` pads a fixed BASE_TEXT by repetition then trims to exactly `target_tokens` ids.
   - `metrics.py` — `timed_generate()` resets GPU stats, calls `model.generate()` with `use_cache=True` / `do_sample=False`, synchronizes CUDA, and returns `(outputs, elapsed_sec)`. `get_peak_memory_mb()` reads `torch.cuda.max_memory_allocated()`.
   - `logger.py` — `append_result(csv_path, row)` appends a dict as a CSV row (creates file with header on first write).
   - `utils.py` — `DTYPE_MAP` maps string dtype names to `torch.*` dtypes.
   - `kv_quant.py` — quantization primitives (`quantize_int8/4/3`, `dequantize_int8/4/3`), `QuantizedKVCache` (subclass of `DynamicCache` that stores K/V chunks in compressed form and dequantizes on retrieval), and `timed_generate_quantized` (manual greedy decode loop using the custom cache).
4. **Plotting** (`scripts/plot_results.py`) — reads `results/raw/baseline.csv` and saves two PNG plots to `results/plots/`.

## Config fields (`configs/baseline.yaml`)

| Field | Description |
|---|---|
| `model_name` | HuggingFace model ID |
| `device` | `"cuda"` or `"cpu"` |
| `dtype` | `"float16"`, `"bfloat16"`, or `"float32"` |
| `input_lengths` | List of prompt token lengths to sweep |
| `max_new_tokens` | Tokens to generate per trial |
| `num_trials` | Timed trials per input length |
| `warmup_trials` | Warmup generations (short, not logged) |
| `output_csv` | Path for result CSV (auto-created) |

## KV Quantization design (`src/kv_quant.py`)

`QuantizedKVCache` subclasses `DynamicCache` and overrides `update()` / `get_seq_length()`. Each layer maintains a list of quantized chunks (one per `update()` call). On every call, new states are quantized and appended; all chunks are dequantized and concatenated for return to the attention kernel. This keeps in-memory K/V in compressed form while the attention computation always receives full-precision tensors.

Memory model per element vs FP16:

| `quant_bits` | Storage | Reduction vs FP16 |
|---|---|---|
| 8 | INT8 (1 byte) | 2× |
| 4 | packed uint8 (0.5 byte) | 4× |
| 3 | INT8 (1 byte) — 3-bit grid, 8 levels | same storage as INT8 |

`run_quant.py` uses a manual greedy decode loop (instead of `model.generate()`) to inject the custom cache. The CSV output adds a `quant_bits` column alongside the same fields as `baseline.csv`.

## Planned extensions (from docs/progress_notes.md)

- TTFT (time-to-first-token) measurement
- Prefill vs. decode phase separation
- OOM handling
- Larger model support
- KV quantization comparison
