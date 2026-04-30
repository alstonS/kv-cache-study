# KV Cache Study

A benchmark comparing four KV cache strategies for serving Mistral 7B on an A100 40GB:
**FP16 baseline**, **INT8/INT4/INT3 quantization**, and **vLLM PagedAttention**. The goal
is to characterize the real world memory throughput tradeoff each method makes, so a
you can pick the right one for a given workload


## What we measured

For every method, input_length, decode_length combination we recorded:

- **Peak GPU memory** (model weights + KV cache + framework overhead)
- **Decode throughput** (tokens/sec generated)
- **Prefill throughput / TTFT** (time to first token is not used for vLLM)
- **OOM status** per trial

Three trials per condition, mean and std recorded.

## Setup

Tested on an NYU HPC A100 40GB node (CUDA-12.2.2, Python 3)

```bash
# For baseline + quant + analysis)
pip install -r requirements.txt

# vLLM for paged-attention benchmarks
pip install 'vllm==0.4.3'
```

## Running the benchmarks

Each script reads a YAML config under `configs_new/`. Outputs a CSV under `results2/raw/`
Three independent sweeps:

```bash
# 1. FP16 baseline (Mistral 7B, no KV quantization)
python scripts/run_baseline.py --config configs_new/baseline_gpu.yaml

# 2. KV cache quantization (INT8 / INT4 / INT3)
python scripts/run_quant.py --config configs_new/quant_gpu.yaml

# 3. PagedAttention via vLLM 
python scripts/run_paged.py --config configs_new/paged_gpu.yaml
```

The default sweep is `input_lengths: [512, 1024, 2048, 4096]` × `max_new_tokens: [32, 64, 128]` × 3 trials × 1 warmup


## Analysis pipeline

```bash
# Aggregates trials, computes derived metrics (memory savings %, throughput loss %,
# slopes, etc.), writes derived.csv + summary.csv to results2/processed/
python scripts/analyze_metric.py \
  --baseline-csv results2/raw/baseline_gpu.csv \
  --quant-csv    results2/raw/quant_gpu.csv \
  --paged-csv    results2/raw/paged_gpu.csv \
  --output-dir   results2/processed
```

## Generating plots

```bash
python scripts/plot_summary.py
python scripts/plot_derived.py
```

PNG outputs are located in `results2/plots/`


## Static dashboard
Install the Python dependencies, then launch the local Streamlit app from the
repository root:

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

The app reads the static CSVs and generated PNGs under `results2/`. It does not
require a GPU unless you choose to rerun the benchmark commands shown in the UI.


## Project layout

```
kv-cache-study/
├── configs_new/         # YAML sweep configs
├── scripts/             # Entry points: run_*, analyze, plot
├── src/                 # Library code: kv_quant, kv_paged, metrics, analysis
├── results2/
│   ├── raw/             # CSVs from each sweep (one row per trial)
│   ├── processed/       # derived.csv (one row per condition, with comparisons)
│   │                    # summary.csv (one row per method, headline metrics)
│   └── plots/           # PNG plots
└── app.py               # Streamlit dashboard
```

