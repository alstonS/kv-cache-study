# KV Cache Study

This project benchmarks KV-cache memory behavior for long-context LLM inference.

## Baseline
The baseline uses standard HuggingFace Transformers generation with full-precision KV cache.

## Metrics
- Peak GPU memory
- Total generation time
- Decode throughput

## First run
```bash
python scripts/run_baseline.py
```

## Static dashboard
Install the Python dependencies, then launch the local Streamlit app from the
repository root:

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

The app reads the static CSVs and generated PNGs under `results2/`. It does not
require a GPU unless you choose to rerun the benchmark commands shown in the UI.

## Notes
- Start with a small model to debug the pipeline.
- After the baseline works, extend this harness with TTFT, prefill/decode separation, OOM handling, and larger models.
