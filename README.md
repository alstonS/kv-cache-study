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

## Notes
- Start with a small model to debug the pipeline.
- After the baseline works, extend this harness with TTFT, prefill/decode separation, OOM handling, and larger models.
