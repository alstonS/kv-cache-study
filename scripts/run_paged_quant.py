"""
Benchmark: vLLM PagedAttention + KV cache quantization.

Sweeps over input lengths and kv_cache_dtype values, logging results to CSV.

Run from repo root:
    python scripts/run_paged_quant.py
    python scripts/run_paged_quant.py --config configs/paged_quant_gpu.yaml
"""

import os
import sys
import yaml
import torch
import argparse

sys.path.append(os.path.abspath("."))

from transformers import AutoTokenizer

from src.prompts import build_prompt_to_length
from src.logger import append_result
from src.kv_paged import _normalize_max_new_tokens
from src.kv_paged_quant import build_vllm_quant_engine, run_vllm_quant_trial


def _fmt_csv(v):
    return "" if v is None else v


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM PagedAttention + KV cache quantization."
    )
    parser.add_argument("--config", type=str, default="configs/paged_quant_gpu.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name               = config["model_name"]
    device                   = config["device"]
    dtype_name               = config["dtype"]
    input_lengths            = config["input_lengths"]
    max_new_tokens           = _normalize_max_new_tokens(config["max_new_tokens"])
    num_trials               = config["num_trials"]
    warmup_trials            = config["warmup_trials"]
    batch_sizes              = config.get("batch_sizes", [1])
    kv_cache_dtypes          = config.get("kv_cache_dtypes", ["auto", "fp8"])
    tensor_parallel_size     = config.get("tensor_parallel_size", 1)
    vllm_dtype               = config.get("vllm_dtype", "half")
    vllm_gpu_memory_util     = float(config.get("vllm_gpu_memory_utilization", 0.9))
    trust_remote_code        = config.get("trust_remote_code", True)
    output_csv               = config["output_csv"]
    overwrite_output         = config.get("overwrite_output", False)

    if device != "cuda":
        raise RuntimeError("This benchmark requires CUDA (vLLM only runs on GPU).")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False.")

    if overwrite_output and os.path.exists(output_csv):
        os.remove(output_csv)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for kv_dtype in kv_cache_dtypes:
        print(f"\n=== kv_cache_dtype: {kv_dtype} ===")

        llm = build_vllm_quant_engine(
            model_name=model_name,
            kv_cache_dtype=kv_dtype,
            tensor_parallel_size=tensor_parallel_size,
            dtype=vllm_dtype,
            gpu_memory_utilization=vllm_gpu_memory_util,
            trust_remote_code=trust_remote_code,
        )

        for input_len in input_lengths:
            print(f"\n  input_length = {input_len}")
            prompt = build_prompt_to_length(tokenizer, input_len)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            actual_input_len = int(enc["input_ids"].shape[1])
            print(f"  Actual tokenized length: {actual_input_len}")

            for mnt in max_new_tokens:
                print(f"  max_new_tokens = {mnt}")
                for bs in batch_sizes:
                    print(f"    batch_size = {bs}")

                    for w in range(warmup_trials):
                        print(f"      Warmup {w + 1}/{warmup_trials}")
                        run_vllm_quant_trial(
                            llm=llm,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            batch_size=bs,
                            max_new_tokens=min(8, mnt),
                            device=device,
                        )

                    for trial in range(num_trials):
                        print(f"      Trial {trial + 1}/{num_trials}")
                        result = run_vllm_quant_trial(
                            llm=llm,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            batch_size=bs,
                            max_new_tokens=mnt,
                            device=device,
                        )
                        row = {
                            "model_name":           model_name,
                            "device":               device,
                            "dtype":                dtype_name,
                            "framework":            "vLLM",
                            "kv_cache_dtype":       kv_dtype,
                            "input_length":         actual_input_len,
                            "max_new_tokens":       mnt,
                            "batch_size":           bs,
                            "trial":                trial + 1,
                            "total_time_sec":       result["total_time_sec"],
                            "generated_tokens":     result["generated_tokens"],
                            "tokens_per_sec":       result["tokens_per_sec"],
                            "decode_tokens_per_sec":result["decode_tokens_per_sec"],
                            "peak_memory_mb":       result["peak_memory_mb"],
                            "prefill_sec":          _fmt_csv(result.get("prefill_sec")),
                            "decode_sec":           _fmt_csv(result.get("decode_sec")),
                            "ttft_sec":             _fmt_csv(result.get("ttft_sec")),
                            "oom":                  result["oom"],
                        }
                        append_result(output_csv, row)
                        print(f"      {row}")

        del llm
        torch.cuda.empty_cache()

    print(f"\nDone. Results saved to {output_csv}")


if __name__ == "__main__":
    main()
