import os
import sys
import yaml
import torch
import argparse

sys.path.append(os.path.abspath("."))

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.prompts import build_prompt_to_length
from src.logger import append_result
from src.utils import DTYPE_MAP
from src.kv_paged import (
    _normalize_max_new_tokens,
    build_vllm_engine,
    run_hf_trial,
    run_vllm_trial,
)


def _fmt_csv(v):
    return "" if v is None else v


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HuggingFace vs vLLM (PagedAttention); logs CSV like baseline/kv_quant."
    )
    parser.add_argument("--config", type=str, default="configs/paged_gpu.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    device = config["device"]
    dtype_name = config["dtype"]
    dtype = DTYPE_MAP[dtype_name]
    input_lengths = config["input_lengths"]
    max_new_tokens = _normalize_max_new_tokens(config["max_new_tokens"])
    num_trials = config["num_trials"]
    warmup_trials = config["warmup_trials"]
    batch_sizes = config.get("batch_sizes", [1])
    frameworks = config.get("frameworks", ["HuggingFace", "vLLM"])
    output_csv = config["output_csv"]
    overwrite_output = config.get("overwrite_output", False)

    tensor_parallel_size = config.get("tensor_parallel_size", 1)
    vllm_dtype = config.get("vllm_dtype", "half")
    vllm_gpu_memory_utilization = float(config.get("vllm_gpu_memory_utilization", 0.9))
    trust_remote_code = config.get("trust_remote_code", True)

    if overwrite_output and os.path.exists(output_csv):
        os.remove(output_csv)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False.")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def run_sweep_hf():
        print(f"\n=== Framework: HuggingFace ===")
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        model.eval()

        for input_len in input_lengths:
            print(f"\nRunning input length = {input_len}")
            prompt = build_prompt_to_length(tokenizer, input_len)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            actual_input_len = int(enc["input_ids"].shape[1])
            print(f"Actual tokenized input length: {actual_input_len}")

            for mnt in max_new_tokens:
                print(f"max_new_tokens = {mnt}")
                for bs in batch_sizes:
                    print(f"  batch_size = {bs}")

                    for w in range(warmup_trials):
                        print(f"    Warmup {w + 1}/{warmup_trials}")
                        _ = run_hf_trial(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            batch_size=bs,
                            max_new_tokens=min(8, mnt),
                            device=device,
                        )

                    for trial in range(num_trials):
                        print(f"    Trial {trial + 1}/{num_trials}")
                        result = run_hf_trial(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            batch_size=bs,
                            max_new_tokens=mnt,
                            device=device,
                        )
                        row = {
                            "model_name": model_name,
                            "device": device,
                            "dtype": dtype_name,
                            "framework": "HuggingFace",
                            "input_length": actual_input_len,
                            "max_new_tokens": mnt,
                            "batch_size": bs,
                            "trial": trial + 1,
                            "total_time_sec": result["total_time_sec"],
                            "generated_tokens": result["generated_tokens"],
                            "tokens_per_sec": result["tokens_per_sec"],
                            "decode_tokens_per_sec": result["decode_tokens_per_sec"],
                            "peak_memory_mb": result["peak_memory_mb"],
                            "prefill_sec": _fmt_csv(result.get("prefill_sec")),
                            "decode_sec": _fmt_csv(result.get("decode_sec")),
                            "ttft_sec": _fmt_csv(result.get("ttft_sec")),
                            "oom": result["oom"],
                        }
                        append_result(output_csv, row)
                        print(row)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    def run_sweep_vllm():
        if device != "cuda":
            print("Skipping vLLM: requires CUDA.")
            return
        print(f"\n=== Framework: vLLM (PagedAttention) ===")
        llm = build_vllm_engine(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=vllm_dtype,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )

        for input_len in input_lengths:
            print(f"\nRunning input length = {input_len}")
            prompt = build_prompt_to_length(tokenizer, input_len)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            actual_input_len = int(enc["input_ids"].shape[1])
            print(f"Actual tokenized input length: {actual_input_len}")

            for mnt in max_new_tokens:
                print(f"max_new_tokens = {mnt}")
                for bs in batch_sizes:
                    print(f"  batch_size = {bs}")

                    for w in range(warmup_trials):
                        print(f"    Warmup {w + 1}/{warmup_trials}")
                        _ = run_vllm_trial(
                            llm=llm,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            batch_size=bs,
                            max_new_tokens=min(8, mnt),
                            device=device,
                        )

                    for trial in range(num_trials):
                        print(f"    Trial {trial + 1}/{num_trials}")
                        result = run_vllm_trial(
                            llm=llm,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            batch_size=bs,
                            max_new_tokens=mnt,
                            device=device,
                        )
                        row = {
                            "model_name": model_name,
                            "device": device,
                            "dtype": dtype_name,
                            "framework": "vLLM",
                            "input_length": actual_input_len,
                            "max_new_tokens": mnt,
                            "batch_size": bs,
                            "trial": trial + 1,
                            "total_time_sec": result["total_time_sec"],
                            "generated_tokens": result["generated_tokens"],
                            "tokens_per_sec": result["tokens_per_sec"],
                            "decode_tokens_per_sec": result["decode_tokens_per_sec"],
                            "peak_memory_mb": result["peak_memory_mb"],
                            "prefill_sec": _fmt_csv(result.get("prefill_sec")),
                            "decode_sec": _fmt_csv(result.get("decode_sec")),
                            "ttft_sec": _fmt_csv(result.get("ttft_sec")),
                            "oom": result["oom"],
                        }
                        append_result(output_csv, row)
                        print(row)

        del llm
        if device == "cuda":
            torch.cuda.empty_cache()

    for fw in frameworks:
        if fw == "HuggingFace":
            run_sweep_hf()
        elif fw == "vLLM":
            run_sweep_vllm()
        else:
            raise ValueError(f"Unknown framework: {fw}")

    print(f"\nDone. Results saved to {output_csv}")


if __name__ == "__main__":
    main()
