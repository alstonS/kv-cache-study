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
from src.kv_quant import run_benchmark_trial_quantized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/quant.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    device = config["device"]
    dtype_name = config["dtype"]
    dtype = DTYPE_MAP[dtype_name]
    input_lengths = config["input_lengths"]
    max_new_tokens = config["max_new_tokens"]
    num_trials = config["num_trials"]
    warmup_trials = config["warmup_trials"]
    quant_bits = config["quant_bits"]
    output_csv = config["output_csv"]

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False.")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Requested device: {device}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Dtype: {dtype_name}")

    for input_len in input_lengths:
        print(f"\nRunning input length = {input_len}")

        prompt = build_prompt_to_length(tokenizer, input_len)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        actual_input_len = encoded["input_ids"].shape[1]
        print(f"Actual tokenized input length: {actual_input_len}")

        for nbits in quant_bits:
            print(f"  quant_bits={nbits}")

            for i in range(warmup_trials):
                print(f"    Warmup {i + 1}/{warmup_trials}")
                _ = run_benchmark_trial_quantized(
                    model=model,
                    inputs=encoded,
                    max_new_tokens=min(8, max_new_tokens),
                    device=device,
                    nbits=nbits,
                )

            for trial in range(num_trials):
                print(f"    Trial {trial + 1}/{num_trials}")
                result = run_benchmark_trial_quantized(
                    model=model,
                    inputs=encoded,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    nbits=nbits,
                )

                row = {
                    "model_name": model_name,
                    "device": device,
                    "dtype": dtype_name,
                    "input_length": actual_input_len,
                    "max_new_tokens": max_new_tokens,
                    "quant_bits": nbits,
                    "trial": trial + 1,
                    "total_time_sec": result["total_time_sec"],
                    "generated_tokens": result["generated_tokens"],
                    "tokens_per_sec": result["tokens_per_sec"],
                    "decode_tokens_per_sec": result["decode_tokens_per_sec"],
                    "peak_memory_mb": result["peak_memory_mb"],
                    "prefill_sec": result["prefill_sec"],
                    "decode_sec": result["decode_sec"],
                    "ttft_sec": result["ttft_sec"],
                    "oom": result["oom"],
                }

                append_result(output_csv, row)
                print(row)

    print(f"\nDone. Results saved to {output_csv}")


if __name__ == "__main__":
    main()
