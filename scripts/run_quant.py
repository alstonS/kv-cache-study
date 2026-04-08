import os
import sys
import yaml
import torch

sys.path.append(os.path.abspath("."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts import build_prompt_to_length
from src.metrics import get_peak_memory_mb
from src.logger import append_result
from src.utils import DTYPE_MAP
from src.kv_quant import timed_generate_quantized


def main():
    with open("configs/quant.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_name     = config["model_name"]
    device         = config["device"]
    dtype          = DTYPE_MAP[config["dtype"]]
    input_lengths  = config["input_lengths"]
    max_new_tokens = config["max_new_tokens"]
    num_trials     = config["num_trials"]
    warmup_trials  = config["warmup_trials"]
    quant_bits     = config["quant_bits"]
    output_csv     = config["output_csv"]

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

    for input_len in input_lengths:
        prompt = build_prompt_to_length(tokenizer, input_len)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        actual_input_len = encoded["input_ids"].shape[1]
        print(f"\nInput length = {actual_input_len}")

        for nbits in quant_bits:
            print(f"  quant_bits={nbits}")

            for i in range(warmup_trials):
                print(f"    Warmup {i + 1}/{warmup_trials}")
                timed_generate_quantized(model, encoded, max_new_tokens=8, nbits=nbits)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            for trial in range(num_trials):
                print(f"    Trial {trial + 1}/{num_trials}")
                outputs, elapsed = timed_generate_quantized(
                    model, encoded, max_new_tokens, nbits
                )
                peak_mem_mb = get_peak_memory_mb()

                generated_tokens = outputs.shape[1] - actual_input_len
                toks_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0

                row = {
                    "model_name":       model_name,
                    "input_length":     actual_input_len,
                    "max_new_tokens":   max_new_tokens,
                    "quant_bits":       nbits,
                    "trial":            trial + 1,
                    "total_time_sec":   elapsed,
                    "generated_tokens": generated_tokens,
                    "tokens_per_sec":   toks_per_sec,
                    "peak_memory_mb":   peak_mem_mb,
                }

                append_result(output_csv, row)
                print(f"    {row}")

    print(f"\nDone. Results saved to {output_csv}")


if __name__ == "__main__":
    main()
