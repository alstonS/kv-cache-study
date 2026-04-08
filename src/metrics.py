import time
import torch


def reset_gpu_stats(device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_memory_mb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def run_benchmark_trial(model, inputs, max_new_tokens: int, device: str):
    result = {
        "total_time_sec": 0.0,
        "generated_tokens": 0,
        "tokens_per_sec": 0.0,
        "peak_memory_mb": 0.0,
        "oom": False,
    }

    try:
        reset_gpu_stats(device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        peak_mem_mb = get_peak_memory_mb(device)
        toks_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0

        result.update({
            "total_time_sec": elapsed,
            "generated_tokens": generated_tokens,
            "tokens_per_sec": toks_per_sec,
            "peak_memory_mb": peak_mem_mb,
        })

    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
        if device == "cuda":
            torch.cuda.empty_cache()

    return result