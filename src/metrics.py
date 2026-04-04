import time
import torch

def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def get_peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0

def timed_generate(model, inputs, max_new_tokens: int):
    reset_gpu_stats()

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    return outputs, elapsed
