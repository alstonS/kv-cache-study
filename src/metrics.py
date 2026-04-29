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


def measure_model_memory_mb(model, device: str) -> float:
    # Measure the GPU memory occupied by model weights only
    # records the fixed model cost separately from KV/activation,
    # dequantization buffers, any overhead etc
    # if cpu return 0.0
    if device != "cuda":
        return 0.0
    
    torch.cuda.synchronize()
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)


def run_benchmark_trial(model, inputs, max_new_tokens: int, device: str, model_memory_mb: float = 0.0):
    result = {
        "total_time_sec": 0.0,
        "generated_tokens": 0,
        "tokens_per_sec": 0.0,
        "decode_tokens_per_sec": 0.0,
        "model_memory_mb": model_memory_mb,
        "peak_memory_mb": 0.0,
        "prefill_sec": 0.0,
        "decode_sec": 0.0,
        "ttft_sec": 0.0,
        "oom": False,
    }

    try:
        reset_gpu_stats(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # Prefill pass
        start_prefill = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        end_prefill = time.perf_counter()

        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        prefill_sec = end_prefill - start_prefill
        ttft_sec = prefill_sec  # first token available after prefill

        generated = [next_token]
        decode_start = time.perf_counter()

        cur_input_ids = next_token
        cur_attention_mask = attention_mask
        if cur_attention_mask is not None:
            ones = torch.ones(
                (cur_attention_mask.shape[0], 1),
                dtype=cur_attention_mask.dtype,
                device=cur_attention_mask.device,
            )
            cur_attention_mask = torch.cat([cur_attention_mask, ones], dim=1)

        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                outputs = model(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            cur_input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(cur_input_ids)

            if cur_attention_mask is not None:
                ones = torch.ones(
                    (cur_attention_mask.shape[0], 1),
                    dtype=cur_attention_mask.dtype,
                    device=cur_attention_mask.device,
                )
                cur_attention_mask = torch.cat([cur_attention_mask, ones], dim=1)

        if device == "cuda":
            torch.cuda.synchronize()
        decode_end = time.perf_counter()

        decode_sec = decode_end - decode_start
        total_time_sec = prefill_sec + decode_sec
        generated_tokens = len(generated)
        peak_memory_mb = get_peak_memory_mb(device)
        tokens_per_sec = generated_tokens / total_time_sec if total_time_sec > 0 else 0.0
        decode_tokens_per_sec = generated_tokens / decode_sec if decode_sec > 0 else 0.0

        result.update({
            "total_time_sec": total_time_sec,
            "generated_tokens": generated_tokens,
            "tokens_per_sec": tokens_per_sec,
            "decode_tokens_per_sec": decode_tokens_per_sec,
            "peak_memory_mb": peak_memory_mb,
            "prefill_sec": prefill_sec,
            "decode_sec": decode_sec,
            "ttft_sec": ttft_sec,
        })

    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
        if device == "cuda":
            torch.cuda.empty_cache()

    return result