"""
HuggingFace Transformers vs vLLM (PagedAttention) benchmark helpers.

HF path (batch_size == 1) reuses the manual prefill/decode loop in ``metrics.run_benchmark_trial``
for comparable TTFT/prefill/decode fields. Larger batches use ``model.generate`` (single call),
so prefill/decode are not split.

vLLM path times ``llm.generate`` and uses request metrics, when available, for TTFT.
For analysis alignment, vLLM uses TTFT as prefill time and the remainder as decode time.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.metrics import get_peak_memory_mb, reset_gpu_stats, run_benchmark_trial


def _normalize_max_new_tokens(max_new_tokens: Union[int, List[int]]) -> List[int]:
    if isinstance(max_new_tokens, int):
        return [max_new_tokens]
    return list(max_new_tokens)


def run_hf_trial(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
    device: str,
    model_memory_mb: float = 0.0,
) -> Dict[str, Any]:
    """
    Greedy generation with HF. batch_size==1 matches baseline/kv_quant manual decode metrics.
    model_memory_mb: captured once after model.to(device), before any forward pass.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if batch_size == 1:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        return run_benchmark_trial(model, encoded, max_new_tokens, device,
                                   model_memory_mb=model_memory_mb)

    prompts = [prompt] * batch_size
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    result: Dict[str, Any] = {
        "total_time_sec": 0.0,
        "generated_tokens": 0,
        "tokens_per_sec": 0.0,
        "decode_tokens_per_sec": 0.0,
        "aggregate_tokens_per_sec": 0.0,
        "model_memory_mb": model_memory_mb,
        "peak_memory_mb": 0.0,
        "prefill_sec": None,
        "decode_sec": None,
        "ttft_sec": None,
        "oom": False,
    }

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    try:
        reset_gpu_stats(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                pad_token_id=pad_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        input_ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        input_lengths = attn.sum(dim=1).tolist()
        generated = 0
        for i in range(batch_size):
            ilen = int(input_lengths[i])
            generated += int(out[i].shape[0]) - ilen

        total_time = t1 - t0
        peak = get_peak_memory_mb(device)
        aggregate_tps = generated / total_time if total_time > 0 else 0.0
        per_request_tps = aggregate_tps / batch_size

        result.update(
            {
                "total_time_sec": total_time,
                "generated_tokens": generated,
                "tokens_per_sec": aggregate_tps,
                "decode_tokens_per_sec": per_request_tps,
                "aggregate_tokens_per_sec": aggregate_tps,
                "peak_memory_mb": peak,
            }
        )
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
        if device == "cuda":
            torch.cuda.empty_cache()

    return result


def build_vllm_engine(
    model_name: str,
    tensor_parallel_size: int = 1,
    dtype: str = "half",
    gpu_memory_utilization: float = 0.9,
    trust_remote_code: bool = True,
    **kwargs: Any,
):
    """Lazy-import vLLM so CPU-only installs can still import this module."""
    try:
        from vllm import LLM  # type: ignore
    except ImportError as e:
        raise ImportError(
            "vLLM is required for PagedAttention benchmarks. "
            "Install with: pip install -r requirements-paged.txt (CUDA Linux recommended)."
        ) from e

    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def run_vllm_trial(
    llm,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
    device: str,
    model_memory_mb: float = 0.0,
) -> Dict[str, Any]:
    """Single batched generate call via vLLM.

    model_memory_mb: weight-only memory, same concept as baseline/quant.
    ttft_sec: first-token latency from vLLM request metrics when exposed.
    prefill_sec: set equal to ttft_sec for comparison with baseline/quant.
    decode_sec: total generation time minus ttft_sec.
    """
    try:
        from vllm import SamplingParams  # type: ignore
    except ImportError as e:
        raise ImportError("vLLM is not installed.") from e

    prompts: List[str] = [prompt] * batch_size
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    result: Dict[str, Any] = {
        "total_time_sec": 0.0,
        "generated_tokens": 0,
        "tokens_per_sec": 0.0,
        "decode_tokens_per_sec": 0.0,
        "aggregate_tokens_per_sec": 0.0,
        "model_memory_mb": model_memory_mb,
        "peak_memory_mb": 0.0,
        "prefill_sec": None,
        "decode_sec": None,
        "ttft_sec": None,
        "oom": False,
    }

    try:
        reset_gpu_stats(device)
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        generated = 0
        ttfts = []
        for req in outputs:
            for comp in req.outputs:
                tid = getattr(comp, "token_ids", None)
                if tid is not None:
                    generated += len(tid)
                else:
                    # Fallback: approximate from text (slower, rare)
                    text = getattr(comp, "text", "") or ""
                    generated += len(tokenizer.encode(text, add_special_tokens=False))

            metrics = getattr(req, "metrics", None)
            arrival_time = getattr(metrics, "arrival_time", None)
            first_token_time = getattr(metrics, "first_token_time", None)
            if arrival_time is not None and first_token_time is not None:
                ttfts.append(first_token_time - arrival_time)

        total_time = t1 - t0
        ttft = float(sum(ttfts) / len(ttfts)) if ttfts else None
        decode_sec = total_time - ttft if ttft is not None else None
        peak = get_peak_memory_mb(device)
        aggregate_tps = generated / total_time if total_time > 0 else 0.0
        per_request_tps = aggregate_tps / batch_size

        result.update(
            {
                "total_time_sec": total_time,
                "generated_tokens": generated,
                "tokens_per_sec": aggregate_tps,
                "decode_tokens_per_sec": per_request_tps,
                "aggregate_tokens_per_sec": aggregate_tps,
                "peak_memory_mb": peak,
                "prefill_sec": ttft,
                "decode_sec": decode_sec,
                "ttft_sec": ttft,
            }
        )
    except Exception as exc:
        # vLLM may raise its own OOM / workspace errors
        msg = str(exc).lower()
        if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
            result["oom"] = True
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            raise

    return result
