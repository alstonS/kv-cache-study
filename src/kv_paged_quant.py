"""
Paged KV cache (vLLM PagedAttention) + KV cache quantization combined.

Builds on src/kv_paged.py by explicitly exposing ``kv_cache_dtype`` so that
vLLM uses both non-contiguous paged memory blocks AND compressed KV values.

Supported kv_cache_dtype values (vLLM):
  "auto"       – same as model dtype, no quantization (default)
  "fp8"        – FP8 KV cache (requires Hopper/Ada GPU: H100, RTX 4090+)
  "fp8_e5m2"   – FP8 with 5-bit exponent
  "fp8_e4m3"   – FP8 with 4-bit mantissa (higher precision)

Usage:
    from src.kv_paged_quant import build_vllm_quant_engine, run_vllm_quant_trial

    llm = build_vllm_quant_engine(model_name="...", kv_cache_dtype="fp8")
    result = run_vllm_quant_trial(llm, tokenizer, prompt,
                                  batch_size=1, max_new_tokens=128, device="cuda")
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from transformers import PreTrainedTokenizerBase

from src.kv_paged import build_vllm_engine, run_vllm_trial


SUPPORTED_KV_DTYPES = ("auto", "fp8", "fp8_e5m2", "fp8_e4m3")


def build_vllm_quant_engine(
    model_name: str,
    kv_cache_dtype: str = "fp8",
    tensor_parallel_size: int = 1,
    dtype: str = "half",
    gpu_memory_utilization: float = 0.9,
    trust_remote_code: bool = True,
    **kwargs: Any,
):
    """
    Build a vLLM engine with PagedAttention + KV cache quantization.
    """
    if kv_cache_dtype not in SUPPORTED_KV_DTYPES:
        raise ValueError(
            f"kv_cache_dtype={kv_cache_dtype!r} not supported. "
            f"Choose from {SUPPORTED_KV_DTYPES}."
        )

    return build_vllm_engine(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        kv_cache_dtype=kv_cache_dtype,
        **kwargs,
    )


def run_vllm_quant_trial(
    llm,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    """
    Run a single generation trial on a vLLM engine built with KV quantization.
    """
    return run_vllm_trial(
        llm=llm,
        tokenizer=tokenizer,
        prompt=prompt,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )
