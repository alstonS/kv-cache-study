"""
KV cache quantization: INT8, INT4 (packed), INT3 (INT8 storage, 3-bit grid).

Memory footprint vs FP16 per element:
  nbits=8  → 1 byte   (2× reduction  — symmetric INT8)
  nbits=4  → 0.5 byte (4× reduction  — two values packed per uint8)
  nbits=3  → 1 byte   (same as INT8  — 3-bit quantization grid, INT8 storage;
                        quantization accuracy of 3-bit without bit-packing complexity)
"""

import time
from typing import List, Optional, Tuple

import torch
from transformers import DynamicCache

from src.metrics import get_peak_memory_mb, reset_gpu_stats


def quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().max().clamp(min=1e-8) / 127.0
    q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantize_int4(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, bool]:
    """
    Pack two 4-bit signed values into each uint8 (low nibble first).
    Returns (packed_uint8, scale, orig_shape, was_padded).
    """
    scale = x.abs().max().clamp(min=1e-8) / 7.0
    q = (x / scale).round().clamp(-8, 7)
    q_uint = (q + 8).to(torch.uint8)          # shift signed [-8,7] → unsigned [0,15]
    flat = q_uint.reshape(-1)
    padded = flat.shape[0] % 2 != 0
    if padded:
        flat = torch.cat([flat, flat.new_zeros(1)])
    packed = (flat[0::2] & 0xF) | ((flat[1::2] & 0xF) << 4)
    return packed, scale, x.shape, padded


def dequantize_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    orig_shape: torch.Size,
    padded: bool,
) -> torch.Tensor:
    low  = (packed & 0xF).to(torch.float32)
    high = ((packed >> 4) & 0xF).to(torch.float32)
    flat = torch.stack([low, high], dim=1).reshape(-1)
    n = 1
    for s in orig_shape:
        n *= s
    if padded:
        flat = flat[:n]
    return ((flat - 8.0) * scale).reshape(orig_shape)


def quantize_int3(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    3-bit quantization grid (8 levels: -4..3), stored as INT8.
    Accuracy corresponds to 3-bit; memory is the same as INT8.
    """
    scale = x.abs().max().clamp(min=1e-8) / 3.0
    q = (x / scale).round().clamp(-4, 3).to(torch.int8)
    return q, scale


def dequantize_int3(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale



class QuantizedKVCache(DynamicCache):

    def __init__(self, nbits: int = 8) -> None:
        super().__init__()
        self.nbits = nbits
        # Per-layer lists of quantized chunk metadata dicts
        self._key_chunks:   List[List[dict]] = []
        self._value_chunks: List[List[dict]] = []
        self._seq_len:      List[int]        = []


    def _quant(self, x: torch.Tensor) -> dict:
        if self.nbits == 8:
            q, sc = quantize_int8(x)
            return {"q": q, "scale": sc, "shape": x.shape}
        elif self.nbits == 4:
            q, sc, sh, pad = quantize_int4(x)
            return {"q": q, "scale": sc, "shape": sh, "padded": pad}
        elif self.nbits == 3:
            q, sc = quantize_int3(x)
            return {"q": q, "scale": sc, "shape": x.shape}
        else:
            raise ValueError(f"Unsupported nbits={self.nbits}. Choose 3, 4, or 8.")

    def _dequant(self, meta: dict, dtype: torch.dtype) -> torch.Tensor:
        if self.nbits == 8:
            return dequantize_int8(meta["q"], meta["scale"]).to(dtype)
        elif self.nbits == 4:
            return dequantize_int4(
                meta["q"], meta["scale"], meta["shape"], meta["padded"]
            ).to(dtype)
        else:  # 3
            return dequantize_int3(meta["q"], meta["scale"]).to(dtype)

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._key_chunks) <= layer_idx:
            self._key_chunks.append([])
            self._value_chunks.append([])
            self._seq_len.append(0)


    def update(
        self,
        key_states:   torch.Tensor,
        value_states: torch.Tensor,
        layer_idx:    int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = key_states.dtype
        self._ensure_layer(layer_idx)

        # Quantize and store the new chunk
        self._key_chunks[layer_idx].append(self._quant(key_states))
        self._value_chunks[layer_idx].append(self._quant(value_states))
        self._seq_len[layer_idx] += key_states.shape[-2]

        # Dequantize all chunks and concatenate for attention
        k = torch.cat(
            [self._dequant(m, orig_dtype) for m in self._key_chunks[layer_idx]],
            dim=-2,
        )
        v = torch.cat(
            [self._dequant(m, orig_dtype) for m in self._value_chunks[layer_idx]],
            dim=-2,
        )
        return k, v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._seq_len):
            return 0
        return self._seq_len[layer_idx]

    def get_max_length(self) -> Optional[int]:
        return None


def run_benchmark_trial_quantized(
    model,
    inputs: dict,
    max_new_tokens: int,
    device: str,
    nbits: int,
    model_memory_mb: float = 0.0,
) -> dict:
    import time

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
        cache = QuantizedKVCache(nbits=nbits)

        prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        prefill_end = time.perf_counter()

        prefill_sec = prefill_end - prefill_start
        ttft_sec = prefill_sec

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token]

        cur_input_ids = next_token
        cur_attention_mask = attention_mask
        if cur_attention_mask is not None:
            ones = torch.ones(
                (cur_attention_mask.shape[0], 1),
                dtype=cur_attention_mask.dtype,
                device=cur_attention_mask.device,
            )
            cur_attention_mask = torch.cat([cur_attention_mask, ones], dim=1)

        decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                outputs = model(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                )
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
