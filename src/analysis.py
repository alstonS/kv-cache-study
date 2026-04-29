
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


# Loading csv 

def load_runs(baseline_csv: str | Path | None = None, quant_csv: str | Path | None = None,
    paged_csv: str | Path | None = None,) -> pd.DataFrame:

    frames = []

    if baseline_csv and Path(baseline_csv).exists():
        df = pd.read_csv(baseline_csv)
        df["method"] = "baseline_fp16"
        df['quant_bits'] = pd.NA
        frames.append(df)
    
    if quant_csv and Path(quant_csv).exists():
        df = pd.read_csv(quant_csv)
        df["method"] = df["quant_bits"].map(
            {8: "int8", 4: "int4", 3: "int3"}
        )
        frames.append(df)

    if paged_csv and Path(paged_csv).exists():
        df = pd.read_csv(paged_csv)
        # making sure frame work is vllm only
        if "framework" in df.columns:
            df = df[df["framework"].str.lower() == "vllm"].copy()
        
        df["method"] = "paged_vllm"
        df["quant_bits"] = pd.NA
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    # CSV FIXES
    # older CSVs don't have model_memory_mb
    if "model_memory_mb" not in out.columns:
        out["model_memory_mb"] = 0.0
    else:
        out["model_memory_mb"] = out["model_memory_mb"].fillna(0.0)

    # baseline/quant CSVs don't have batch_size — fill with 1
    if "batch_size" not in out.columns:
        out["batch_size"] = 1
    else:
        out["batch_size"] = out["batch_size"].fillna(1).astype(int)

    # engine_baseline_mb only in vLLM rows — fill missing with 0.0
    if "engine_baseline_mb" not in out.columns:
        out["engine_baseline_mb"] = 0.0
    else:
        out["engine_baseline_mb"] = pd.to_numeric(
            out["engine_baseline_mb"], errors="coerce"
        ).fillna(0.0)

    # aggregate_tokens_per_sec — use tokens_per_sec if missing
    if "aggregate_tokens_per_sec" not in out.columns:
        out["aggregate_tokens_per_sec"] = out["tokens_per_sec"]
    else:
        out["aggregate_tokens_per_sec"] = out["aggregate_tokens_per_sec"].fillna(
            out["tokens_per_sec"]
        )

    return out



# aggregate data

NUMERIC_AGG_COLS = [
    "total_time_sec",
    "generated_tokens",
    "tokens_per_sec",
    "decode_tokens_per_sec",
    "model_memory_mb",
    "peak_memory_mb",
    "prefill_sec",
    "decode_sec",
    "ttft_sec",
]

GROUP_KEYS = ["method", "input_length", "max_new_tokens"]

def _build_group_keys(df: pd.DataFrame) -> list:
    """Include batch_size in grouping only when multiple values exist.
    All baseline/quant runs are batch_size=1 so adding it would break the merge."""
    if "batch_size" in df.columns and df["batch_size"].nunique() > 1:
        return GROUP_KEYS + ["batch_size"]
    return list(GROUP_KEYS)

# drop oom and calculate mean and std for the non-oom trials
def aggregate_trials(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    gkeys = _build_group_keys(df)
    agg_cols = [c for c in NUMERIC_AGG_COLS if c in df.columns]

    success = (
        df.groupby(gkeys)["oom"]
        .apply(lambda s: (~s.astype(bool)).mean())
        .rename("success_rate")
        .reset_index()
    )

    clean = df[~df["oom"].astype(bool)].copy()

    if clean.empty:
        out = success.copy()
        for col in agg_cols:
            out[col] = np.nan
            out[col + "_std"] = np.nan
        return out

    means = (
        clean.groupby(gkeys)[agg_cols]
        .mean()
        .reset_index()
    )

    stds = (
        clean.groupby(gkeys)[agg_cols]
        .std(ddof=1)
        .fillna(0.0)
        .add_suffix("_std")
        .reset_index()
    )

    id_cols = [
        c for c in df.columns
        if c not in agg_cols + gkeys + ["trial", "oom"]
    ]
    if id_cols:
        ids = (
            clean.groupby(gkeys)[id_cols]
            .first()
            .reset_index()
        )
    else:
        ids = pd.DataFrame(columns=gkeys)

    out = means.merge(stds, on=gkeys).merge(success, on=gkeys)
    if not ids.empty:
        out = out.merge(ids, on=gkeys, how="left")

    return out



# Derviving metrics
# gpu_budget_mb using A100 40GB 
def derive_metrics(aggregated: pd.DataFrame, gpu_budget_mb: float = 40 * 1024) -> pd.DataFrame:
    if aggregated.empty:
        return aggregated

    df = aggregated.copy()

    
    # peak memory usage - model weights
    # to get approx generation time memory overhead
    # KV cache memory growth
    df["kv_cache_memory_mb"] = (
        df["peak_memory_mb"] - df["model_memory_mb"]
    ).clip(lower=0.0)


    # total tokens = prompt + generation
    total_tokens = df["input_length"] + df["generated_tokens"]


    # results are in bytes
    # Purpose: used to show lower quant bits have
    # reduced memory growth compared to fp16
    df["incremental_memory_bytes_per_token"] = (
        (df["kv_cache_memory_mb"] * 1024 ** 2) / total_tokens
    )
    # in mega-bytes for bigger picture
    df["kv_mb_per_token"] = df["kv_cache_memory_mb"] / total_tokens

    # token/s hard to visualize, flipped to
    # how many output tokens the model generates per second
    df["ms_per_decode_token"] = 1000.0 / df["decode_tokens_per_sec"]
    # Prefill throughput, higher is better. 
    df["prefill_tokens_per_sec"] = df["input_length"] / df["prefill_sec"]
    

    # value normalized by GPU memory usage
    # how many decode tokens per sec per GB of gpu memory
    # Higher is better. More throughput per VRAM used.
    df["decode_tokens_per_sec_per_gb"] = (
        df["decode_tokens_per_sec"] / (df["peak_memory_mb"] / 1024.0)
    )


    # estimate context length
    fixed_cost = df["model_memory_mb"]
    available_for_kv = (gpu_budget_mb - fixed_cost).clip(lower=0.0)
    df["max_feasible_context_tokens"] = (
        available_for_kv / df["kv_mb_per_token"].replace(0, np.nan)
    )


    # Get baseline rows out of df and rename columns with prefix to identify later
    baseline = df[df["method"] == "baseline_fp16"].copy()
    if baseline.empty:
        for col in [
            "memory_savings_mb", "memory_savings_pct",
            "throughput_ratio", "throughput_loss_pct",
            "decode_slowdown_factor",
            "extra_ms_per_token",
            "ttft_ratio", "prefill_throughput_ratio",
            "memory_saved_per_1pct_throughput_loss_mb",
        ]:
            df[col] = np.nan
        return df
    # rename for baseline
    baseline_cols = {
        "peak_memory_mb": "baseline_peak_memory_mb",
        "kv_cache_memory_mb": "baseline_kv_cache_memory_mb",
        "decode_tokens_per_sec": "baseline_decode_tokens_per_sec",
        "ttft_sec": "baseline_ttft_sec",
        "prefill_sec": "baseline_prefill_sec",
        "ms_per_decode_token": "baseline_ms_per_decode_token",
        "prefill_tokens_per_sec": "baseline_prefill_tokens_per_sec",
    }
    baseline_partial = baseline[
        ["input_length", "max_new_tokens"] + list(baseline_cols.keys())
    ].rename(columns=baseline_cols)

    df = df.merge(
        baseline_partial, on=["input_length", "max_new_tokens"], how="left"
    )
    
    # Now each row will have the baseline column numbers
    # Comepare baseline with other models

    # Memory
    df["memory_savings_mb"] = (
        df["baseline_peak_memory_mb"] - df["peak_memory_mb"]
    )
    df["memory_savings_pct"] = (
        100.0 * df["memory_savings_mb"] / df["baseline_peak_memory_mb"]
    )

    # Throughput
    df["throughput_ratio"] = (
        df["decode_tokens_per_sec"] / df["baseline_decode_tokens_per_sec"]
    )
    df["throughput_loss_pct"] = 100.0 * (1.0 - df["throughput_ratio"])
    # Inversed, easier to visualize when quant is slower
    df["decode_slowdown_factor"] = 1.0 / df["throughput_ratio"]


    # Latency
    df["extra_ms_per_token"] = (
        df["ms_per_decode_token"] - df["baseline_ms_per_decode_token"]
    )
    df["ttft_ratio"] = df["ttft_sec"] / df["baseline_ttft_sec"]
    df["prefill_throughput_ratio"] = (df["prefill_tokens_per_sec"]/ df["baseline_prefill_tokens_per_sec"])


    # Calculate how much memory saved per 1 percent throughput
    # ex: mem = 500mb, throughput loss = 50%, you get 10mb per 1% loss.
    # higher is better here
    df["memory_saved_per_1pct_throughput_loss_mb"] = (df["memory_savings_mb"]/ df["throughput_loss_pct"].clip(lower=0.01))


    # vllm metrics need to be added


    return df


def summarize_by_method(derived: pd.DataFrame) -> pd.DataFrame:

    if derived.empty:
        return derived

    # loop one method at a time, baseline, int8, int4, etc
    rows = []
    for method, g in derived.groupby("method"):
        row = {"method": method}

        # select largest input_length where trial did not oom
        successful_rows = g[g["success_rate"] >= 1.0]
        row["max_successful_input_length"] = (
            int(successful_rows["input_length"].max())
            if not successful_rows.empty
            else 0
        )


        # Memory and prefill slopes

        # memory_slope_mb_per_input_token:
        #   How much peak memory increases per extra input token.

        # prefill_slope_ms_per_input_token:
        #   How much prefill latency increases per extra input token
        #

        # Memory slope: pick a single max_new_tokens cross-section to
        # compare memory growth on a fixed decode lengt
        # pick the most common one
        mnt_mode = g["max_new_tokens"].mode()

        if len(mnt_mode):

            # Keep only rows with that selected max_new_tokens value,
            # then sort by input length so the first row is smallest context
            # and the last row is largest context
            chosen_mnt = mnt_mode.iloc[0]
            cross = (g[g["max_new_tokens"] == chosen_mnt].sort_values("input_length"))

            # need two input lengths to get a slope
            if len(cross) >= 2:
                # iloc[-1] = last and iloc[0] = first
                # largest input length and the smallest input length.

                # input length change
                dx = cross["input_length"].iloc[-1] - cross["input_length"].iloc[0]

                # peak memory change
                dy_mem = cross["peak_memory_mb"].iloc[-1] - cross["peak_memory_mb"].iloc[0]

                # prefill time change, in ms 
                dy_pf = (cross["prefill_sec"].iloc[-1] - cross["prefill_sec"].iloc[0]) * 1000.0

                # calc slope change in y / change in x
                row["memory_slope_mb_per_input_token"] = dy_mem / dx if dx else np.nan
                row["prefill_slope_ms_per_input_token"] = dy_pf / dx if dx else np.nan
            else:
                row["memory_slope_mb_per_input_token"] = np.nan
                row["prefill_slope_ms_per_input_token"] = np.nan
        else:
            row["memory_slope_mb_per_input_token"] = np.nan
            row["prefill_slope_ms_per_input_token"] = np.nan

        # Median metrics 
        if method != "baseline_fp16" and "memory_savings_pct" in g.columns:
            row["median_memory_savings_pct"] = g["memory_savings_pct"].median()
            row["median_throughput_loss_pct"] = g["throughput_loss_pct"].median()
            row["median_extra_ms_per_token"] = g["extra_ms_per_token"].median()
        else:
            row["median_memory_savings_pct"] = 0.0
            row["median_throughput_loss_pct"] = 0.0
            row["median_extra_ms_per_token"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)



# main 

def run_full_pipeline(
    baseline_csv: str | Path | None = None,
    quant_csv: str | Path | None = None,
    paged_csv: str | Path | None = None,
    gpu_budget_mb: float = 40 * 1024,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    raw = load_runs(baseline_csv, quant_csv, paged_csv)
    aggregated = aggregate_trials(raw)
    derived = derive_metrics(aggregated, gpu_budget_mb=gpu_budget_mb)
    summary = summarize_by_method(derived)
    return derived, summary








