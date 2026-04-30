from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results2"
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"
PLOTS_DIR = RESULTS_DIR / "plots"

PLOT_OPTIONS = {
    "Baseline FP16 derived metrics": {
        "path": PLOTS_DIR / "baseline_fp16_derived_metrics.png",
        "raw_command": "python3 scripts/run_baseline.py --config configs_new/baseline_gpu.yaml",
        "plot_command": "python3 scripts/plot_derived.py",
    },
    "INT3 derived metrics": {
        "path": PLOTS_DIR / "int3_derived_metrics.png",
        "raw_command": "python3 scripts/run_quant.py --config configs_new/quant_gpu.yaml",
        "plot_command": "python3 scripts/plot_derived.py",
    },
    "INT4 derived metrics": {
        "path": PLOTS_DIR / "int4_derived_metrics.png",
        "raw_command": "python3 scripts/run_quant.py --config configs_new/quant_gpu.yaml",
        "plot_command": "python3 scripts/plot_derived.py",
    },
    "INT8 derived metrics": {
        "path": PLOTS_DIR / "int8_derived_metrics.png",
        "raw_command": "python3 scripts/run_quant.py --config configs_new/quant_gpu.yaml",
        "plot_command": "python3 scripts/plot_derived.py",
    },
    "Paged vLLM derived metrics": {
        "path": PLOTS_DIR / "paged_vllm_derived_metrics.png",
        "raw_command": "python3 scripts/run_paged.py --config configs_new/paged_gpu.yaml",
        "plot_command": "python3 scripts/plot_derived.py",
    },
    "Summary comparison": {
        "path": PLOTS_DIR / "summary_comparison.png",
        "raw_command": (
            "python3 scripts/analyze_metric.py "
            "--baseline-csv results2/raw/baseline_gpu.csv "
            "--quant-csv results2/raw/quant_gpu.csv "
            "--paged-csv results2/raw/paged_gpu.csv"
        ),
        "plot_command": "python3 scripts/plot_summary.py",
    },
}


def _show_data_status() -> None:
    raw_files = [
        RAW_DIR / "baseline_gpu.csv",
        RAW_DIR / "quant_gpu.csv",
        RAW_DIR / "paged_gpu.csv",
    ]
    processed_files = [
        PROCESSED_DIR / "derived.csv",
        PROCESSED_DIR / "summary.csv",
    ]

    with st.expander("Static data files", expanded=False):
        st.write("Raw GPU CSVs:")
        for path in raw_files:
            st.write(f"- `{path.relative_to(ROOT)}` {'found' if path.exists() else 'missing'}")

        st.write("Processed CSVs:")
        for path in processed_files:
            st.write(f"- `{path.relative_to(ROOT)}` {'found' if path.exists() else 'missing'}")


def _show_preview_tables() -> None:
    summary_path = PROCESSED_DIR / "summary.csv"
    if not summary_path.exists():
        return

    with st.expander("Summary table", expanded=False):
        summary = pd.read_csv(summary_path)
        st.dataframe(summary, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="KV Cache Study", layout="wide")

    st.title("KV Cache Study")
    st.write(
        "Static dashboard for the GPU benchmark outputs in `results2/`. "
        "Use the dropdown to inspect each generated plot and the commands that reproduce it."
    )

    selected = st.selectbox("Select plot", list(PLOT_OPTIONS.keys()))
    plot_info = PLOT_OPTIONS[selected]
    plot_path = plot_info["path"]

    left, right = st.columns([2, 1])
    with left:
        st.subheader(selected)
        if plot_path.exists():
            st.image(str(plot_path), use_container_width=True)
        else:
            st.error(f"Missing plot: `{plot_path.relative_to(ROOT)}`")

    with right:
        st.subheader("Reproduce")
        st.caption("Run commands from the repository root.")

        st.write("Generate raw data:")
        st.code(plot_info["raw_command"], language="bash")

        st.write("Analyze raw CSVs into processed CSVs:")
        st.code(
            "python3 scripts/analyze_metric.py "
            "--baseline-csv results2/raw/baseline_gpu.csv "
            "--quant-csv results2/raw/quant_gpu.csv "
            "--paged-csv results2/raw/paged_gpu.csv",
            language="bash",
        )

        st.write("Build plot:")
        st.code(plot_info["plot_command"], language="bash")

    _show_data_status()
    _show_preview_tables()


if __name__ == "__main__":
    main()
