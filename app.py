"""CompressIQ: Research-Grade Neural Network Compression and Optimization Platform.

Interactive Streamlit research dashboard connecting directly to the compression pipeline.
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
import torch
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="CompressIQ | Neural Network Compression Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current workspace to path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from compressiq.configs.schema import CompressIQConfig, DataConfig, ModelConfig, OptimizationConfig, EvaluationConfig
from compressiq.configs.default_config import get_default_config, save_config
from compressiq.experiments.runner import run_benchmark_pipeline
from compressiq.optimization.sensitivity import SensitivityProfile
from compressiq.optimization.budget_optimizer import OptimizationResult
from compressiq.evaluation.plots import (
    plot_accuracy_vs_size,
    plot_accuracy_vs_latency,
    plot_compression_vs_acc_drop,
    plot_layer_sensitivity_bars,
    plot_ablation_summary
)


# Custom CSS for clean research-lab aesthetics
st.markdown("""
<style>
    /* Research lab styling */
    .main { background-color: #f8fafc; }
    h1, h2, h3 { font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; color: #0f172a; }
    
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-delta-pos {
        font-size: 0.85rem;
        font-weight: 600;
        color: #059669;
    }
    .metric-delta-neg {
        font-size: 0.85rem;
        font-weight: 600;
        color: #dc2626;
    }
    .badge-feasible {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-infeasible {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def get_hardware_info():
    """Detect available hardware runtime."""
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Runtime"
    mem_info = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM" if cuda_avail else "Host RAM"
    return {
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "memory": mem_info,
        "torch_version": torch.__version__
    }


def load_cached_results():
    """Load latest results from outputs directory if available."""
    res_path = WORKSPACE_ROOT / "outputs" / "results" / "benchmark_summary.csv"
    json_path = WORKSPACE_ROOT / "outputs" / "results" / "benchmark_summary.json"
    sens_path = WORKSPACE_ROOT / "outputs" / "results" / "sensitivity_profile.json"
    opt_path = WORKSPACE_ROOT / "outputs" / "results" / "optimization_result.json"

    data = {}
    if res_path.exists():
        data["df"] = pd.read_csv(res_path)
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data["summary"] = json.load(f)
    if sens_path.exists():
        with open(sens_path, "r", encoding="utf-8") as f:
            data["profile"] = json.load(f)
    if opt_path.exists():
        with open(opt_path, "r", encoding="utf-8") as f:
            data["optimization"] = json.load(f)
    return data


# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🔬 CompressIQ")
st.sidebar.markdown("**Research Configuration & Controls**")

hw = get_hardware_info()
st.sidebar.info(
    f"**Device:** {hw['device_name']}\n\n"
    f"**Memory:** {hw['memory']}\n\n"
    f"**PyTorch:** v{hw['torch_version']}"
)

st.sidebar.markdown("### Model & Dataset")
arch = st.sidebar.selectbox("Architecture", ["mobilenet_v2", "resnet18"], index=0)
dataset = st.sidebar.selectbox("Dataset", ["CIFAR-100 (Official Split)"], index=0)

st.sidebar.markdown("### Optimization Budget")
target_budget = st.sidebar.slider(
    "Target Model Size Budget (MB)",
    min_value=1.0,
    max_value=12.0,
    value=4.0,
    step=0.25,
    help="Target disk/serialized size in megabytes. Baseline MobileNetV2 is ~14 MB."
)

st.sidebar.markdown("### Execution Settings")
calibration_batches = st.sidebar.slider(
    "Evaluation Batches",
    min_value=5,
    max_value=40,
    value=15,
    step=5,
    help="Batch limit used for sensitivity calibration and metric evaluation."
)

run_button = st.sidebar.button("🚀 Run Compression Pipeline", type="primary", use_container_width=True)

# Main Title & Subtitle
st.title("CompressIQ: Neural Network Compression Platform")
st.markdown(
    "A research-grade platform evaluating **sensitivity-guided layer-wise compression** "
    "(mixed-precision simulated quantization and structured channel pruning) "
    "against uniform compression baselines on **CIFAR-100**."
)

# Tabs
tab_exec, tab_results, tab_sensitivity, tab_trace, tab_export, tab_methodology = st.tabs([
    "📊 Experiment Dashboard",
    "📈 Research Plots & Baselines",
    "🎯 Layer Sensitivity Profile",
    "🔍 Optimizer Decision Trace",
    "💾 Model Export & Reproducibility",
    "📖 Methodology & Hypothesis"
])

# Handle execution
if run_button:
    cfg = get_default_config()
    cfg.model.architecture = arch
    cfg.optimization.target_budget_mb = target_budget
    cfg.evaluation.device = "auto"

    st.sidebar.success("Pipeline running...")
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(msg, frac):
        status_text.markdown(f"**Current Phase:** `{msg}`")
        progress_bar.progress(frac)

    try:
        with st.spinner("Executing compression experiments on real model weights..."):
            res = run_benchmark_pipeline(
                config=cfg,
                target_budget_mb=target_budget,
                max_eval_batches=calibration_batches,
                progress_callback=update_progress
            )
        status_text.markdown("✅ **Benchmark Pipeline Completed Successfully!**")
        st.session_state["cached_results"] = load_cached_results()
        st.rerun()
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Load existing results
results_data = st.session_state.get("cached_results", load_cached_results())


# ==========================================
# TAB 1: EXPERIMENT DASHBOARD
# ==========================================
with tab_exec:
    if "df" not in results_data or results_data["df"].empty:
        st.info("No benchmark run detected in `outputs/results/`. Click **Run Compression Pipeline** in the sidebar to execute the real benchmark on CIFAR-100.")
    else:
        df = results_data["df"]
        opt_data = results_data.get("optimization", {})

        # Extract baseline vs proposed
        fp32_row = df[df["Method"] == "FP32 Baseline"]
        prop_row = df[df["Method"] == "Sensitivity-Guided (Proposed)"]

        if not fp32_row.empty and not prop_row.empty:
            fp32 = fp32_row.iloc[0]
            prop = prop_row.iloc[0]

            st.markdown("### Baseline vs Sensitivity-Guided (Proposed) at a Glance")
            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Top-1 Accuracy</div>', unsafe_allow_html=True)
                acc_val = prop.get("top1_acc", 0.0)
                acc_base = fp32.get("top1_acc", 0.0)
                diff = acc_val - acc_base
                st.markdown(f'<div class="metric-value">{acc_val:.2f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="{ "metric-delta-pos" if diff >= 0 else "metric-delta-neg" }">{diff:+.2f}% vs FP32</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Model Size</div>', unsafe_allow_html=True)
                size_val = prop.get("model_size_mb", 0.0)
                size_base = fp32.get("model_size_mb", 0.0)
                st.markdown(f'<div class="metric-value">{size_val:.2f} MB</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-delta-pos">Budget: ≤ {target_budget:.1f} MB</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Compression Ratio</div>', unsafe_allow_html=True)
                ratio_val = prop.get("compression_ratio", 1.0)
                st.markdown(f'<div class="metric-value">{ratio_val:.1f}x</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-delta-pos">Footprint: {100/max(ratio_val, 1e-3):.1f}% of orig</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">MACs / FLOPs</div>', unsafe_allow_html=True)
                mac_val = prop.get("macs", 0) / 1e6
                mac_base = fp32.get("macs", 1) / 1e6
                st.markdown(f'<div class="metric-value">{mac_val:.1f} M</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-delta-pos">{-(1.0 - mac_val/max(mac_base, 1e-3))*100:.1f}% reduction</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c5:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Median Latency</div>', unsafe_allow_html=True)
                lat_val = prop.get("median_latency_ms", 0.0)
                st.markdown(f'<div class="metric-value">{lat_val:.2f} ms</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-delta-pos">p95: {prop.get("p95_latency_ms", 0.0):.2f} ms</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Comprehensive Baseline Comparison Table")
        # Format table cleanly
        display_df = df.copy()
        
        # Add visual tag for budget satisfied
        if "budget_satisfied" in display_df.columns:
            display_df["Budget Met"] = display_df["budget_satisfied"].apply(
                lambda x: "✅ Satisfied" if x else "❌ Exceeded"
            )

        cols_order = [
            "Method", "Type", "top1_acc", "accuracy_drop", "model_size_mb",
            "compression_ratio", "param_count", "macs", "median_latency_ms", "Budget Met"
        ]
        available_cols = [c for c in cols_order if c in display_df.columns]
        
        formatted_df = display_df[available_cols].rename(columns={
            "top1_acc": "Top-1 Acc (%)",
            "accuracy_drop": "Acc Drop (%)",
            "model_size_mb": "Size (MB)",
            "compression_ratio": "Ratio",
            "param_count": "Parameters",
            "macs": "MACs",
            "median_latency_ms": "Latency (ms)"
        })

        st.dataframe(
            formatted_df.style.highlight_max(subset=["Top-1 Acc (%)"], color="#dcfce7")
                              .highlight_min(subset=["Size (MB)", "Acc Drop (%)"], color="#dcfce7"),
            use_container_width=True
        )

        # Budget Feasibility status
        if opt_data:
            st.markdown("### Budget Constraint Analysis")
            is_feas = opt_data.get("is_feasible", True)
            if is_feas:
                st.success(
                    f"**Budget Constraint Feasible:** Target budget of **{opt_data.get('target_budget_mb'):.2f} MB** "
                    f"was successfully met with a final model size of **{opt_data.get('final_size_mb'):.2f} MB**."
                )
            else:
                st.warning(f"**Infeasible Budget Warning:** {opt_data.get('infeasibility_reason')}")


# ==========================================
# TAB 2: RESEARCH PLOTS & BASELINES
# ==========================================
with tab_results:
    if "df" not in results_data or results_data["df"].empty:
        st.info("Run the benchmark to generate research plots.")
    else:
        df = results_data["df"]
        st.markdown("### Empirical Trade-off Visualizations")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plot_accuracy_vs_size(df, target_budget_mb=target_budget)
            st.pyplot(fig1)

        with col2:
            fig2 = plot_accuracy_vs_latency(df)
            st.pyplot(fig2)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = plot_compression_vs_acc_drop(df)
            st.pyplot(fig3)

        with col4:
            fig4 = plot_ablation_summary(df)
            st.pyplot(fig4)


# ==========================================
# TAB 3: LAYER SENSITIVITY PROFILE
# ==========================================
with tab_sensitivity:
    profile_data = results_data.get("profile", {})
    if not profile_data:
        st.info("Run the benchmark to view layer-wise sensitivity profiles.")
    else:
        st.markdown("### Layer-Wise Sensitivity & Fragility Analysis")
        st.markdown(
            "Sensitivity Score is defined as $\\text{Score} = \\frac{\\Delta \\text{Val Loss}}{\\Delta \\text{MB Saved}}$. "
            "Layers with **lower scores** (green) can be heavily compressed with minimal accuracy loss, "
            "while layers with **higher scores** (red) are structurally fragile."
        )

        top_k = st.slider("Number of layers to display", min_value=5, max_value=30, value=15, step=5)
        fig_sens = plot_layer_sensitivity_bars(profile_data, top_k=top_k)
        st.pyplot(fig_sens)

        # Interactive Table of all records
        records = profile_data.get("records", [])
        if records:
            rec_df = pd.DataFrame(records)
            st.markdown("#### Full Per-Layer Sensitivity Records")
            st.dataframe(
                rec_df.sort_values(by="sensitivity_score", ascending=True),
                use_container_width=True
            )


# ==========================================
# TAB 4: OPTIMIZER DECISION TRACE
# ==========================================
with tab_trace:
    opt_data = results_data.get("optimization", {})
    trace = opt_data.get("decision_trace", []) if opt_data else []

    if not trace:
        st.info("No decision trace available. Run the optimization pipeline to see step-by-step greedy selections.")
    else:
        st.markdown("### Deterministic Greedy Search Decision Trace")
        st.markdown(
            "CompressIQ's budget-aware optimizer greedily evaluates each layer's sensitivity and selects "
            "the optimal compression configuration step-by-step until the model fits strictly within the budget."
        )

        trace_df = pd.DataFrame(trace)
        trace_df = trace_df.rename(columns={
            "step": "Step",
            "layer_name": "Layer Name",
            "compression_type": "Action Type",
            "selected_setting": "Selected Precision / Prune",
            "marginal_mb_saved": "Marginal MB Saved",
            "marginal_loss_penalty": "Marginal Loss Δ",
            "sensitivity_score": "Sensitivity Score",
            "cumulative_size_mb": "Cumulative Model Size (MB)"
        })

        st.dataframe(trace_df, use_container_width=True)


# ==========================================
# TAB 5: MODEL EXPORT & REPRODUCIBILITY
# ==========================================
with tab_export:
    st.markdown("### Artifact Exports & Checkpoint Verification")

    ckpt_base = WORKSPACE_ROOT / "outputs" / "checkpoints" / "baseline_fp32.pt"
    ckpt_opt = WORKSPACE_ROOT / "outputs" / "checkpoints" / "sensitivity_guided_opt.pt"

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Baseline FP32 Checkpoint")
        if ckpt_base.exists():
            st.success(f"File exists: `{ckpt_base.name}` ({ckpt_base.stat().st_size / (1024*1024):.2f} MB)")
            with open(ckpt_base, "rb") as f:
                st.download_button("📥 Download Baseline Checkpoint", f.read(), file_name="baseline_fp32.pt")
        else:
            st.info("Run benchmark to generate baseline checkpoint.")

    with col_b:
        st.markdown("#### Optimized Sensitivity-Guided Checkpoint")
        if ckpt_opt.exists():
            st.success(f"File exists: `{ckpt_opt.name}` ({ckpt_opt.stat().st_size / (1024*1024):.2f} MB)")
            with open(ckpt_opt, "rb") as f:
                st.download_button("📥 Download Compressed Model", f.read(), file_name="sensitivity_guided_opt.pt")
        else:
            st.info("Run benchmark to generate compressed checkpoint.")

    st.markdown("---")
    st.markdown("### Command-Line Reproducibility")
    st.markdown("To reproduce these benchmark results identically via the CLI:")
    st.code(
        f"python main.py --arch {arch} --budget {target_budget} --batches {calibration_batches} --device auto",
        language="bash"
    )


# ==========================================
# TAB 6: METHODOLOGY & HYPOTHESIS
# ==========================================
with tab_methodology:
    st.markdown("### Research Methodology & Scientific Rigor")
    st.markdown("""
    #### 1. Core Research Hypothesis
    > **Hypothesis:** *Sensitivity-guided, layer-wise compression can outperform uniform compression at comparable model sizes.*
    
    CompressIQ **does not assume this hypothesis to be true**. It subjects the hypothesis to rigorous empirical comparison against four standard baselines:
    - **Uniform INT8 Quantization:** All eligible layers quantised uniformly to 8-bit precision.
    - **Uniform Structured Pruning:** Uniformly prunes 30% of candidate intermediate channels.
    - **Uniform Hybrid Compression:** Uniform 8-bit precision combined with uniform 15% channel pruning.
    - **Native INT8 PTQ:** Genuine dynamic quantization executed via PyTorch runtime on CPU.

    #### 2. Physical Structured Pruning vs Zero-Masking
    CompressIQ implements **genuine physical structured pruning**:
    - Pruned output channels and corresponding BatchNorm running stats are physically sliced and excised.
    - Downstream input weights are resized.
    - InvertedResidual intermediate expansion dimensions in MobileNetV2 are physically trimmed, preserving skip-connections intact.
    - Parameter counts, MAC operations, and serialized footprint reflect actual structural reductions.

    #### 3. Honest Latency & Precision Reporting
    - **Simulated Quantization (2-bit, 4-bit, 8-bit):** Maps continuous weights to discrete quantized numerical levels. Footprint is calculated via bit-width serialization. Floating-point matrix multipliers are maintained for execution compatibility without claiming illusory speedups without custom hardware.
    - **Native INT8 PTQ:** Uses native PyTorch integer arithmetic kernels on CPU to provide realistic empirical runtime latency.
    """)
