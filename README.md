# CompressIQ: Research-Grade Neural Network Compression Platform

**CompressIQ** is a research-grade, end-to-end neural network compression and optimization platform designed to find efficient compression configurations under user-defined model-size budgets while preserving classification accuracy.

---

## 1. Research Hypothesis & Scientific Methodology

### Core Hypothesis
> *Sensitivity-guided, layer-wise compression can outperform uniform compression at comparable model sizes.*

CompressIQ **does not presume this hypothesis to be true**. It subjects the hypothesis to rigorous empirical evaluation against standard uniform baselines on **CIFAR-100**:
1. **FP32 Baseline:** Uncompressed full-precision model (`MobileNetV2` primary, `ResNet-18` validation).
2. **Uniform INT8 Quantization:** All eligible layers uniformly quantized to 8-bit precision (~4x theoretical compression).
3. **Uniform Structured Pruning:** Uniformly removes 30% of candidate intermediate channels across all blocks.
4. **Uniform Hybrid Compression:** Combines uniform 8-bit quantization with uniform 15% channel pruning.
5. **Native INT8 PTQ:** Genuine dynamic integer post-training quantization executed via PyTorch runtime on CPU.
6. **Sensitivity-Guided (Proposed):** Deterministic greedy optimizer navigating per-layer sensitivity profiles to assign mixed-precision bit-widths and structured pruning ratios under a user-defined size budget (MB).

### Genuine Structured Pruning vs. Zero-Masking
Unlike naive pruning approaches that merely multiply weights by a binary mask (retaining full memory footprint and FLOP count), CompressIQ performs **true physical channel excision and architecture reconstruction**:
- Computes filter importance using the $L_1$-norm: $\sum |W_{c, :, :, :}|$.
- Physically instantiates smaller `nn.Conv2d` layers with fewer output channels.
- Updates matching `nn.BatchNorm2d` parameters (`weight`, `bias`, `running_mean`, `running_var`).
- Resizes downstream input channel dimensions to ensure graph consistency.
- Trims `InvertedResidual` intermediate expansion dimensions in MobileNetV2 while preserving residual addition branches.

### Honest Speedup & Latency Reporting
- **Simulated Low-Bit Quantization (2-bit, 4-bit, 8-bit):** Quantizes floating-point weights to discrete levels and models accurate bit-width storage footprints. Standard PyTorch floating-point matrix multipliers execute the forward pass without claiming sub-byte speedups without dedicated hardware kernels.
- **Native INT8 PTQ:** Uses real PyTorch integer execution routines on CPU to measure genuine hardware execution latencies.

---

## 2. Project Architecture

```
project-root/
│
├── compressiq/
│   ├── configs/
│   │   ├── schema.py              # Strongly-typed configuration dataclasses
│   │   └── default_config.py      # Default configuration presets & serializers
│   ├── data/
│   │   └── dataset.py             # Official CIFAR-100 loading & deterministic train/val split
│   ├── models/
│   │   ├── mobilenet.py           # MobileNetV2 adapted for CIFAR-100 resolution
│   │   ├── resnet.py              # ResNet-18 validation architecture
│   │   └── factory.py             # Model instantiation, layer discovery & checkpointing
│   ├── compression/
│   │   ├── quantization.py        # Simulated low-bit quantization & Native INT8 PTQ
│   │   └── pruning.py             # Physical structured pruning & module reconstruction
│   ├── optimization/
│   │   ├── sensitivity.py         # Per-layer sensitivity profiling & metric scoring
│   │   └── budget_optimizer.py    # Deterministic greedy budget-aware optimizer
│   ├── evaluation/
│   │   ├── metrics.py             # Top-1/Top-5 acc, MACs, parameter count, latency, size
│   │   └── plots.py               # Research-grade publication figures
│   └── experiments/
│       └── runner.py              # End-to-end benchmark pipeline orchestration
│
├── tests/
│   ├── test_quantization.py       # Unit tests for uniform/simulated/native quantization
│   ├── test_pruning.py            # Unit tests for physical structured channel pruning
│   ├── test_optimizer.py          # Unit tests for greedy budget solver & infeasible budgets
│   └── test_metrics.py            # Unit tests for parameter count, size, MACs, latency
│
├── outputs/
│   ├── checkpoints/               # Exported model weights (.pt)
│   ├── plots/                     # Generated publication figures (.png)
│   └── results/                   # JSON & CSV benchmark metrics and sensitivity profiles
│
├── app.py                         # Interactive Streamlit research dashboard
├── main.py                        # Command-line benchmark execution entrypoint
├── requirements.txt               # Pinned core dependencies
├── pytest.ini                     # Pytest workspace configuration
└── README.md                      # Documentation & reproducibility guide
```

---

## 3. Quickstart & Installation

### Step 1: Set Up Python Environment
Ensure Python 3.10+ is installed:
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Core Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Running the Benchmark via CLI

Run the full benchmark pipeline on CIFAR-100:
```bash
python main.py --arch mobilenet_v2 --budget 4.0 --batches 15 --device auto
```

### Command-line Arguments:
- `--arch`: Model architecture (`mobilenet_v2` or `resnet18`).
- `--budget`: Target model size budget in MB (default: `4.0` MB, ~3.5x compression from ~14 MB baseline).
- `--batches`: Number of evaluation batches for calibration (default: `15`).
- `--device`: Target inference device (`auto`, `cuda`, `cpu`).
- `--config`: (Optional) Path to a customized JSON configuration file.

---

## 5. Launching the Interactive Streamlit Dashboard

Launch the research dashboard:
```bash
streamlit run app.py
```

The interactive dashboard provides:
1. **Experiment Dashboard:** Live metric cards, baseline comparison table, and budget feasibility indicators.
2. **Research Plots & Baselines:**
   - Accuracy vs. Model Size (MB) with budget lines.
   - Accuracy vs. Latency (ms).
   - Compression Ratio vs. Accuracy Drop.
   - Ablation Comparison summary.
3. **Layer Sensitivity Profile:** Bar charts highlighting fragile vs. compression-tolerant layers.
4. **Optimizer Decision Trace:** Step-by-step audit of greedy selections, marginal MB saved, and sensitivity scores.
5. **Model Export & Reproducibility:** Downloadable checkpoints (`baseline_fp32.pt`, `sensitivity_guided_opt.pt`) and CLI commands.
6. **Methodology & Hypothesis:** Full mathematical and research documentation.

---

## 6. Running Unit Tests

Run the complete test suite:
```bash
python -m pytest tests/ -v
```
All 11 unit tests validate quantization numerical fidelity, physical filter pruning, greedy budget compliance, and metric tracking.

---

## 7. Known Limitations

1. **Hardware Acceleration on Sub-Byte Precisions:** Simulated 2-bit and 4-bit quantization model memory footprints and numerical degradation accurately, but require custom CUDA/CUTLASS kernels for real inference acceleration beyond native INT8.
2. **Fine-Tuning:** The greedy optimizer evaluates post-training sensitivity without retraining. Accuracy retention under extreme pruning (>50%) can be further improved with iterative pruning-aware fine-tuning.
