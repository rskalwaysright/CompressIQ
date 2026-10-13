"""Command-line entrypoint for CompressIQ benchmark experiments."""

import argparse
import sys
from pathlib import Path

from compressiq.configs.default_config import get_default_config, load_config, save_config
from compressiq.experiments.runner import run_benchmark_pipeline


def main():
    parser = argparse.ArgumentParser(description="CompressIQ: Research-Grade Neural Network Compression Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--budget", type=float, default=4.0, help="Target model size budget in MB (default: 4.0 MB)")
    parser.add_argument("--arch", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "resnet18"], help="Model architecture")
    parser.add_argument("--batches", type=int, default=15, help="Number of evaluation batches for fast calibration (default: 15)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")

    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)
    else:
        cfg = get_default_config()

    cfg.model.architecture = args.arch
    cfg.optimization.target_budget_mb = args.budget
    cfg.evaluation.device = args.device

    print("=" * 70)
    print("COMPRESSIQ: Neural Network Compression Benchmark")
    print(f"Architecture:      {cfg.model.architecture}")
    print(f"Target Budget:     {cfg.optimization.target_budget_mb:.2f} MB")
    print(f"Device:            {cfg.evaluation.device}")
    print(f"Eval Batches:      {args.batches}")
    print("=" * 70)

    try:
        results = run_benchmark_pipeline(
            config=cfg,
            target_budget_mb=args.budget,
            max_eval_batches=args.batches
        )
        print("\n" + "=" * 70)
        print("EXPERIMENT BENCHMARK SUMMARY TABLE:")
        print("=" * 70)
        cols = ["Method", "top1_acc", "accuracy_drop", "model_size_mb", "compression_ratio", "median_latency_ms", "budget_satisfied"]
        available_cols = [c for c in cols if c in results["results_df"].columns]
        print(results["results_df"][available_cols].to_string(index=False))
        print("=" * 70)
        print(f"Inference Validation: {results['validation_status']}")
        print(f"Artifacts saved to:   {cfg.output_dir}/")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
