import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kv_profile_utils import get_model_longseqlen, load_wikitext_layers_tensors

TENSOR_ORDER = ["k_pre_rope", "k_post_rope", "values"]
TENSOR_TITLES = {
    "k_pre_rope": "Keys (pre-RoPE)",
    "k_post_rope": "Keys (post-RoPE)",
    "values": "Values",
}


def load_single_layer_utils():
    script_path = Path(__file__).with_name("analyze-llama-kv-blocks.py")
    spec = importlib.util.spec_from_file_location("single_layer_block_analysis", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SINGLE_LAYER = load_single_layer_utils()


def resolve_layer_indices(model_name, seqlen, maxseqlen, requested_layers):
    if requested_layers:
        return sorted(set(requested_layers))

    model = get_model_longseqlen(model_name, seqlen, maxseqlen)
    num_layers = len(model.model.layers)
    del model
    return list(range(num_layers))


def build_results_for_layers(layer_tensors, block_sizes, num_bits):
    results = {}
    for layer_idx, tensor_map in layer_tensors.items():
        results[layer_idx] = {
            tensor_name: SINGLE_LAYER.build_tensor_report(tensor, block_sizes, num_bits)
            for tensor_name, tensor in tensor_map.items()
        }
    return results


def summarize_across_layers(results, layer_indices, block_sizes):
    aggregate = {}
    for tensor_name in TENSOR_ORDER:
        tensor_summary = {}
        for granularity in ["per_tensor", "per_token", "per_channel"]:
            values = np.array([results[layer_idx][tensor_name][granularity]["rel_rmse"] for layer_idx in layer_indices])
            tensor_summary[granularity] = {
                "rel_rmse_mean": float(values.mean()),
                "rel_rmse_std": float(values.std()),
                "rel_rmse_min": float(values.min()),
                "rel_rmse_max": float(values.max()),
            }

        tensor_summary["per_block"] = {}
        previous_block = None
        monotonic_count = 0
        for layer_idx in layer_indices:
            layer_ok = True
            previous_value = None
            for block_size in block_sizes:
                value = results[layer_idx][tensor_name]["per_block"][str(block_size)]["summary"]["rel_rmse"]
                if previous_value is not None and value < previous_value:
                    layer_ok = False
                previous_value = value
            monotonic_count += int(layer_ok)

        for block_size in block_sizes:
            rel_rmse = np.array(
                [
                    results[layer_idx][tensor_name]["per_block"][str(block_size)]["summary"]["rel_rmse"]
                    for layer_idx in layer_indices
                ]
            )
            block_cv = np.array(
                [
                    results[layer_idx][tensor_name]["per_block"][str(block_size)]["summary"]["block_cv_mean"]
                    for layer_idx in layer_indices
                ]
            )
            tensor_summary["per_block"][str(block_size)] = {
                "rel_rmse_mean": float(rel_rmse.mean()),
                "rel_rmse_std": float(rel_rmse.std()),
                "rel_rmse_min": float(rel_rmse.min()),
                "rel_rmse_max": float(rel_rmse.max()),
                "block_cv_mean": float(block_cv.mean()),
                "block_cv_std": float(block_cv.std()),
            }

        tensor_summary["layer_monotonic_fraction"] = monotonic_count / max(len(layer_indices), 1)
        best_blocks = []
        for layer_idx in layer_indices:
            scores = {
                block_size: results[layer_idx][tensor_name]["per_block"][str(block_size)]["summary"]["rel_rmse"]
                for block_size in block_sizes
            }
            best_blocks.append(min(scores, key=scores.get))
        counts = {block_size: best_blocks.count(block_size) for block_size in block_sizes}
        tensor_summary["best_block_counts"] = counts
        aggregate[tensor_name] = tensor_summary

    return aggregate


def write_layerwise_csv(results, layer_indices, block_sizes, output_path):
    fieldnames = [
        "layer",
        "tensor",
        "granularity",
        "block_size",
        "rel_rmse",
        "rel_mae",
        "rmse",
        "mae",
        "block_cv_mean",
        "block_cv_p90",
        "block_range_mean",
        "block_range_p90",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer_idx in layer_indices:
            for tensor_name in TENSOR_ORDER:
                tensor_report = results[layer_idx][tensor_name]
                for granularity in ["per_tensor", "per_token", "per_channel"]:
                    metrics = tensor_report[granularity]
                    writer.writerow(
                        {
                            "layer": layer_idx,
                            "tensor": tensor_name,
                            "granularity": granularity,
                            "block_size": "",
                            "rel_rmse": metrics["rel_rmse"],
                            "rel_mae": metrics["rel_mae"],
                            "rmse": metrics["rmse"],
                            "mae": metrics["mae"],
                            "block_cv_mean": "",
                            "block_cv_p90": "",
                            "block_range_mean": "",
                            "block_range_p90": "",
                        }
                    )
                for block_size in block_sizes:
                    metrics = tensor_report["per_block"][str(block_size)]["summary"]
                    writer.writerow(
                        {
                            "layer": layer_idx,
                            "tensor": tensor_name,
                            "granularity": "per_block",
                            "block_size": block_size,
                            "rel_rmse": metrics["rel_rmse"],
                            "rel_mae": metrics["rel_mae"],
                            "rmse": metrics["rmse"],
                            "mae": metrics["mae"],
                            "block_cv_mean": metrics["block_cv_mean"],
                            "block_cv_p90": metrics["block_cv_p90"],
                            "block_range_mean": metrics["block_range_mean"],
                            "block_range_p90": metrics["block_range_p90"],
                        }
                    )


def write_aggregate_csv(aggregate, block_sizes, output_path):
    fieldnames = [
        "tensor",
        "granularity",
        "block_size",
        "rel_rmse_mean",
        "rel_rmse_std",
        "rel_rmse_min",
        "rel_rmse_max",
        "block_cv_mean",
        "block_cv_std",
        "layer_monotonic_fraction",
        "best_block_counts",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tensor_name in TENSOR_ORDER:
            tensor_summary = aggregate[tensor_name]
            for granularity in ["per_tensor", "per_token", "per_channel"]:
                metrics = tensor_summary[granularity]
                writer.writerow(
                    {
                        "tensor": tensor_name,
                        "granularity": granularity,
                        "block_size": "",
                        "rel_rmse_mean": metrics["rel_rmse_mean"],
                        "rel_rmse_std": metrics["rel_rmse_std"],
                        "rel_rmse_min": metrics["rel_rmse_min"],
                        "rel_rmse_max": metrics["rel_rmse_max"],
                        "block_cv_mean": "",
                        "block_cv_std": "",
                        "layer_monotonic_fraction": tensor_summary["layer_monotonic_fraction"],
                        "best_block_counts": json.dumps(tensor_summary["best_block_counts"]),
                    }
                )
            for block_size in block_sizes:
                metrics = tensor_summary["per_block"][str(block_size)]
                writer.writerow(
                    {
                        "tensor": tensor_name,
                        "granularity": "per_block",
                        "block_size": block_size,
                        "rel_rmse_mean": metrics["rel_rmse_mean"],
                        "rel_rmse_std": metrics["rel_rmse_std"],
                        "rel_rmse_min": metrics["rel_rmse_min"],
                        "rel_rmse_max": metrics["rel_rmse_max"],
                        "block_cv_mean": metrics["block_cv_mean"],
                        "block_cv_std": metrics["block_cv_std"],
                        "layer_monotonic_fraction": tensor_summary["layer_monotonic_fraction"],
                        "best_block_counts": json.dumps(tensor_summary["best_block_counts"]),
                    }
                )


def plot_layerwise_rel_rmse(results, layer_indices, block_sizes, output_path):
    fig, axes = plt.subplots(len(TENSOR_ORDER), 1, figsize=(14, 12), sharex=True)
    if len(TENSOR_ORDER) == 1:
        axes = [axes]

    for ax, tensor_name in zip(axes, TENSOR_ORDER):
        for block_size in block_sizes:
            values = [
                results[layer_idx][tensor_name]["per_block"][str(block_size)]["summary"]["rel_rmse"]
                for layer_idx in layer_indices
            ]
            ax.plot(layer_indices, values, marker="o", linewidth=1.8, label=f"block{block_size}")

        ax.plot(
            layer_indices,
            [results[layer_idx][tensor_name]["per_tensor"]["rel_rmse"] for layer_idx in layer_indices],
            linestyle="--",
            linewidth=1.5,
            label="per_tensor",
        )
        ax.plot(
            layer_indices,
            [results[layer_idx][tensor_name]["per_token"]["rel_rmse"] for layer_idx in layer_indices],
            linestyle="--",
            linewidth=1.5,
            label="per_token",
        )
        ax.plot(
            layer_indices,
            [results[layer_idx][tensor_name]["per_channel"]["rel_rmse"] for layer_idx in layer_indices],
            linestyle="--",
            linewidth=1.5,
            label="per_channel",
        )
        ax.set_title(f"{TENSOR_TITLES[tensor_name]} across layers")
        ax.set_ylabel("Relative RMSE")
        ax.grid(alpha=0.3)
        ax.legend(ncol=3, fontsize=9)

    axes[-1].set_xlabel("Layer index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_summary(aggregate, block_sizes, output_path):
    fig, axes = plt.subplots(len(TENSOR_ORDER), 2, figsize=(14, 12))
    for row, tensor_name in enumerate(TENSOR_ORDER):
        ax_err = axes[row, 0]
        labels = ["tensor"] + [f"block{b}" for b in block_sizes] + ["token", "channel"]
        values = [aggregate[tensor_name]["per_tensor"]["rel_rmse_mean"]]
        values += [aggregate[tensor_name]["per_block"][str(b)]["rel_rmse_mean"] for b in block_sizes]
        values += [
            aggregate[tensor_name]["per_token"]["rel_rmse_mean"],
            aggregate[tensor_name]["per_channel"]["rel_rmse_mean"],
        ]
        ax_err.plot(labels, values, marker="o", linewidth=2)
        ax_err.set_title(f"{TENSOR_TITLES[tensor_name]} mean rel_rmse")
        ax_err.set_ylabel("Mean relative RMSE")
        ax_err.grid(alpha=0.3)

        ax_cv = axes[row, 1]
        mean_cv = [aggregate[tensor_name]["per_block"][str(b)]["block_cv_mean"] for b in block_sizes]
        std_cv = [aggregate[tensor_name]["per_block"][str(b)]["block_cv_std"] for b in block_sizes]
        ax_cv.errorbar(block_sizes, mean_cv, yerr=std_cv, marker="o", linewidth=2, capsize=4)
        ax_cv.set_title(f"{TENSOR_TITLES[tensor_name]} block CV")
        ax_cv.set_xlabel("Block size")
        ax_cv.set_ylabel("Mean block CV")
        ax_cv.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_console_summary(aggregate, block_sizes):
    print("Cross-layer summary")
    for tensor_name in TENSOR_ORDER:
        summary = aggregate[tensor_name]
        print(f"\n[{tensor_name}]")
        print(f"  monotonic_fraction: {summary['layer_monotonic_fraction']:.3f}")
        print(f"  best_block_counts : {summary['best_block_counts']}")
        print(f"  per_tensor mean   : {summary['per_tensor']['rel_rmse_mean']:.6f}")
        for block_size in block_sizes:
            metrics = summary["per_block"][str(block_size)]
            print(
                f"  block{block_size:<4}: mean_rel_rmse={metrics['rel_rmse_mean']:.6f} "
                f"std_rel_rmse={metrics['rel_rmse_std']:.6f} "
                f"mean_block_cv={metrics['block_cv_mean']:.6f}"
            )
        print(f"  per_token mean    : {summary['per_token']['rel_rmse_mean']:.6f}")
        print(f"  per_channel mean  : {summary['per_channel']['rel_rmse_mean']:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze whether block quantization conclusions stay consistent across many LLaMA layers.",
    )
    parser.add_argument("model", type=str, help="Path or HF name for the LLaMA model.")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--maxseqlen", type=int, default=2048)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--layer-indices", type=int, nargs="*", default=None, help="Explicit layer indices to analyze.")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="block-analysis-multilayer")
    args = parser.parse_args()

    block_sizes = sorted(set(args.block_sizes))
    layer_indices = resolve_layer_indices(args.model, args.seqlen, args.maxseqlen, args.layer_indices)
    layer_tensors = load_wikitext_layers_tensors(
        model_name=args.model,
        seqlen=args.seqlen,
        maxseqlen=args.maxseqlen,
        sample_index=args.sample_index,
        layer_indices=layer_indices,
        device=args.device,
    )
    results = build_results_for_layers(layer_tensors, block_sizes, args.num_bits)
    aggregate = summarize_across_layers(results, layer_indices, block_sizes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "layer_indices": layer_indices,
        "seqlen": args.seqlen,
        "sample_index": args.sample_index,
        "num_bits": args.num_bits,
        "block_sizes": block_sizes,
        "aggregate": aggregate,
        "results": results,
    }

    SINGLE_LAYER.save_json_report(payload, output_dir / "multilayer_block_quant_report.json")
    write_layerwise_csv(results, layer_indices, block_sizes, output_dir / "multilayer_layerwise_summary.csv")
    write_aggregate_csv(aggregate, block_sizes, output_dir / "multilayer_aggregate_summary.csv")
    plot_layerwise_rel_rmse(results, layer_indices, block_sizes, output_dir / "multilayer_layerwise_rel_rmse.png")
    plot_aggregate_summary(aggregate, block_sizes, output_dir / "multilayer_aggregate_summary.png")

    print_console_summary(aggregate, block_sizes)
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
