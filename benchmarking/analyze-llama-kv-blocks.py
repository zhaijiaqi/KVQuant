import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kv_profile_utils import load_wikitext_layer_tensors, resolve_results_root

RESULTS_ROOT = resolve_results_root(Path(__file__))

EPS = 1e-8
TENSOR_ORDER = ["k_pre_rope", "k_post_rope", "values"]
TENSOR_TITLES = {
    "k_pre_rope": "Keys (pre-RoPE)",
    "k_post_rope": "Keys (post-RoPE)",
    "values": "Values",
}


def _error_metrics(reference, reconstructed):
    diff = (reconstructed - reference).float()
    ref = reference.float()
    mse = diff.pow(2).mean().item()
    rmse = mse ** 0.5
    mae = diff.abs().mean().item()
    ref_rms = ref.pow(2).mean().sqrt().item()
    ref_mean_abs = ref.abs().mean().item()
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rel_rmse": rmse / max(ref_rms, EPS),
        "rel_mae": mae / max(ref_mean_abs, EPS),
    }


def _symmetric_quantize(tensor, scale, num_bits):
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -qmax
    scale = torch.clamp(scale, min=EPS)
    quantized = torch.round(tensor / scale).clamp(qmin, qmax)
    return quantized * scale


def quantize_per_tensor(tensor, num_bits):
    scale = tensor.abs().amax() / ((1 << (num_bits - 1)) - 1)
    reconstructed = _symmetric_quantize(tensor, scale, num_bits)
    return _error_metrics(tensor, reconstructed)


def quantize_per_token(tensor, num_bits):
    scale = tensor.abs().amax(dim=1, keepdim=True) / ((1 << (num_bits - 1)) - 1)
    reconstructed = _symmetric_quantize(tensor, scale, num_bits)
    return _error_metrics(tensor, reconstructed)


def quantize_per_channel(tensor, num_bits):
    scale = tensor.abs().amax(dim=0, keepdim=True) / ((1 << (num_bits - 1)) - 1)
    reconstructed = _symmetric_quantize(tensor, scale, num_bits)
    return _error_metrics(tensor, reconstructed)


def quantize_per_block(tensor, block_size, num_bits):
    seq_len, hidden_size = tensor.shape
    seq_trim = (seq_len // block_size) * block_size
    hidden_trim = (hidden_size // block_size) * block_size
    trimmed = tensor[:seq_trim, :hidden_trim].contiguous()

    blocks = trimmed.view(
        seq_trim // block_size,
        block_size,
        hidden_trim // block_size,
        block_size,
    ).permute(0, 2, 1, 3).contiguous()

    abs_blocks = blocks.abs().float()
    scale = abs_blocks.amax(dim=(-1, -2), keepdim=True) / ((1 << (num_bits - 1)) - 1)
    reconstructed_blocks = _symmetric_quantize(blocks, scale, num_bits)

    mean_abs = abs_blocks.mean(dim=(-1, -2))
    std_abs = abs_blocks.std(dim=(-1, -2), unbiased=False)
    max_abs = abs_blocks.amax(dim=(-1, -2))
    p99_abs = torch.quantile(abs_blocks.reshape(*abs_blocks.shape[:2], -1), 0.99, dim=-1)

    block_rmse = (reconstructed_blocks.float() - blocks.float()).pow(2).mean(dim=(-1, -2)).sqrt()
    block_ref_rms = blocks.float().pow(2).mean(dim=(-1, -2)).sqrt()
    block_rel_rmse = block_rmse / torch.clamp(block_ref_rms, min=EPS)

    reconstructed = reconstructed_blocks.permute(0, 2, 1, 3).reshape(seq_trim, hidden_trim)
    overall = _error_metrics(trimmed, reconstructed)

    cv_abs = std_abs / torch.clamp(mean_abs, min=EPS)
    range_ratio = max_abs / torch.clamp(mean_abs, min=EPS)

    overall.update(
        {
            "block_size": block_size,
            "num_blocks": int((seq_trim // block_size) * (hidden_trim // block_size)),
            "trimmed_shape": [int(seq_trim), int(hidden_trim)],
            "block_cv_mean": float(cv_abs.mean().item()),
            "block_cv_p90": float(torch.quantile(cv_abs.reshape(-1), 0.90).item()),
            "block_cv_p99": float(torch.quantile(cv_abs.reshape(-1), 0.99).item()),
            "block_range_mean": float(range_ratio.mean().item()),
            "block_range_p90": float(torch.quantile(range_ratio.reshape(-1), 0.90).item()),
        }
    )

    heatmaps = {
        "mean_abs": mean_abs.cpu(),
        "cv_abs": cv_abs.cpu(),
        "max_abs": max_abs.cpu(),
        "p99_abs": p99_abs.cpu(),
        "rel_rmse": block_rel_rmse.cpu(),
    }
    return overall, heatmaps


def build_tensor_report(tensor, block_sizes, num_bits):
    report = {
        "per_tensor": quantize_per_tensor(tensor, num_bits),
        "per_token": quantize_per_token(tensor, num_bits),
        "per_channel": quantize_per_channel(tensor, num_bits),
        "per_block": {},
    }
    for block_size in block_sizes:
        summary, heatmaps = quantize_per_block(tensor, block_size, num_bits)
        report["per_block"][str(block_size)] = {
            "summary": summary,
            "heatmaps": heatmaps,
        }
    return report


def write_summary_csv(results, block_sizes, output_path):
    fieldnames = [
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
        for tensor_name in TENSOR_ORDER:
            tensor_report = results[tensor_name]
            for granularity in ["per_tensor", "per_token", "per_channel"]:
                metrics = tensor_report[granularity]
                writer.writerow(
                    {
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


def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    return obj


def save_json_report(payload, output_path):
    with output_path.open("w") as handle:
        json.dump(tensor_to_serializable(payload), handle, indent=2)


def plot_error_summary(results, block_sizes, output_path):
    fig, axes = plt.subplots(len(TENSOR_ORDER), 2, figsize=(16, 12))

    for row, tensor_name in enumerate(TENSOR_ORDER):
        tensor_report = results[tensor_name]
        ax_err = axes[row, 0]
        labels = ["tensor"] + [f"block{b}" for b in block_sizes] + ["token", "channel"]
        values = [tensor_report["per_tensor"]["rel_rmse"]]
        values += [tensor_report["per_block"][str(b)]["summary"]["rel_rmse"] for b in block_sizes]
        values += [
            tensor_report["per_token"]["rel_rmse"],
            tensor_report["per_channel"]["rel_rmse"],
        ]
        ax_err.plot(labels, values, marker="o", linewidth=2)
        ax_err.set_title(f"{TENSOR_TITLES[tensor_name]}: Relative RMSE")
        ax_err.set_ylabel("Relative RMSE")
        ax_err.grid(alpha=0.3)

        ax_cv = axes[row, 1]
        mean_cv = [tensor_report["per_block"][str(b)]["summary"]["block_cv_mean"] for b in block_sizes]
        p90_cv = [tensor_report["per_block"][str(b)]["summary"]["block_cv_p90"] for b in block_sizes]
        ax_cv.plot(block_sizes, mean_cv, marker="o", linewidth=2, label="mean block CV")
        ax_cv.plot(block_sizes, p90_cv, marker="s", linewidth=2, label="p90 block CV")
        ax_cv.set_title(f"{TENSOR_TITLES[tensor_name]}: Block Homogeneity")
        ax_cv.set_xlabel("Block size")
        ax_cv.set_ylabel("Std(|x|) / Mean(|x|)")
        ax_cv.grid(alpha=0.3)
        ax_cv.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(results, block_sizes, stat_name, output_path):
    fig, axes = plt.subplots(len(TENSOR_ORDER), len(block_sizes), figsize=(4 * len(block_sizes), 10))
    if len(TENSOR_ORDER) == 1:
        axes = np.array([axes])
    if len(block_sizes) == 1:
        axes = axes.reshape(len(TENSOR_ORDER), 1)

    for row, tensor_name in enumerate(TENSOR_ORDER):
        for col, block_size in enumerate(block_sizes):
            ax = axes[row, col]
            heatmap = results[tensor_name]["per_block"][str(block_size)]["heatmaps"][stat_name].numpy()
            image = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
            if row == 0:
                ax.set_title(f"block={block_size}")
            if col == 0:
                ax.set_ylabel(TENSOR_TITLES[tensor_name])
            ax.set_xlabel("Channel block")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_console_summary(results, block_sizes):
    print("Relative RMSE summary")
    for tensor_name in TENSOR_ORDER:
        tensor_report = results[tensor_name]
        print(f"\n[{tensor_name}]")
        print(f"  per_tensor  : {tensor_report['per_tensor']['rel_rmse']:.6f}")
        for block_size in block_sizes:
            summary = tensor_report["per_block"][str(block_size)]["summary"]
            print(
                f"  block{block_size:<4}: rel_rmse={summary['rel_rmse']:.6f} "
                f"block_cv_mean={summary['block_cv_mean']:.6f} "
                f"block_range_mean={summary['block_range_mean']:.6f}"
            )
        print(f"  per_token   : {tensor_report['per_token']['rel_rmse']:.6f}")
        print(f"  per_channel : {tensor_report['per_channel']['rel_rmse']:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze whether LLaMA K/V activations are suitable for per-block quantization.",
    )
    parser.add_argument("model", type=str, help="Path or HF name for the LLaMA model.")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length for the Wikitext-2 sample.")
    parser.add_argument("--maxseqlen", type=int, default=2048, help="Context length used when loading the model.")
    parser.add_argument("--sample-index", type=int, default=0, help="Which consecutive 2K chunk to profile.")
    parser.add_argument("--layer-idx", type=int, default=10, help="Zero-based transformer layer index to profile.")
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Block sizes to analyze.",
    )
    parser.add_argument("--num-bits", type=int, default=4, help="Quantization bit-width for error estimation.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device to run the model on.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_ROOT / "block-analysis-layer10"),
        help="Directory used to store JSON/CSV/PNG outputs.",
    )
    args = parser.parse_args()

    block_sizes = sorted(set(args.block_sizes))
    tensors = load_wikitext_layer_tensors(
        model_name=args.model,
        seqlen=args.seqlen,
        maxseqlen=args.maxseqlen,
        sample_index=args.sample_index,
        layer_idx=args.layer_idx,
        device=args.device,
    )

    results = {
        tensor_name: build_tensor_report(tensor, block_sizes, args.num_bits)
        for tensor_name, tensor in tensors.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "layer_idx": args.layer_idx,
        "sample_index": args.sample_index,
        "seqlen": args.seqlen,
        "num_bits": args.num_bits,
        "block_sizes": block_sizes,
        "results": results,
        "notes": {
            "interpretation": [
                "Lower rel_rmse means the shared quantization scale is losing less information.",
                "Lower block_cv_mean means values inside each block are more homogeneous.",
                "If per-block rel_rmse stays close to finer-grained baselines and block_cv remains moderate, block quantization is more plausible.",
            ]
        },
    }

    save_json_report(payload, output_dir / "block_quant_report.json")
    write_summary_csv(results, block_sizes, output_dir / "block_quant_summary.csv")
    plot_error_summary(results, block_sizes, output_dir / "block_quant_summary.png")
    plot_heatmaps(results, block_sizes, "max_abs", output_dir / "block_max_abs.png")
    plot_heatmaps(results, block_sizes, "rel_rmse", output_dir / "block_rel_rmse.png")

    print_console_summary(results, block_sizes)
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
