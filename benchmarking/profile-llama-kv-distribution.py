import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kv_profile_utils import load_wikitext_layer_tensors, reduce_blocks, resolve_results_root

RESULTS_ROOT = resolve_results_root(Path(__file__))


def plot_profiles(profiles, token_block, channel_block, reduction, output_path):
    fig = plt.figure(figsize=(18, 6))
    titles = [
        ("k_pre_rope", "Layer Keys (pre-RoPE)"),
        ("k_post_rope", "Layer Keys (post-RoPE)"),
        ("values", "Layer Values"),
    ]
    z_label = {
        "max_abs": "Max |value|",
        "mean_abs": "Mean |value|",
        "p99_abs": "P99 |value|",
    }[reduction]

    for idx, (key, title) in enumerate(titles, start=1):
        grid = profiles[key]["grid"]
        token_positions = np.arange(grid.shape[0]) * token_block
        channel_positions = np.arange(grid.shape[1]) * channel_block
        x_grid, y_grid = np.meshgrid(channel_positions, token_positions)

        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.plot_surface(
            x_grid,
            y_grid,
            grid.numpy(),
            cmap="coolwarm",
            linewidth=0,
            antialiased=True,
            alpha=0.95,
        )
        ax.set_title(title)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Token")
        ax.set_zlabel(z_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Profile LLaMA key/value activation magnitudes on Wikitext-2 at block granularity.",
    )
    parser.add_argument("model", type=str, help="Path or HF name for the LLaMA model.")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length for the Wikitext-2 sample.")
    parser.add_argument("--maxseqlen", type=int, default=2048, help="Context length used when loading the model.")
    parser.add_argument("--sample-index", type=int, default=0, help="Which consecutive 2K chunk to profile.")
    parser.add_argument("--layer-idx", type=int, default=10, help="Zero-based transformer layer index to profile.")
    parser.add_argument("--token-block", type=int, default=64, help="Token block size for aggregation.")
    parser.add_argument("--channel-block", type=int, default=64, help="Channel block size for aggregation.")
    parser.add_argument(
        "--reduction",
        type=str,
        default="max_abs",
        choices=["max_abs", "mean_abs", "p99_abs"],
        help="Statistic used to summarize each block.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device to run the model on.")
    parser.add_argument(
        "--output-data",
        type=str,
        default=str(RESULTS_ROOT / "kv-profile-layer10-64x64.pkl"),
        help="Output pickle for the aggregated profiling data.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(RESULTS_ROOT / "kv-profile-layer10-64x64.png"),
        help="Output PNG for the 3-panel profiling figure.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also store the raw [token, channel] tensors in the output pickle.",
    )
    args = parser.parse_args()

    captured = load_wikitext_layer_tensors(
        model_name=args.model,
        seqlen=args.seqlen,
        maxseqlen=args.maxseqlen,
        sample_index=args.sample_index,
        layer_idx=args.layer_idx,
        device=args.device,
    )

    profiles = {}
    for key, tensor in captured.items():
        grid, seq_trim, hidden_trim = reduce_blocks(
            tensor,
            token_block=args.token_block,
            channel_block=args.channel_block,
            reduction=args.reduction,
        )
        profiles[key] = {
            "grid": grid,
            "raw_shape": tuple(tensor.shape),
            "trimmed_shape": (seq_trim, hidden_trim),
        }
        if args.save_raw:
            profiles[key]["raw"] = tensor

    payload = {
        "model": args.model,
        "layer_idx": args.layer_idx,
        "sample_index": args.sample_index,
        "seqlen": args.seqlen,
        "token_block": args.token_block,
        "channel_block": args.channel_block,
        "reduction": args.reduction,
        "profiles": profiles,
    }

    output_data = Path(args.output_data)
    output_data.parent.mkdir(parents=True, exist_ok=True)
    with output_data.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    output_plot = Path(args.output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plot_profiles(profiles, args.token_block, args.channel_block, args.reduction, output_plot)

    print(f"Saved profiling data to {output_data}")
    print(f"Saved profiling figure to {output_plot}")


if __name__ == "__main__":
    main()
