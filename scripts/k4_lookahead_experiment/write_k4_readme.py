import argparse
import csv
import json
from pathlib import Path


BASELINES = {
    "nuq4-1%": {"baseline": 5.701, "val_rmse": 5.727, "attnout": 5.727, "strategy": "k4_fixed4_value_strategy"},
    "nuq3-1%": {"baseline": 5.760, "val_rmse": 5.952, "attnout": 5.972, "strategy": "k4_fixed3_value_strategy"},
    "nuq2-1%": {"baseline": 6.069, "val_rmse": 40.988, "attnout": 41.373, "strategy": "k4_fixed2_value_strategy"},
}


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--result-dir", default="results/k4-lookahead")
    parser.add_argument("--output", default="README_k4_lookahead_experiment.md")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    result_dir = repo_root / args.result_dir
    summary_rows = read_csv(result_dir / "k4_strategy_summary.csv")
    ppl_rows = read_csv(result_dir / "k4_full_ppl_summary.csv")
    ppl_by_name = {row["strategy"]: row for row in ppl_rows}

    lines = []
    lines.append("# K4 Lookahead Value Granularity Experiment")
    lines.append("")
    lines.append("This preliminary experiment evaluates Value quantization candidates with a K=4 hidden-state lookahead RMSE. For a target layer, only that layer's Value granularity is varied; the Key path and subsequent layers use the KVQuant baseline setting for the same bit-width.")
    lines.append("")
    lines.append("## Strategy Summary")
    lines.append("")
    lines.append("| Strategy | Avg Bit | #Per-token | #Per-channel | #Tile32 | Per-token Layers | Per-channel Layers | Tile32 Layers |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for row in summary_rows:
        lines.append(
            f"| {row['strategy']} | {row['avg_bit']} | {row['num_per_token']} | {row['num_per_channel']} | {row['num_tile32']} | `{row['per_token_layers']}` | `{row['per_channel_layers']}` | `{row['tile32_layers']}` |"
        )

    lines.append("")
    lines.append("## Full PPL")
    lines.append("")
    lines.append("| Config | Baseline PPL | val_rmse PPL | attnout PPL | K4 PPL | Delta vs Baseline |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for config, meta in BASELINES.items():
        row = ppl_by_name.get(meta["strategy"])
        if row:
            lines.append(
                f"| {config} | {meta['baseline']:.3f} | {meta['val_rmse']:.3f} | {meta['attnout']:.3f} | {float(row['full_ppl']):.6f} | {float(row['delta_ppl_vs_baseline']):+.6f} |"
            )
        else:
            lines.append(
                f"| {config} | {meta['baseline']:.3f} | {meta['val_rmse']:.3f} | {meta['attnout']:.3f} | TODO | TODO |"
            )

    lines.append("")
    lines.append("## Interim Answers")
    lines.append("")
    lines.append("1. The selected per-token/per-channel/tile32 counts are listed in the strategy table above.")
    lines.append("2. The K4 policies should be compared with the earlier val_rmse/attnout policies by the per-channel count and final PPL.")
    lines.append("3. For fixed2, success means avoiding the 40+ PPL failure mode seen in val_rmse and attnout.")
    lines.append("4. For fixed3, success means landing clearly below 5.952/5.972.")
    lines.append("5. For fixed4, success means staying close to the 5.701 baseline.")
    lines.append("6. If K4 fails, inspect `k4_lookahead_layer_candidate_rmse.csv` for layers where per-channel/tile32 was selected with a small margin.")
    lines.append("7. Next candidates are K=2, K=8, and adding cosine drift or logits-KL as an auxiliary signal.")
    lines.append("")

    (repo_root / args.output).write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {repo_root / args.output}")


if __name__ == "__main__":
    main()
