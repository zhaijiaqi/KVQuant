"""
analyze-llama-kv-attnout-layer10.py

Repeat the block-analysis for a single LLaMA layer but replace the "Value
tensor rel_rmse" with "attention-output rel_rmse":

    attn_out_fp16  = softmax( Q · K_post^T / sqrt(d_k) ) · V_fp16
    attn_out_quant = softmax( Q · K_post^T / sqrt(d_k) ) · V_quant
    rel_rmse_attnout = ||attn_out_quant - attn_out_fp16|| / ||attn_out_fp16||

Key granularity is fixed to per-channel (matches the KVQuant baseline).
Value granularity is swept: per_tensor / per_token / per_channel / tile32.

This lets us see whether the per_token advantage that shows up in PPL
(but not in naive Value-tensor rel_rmse) also shows up under the
attention-output metric.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from kv_profile_utils import (
    get_attention_layout,
    get_model_longseqlen,
    build_sample,
    get_rope_cos_sin,
    apply_rope,
    project_qkv,
    flatten_heads,
    resolve_results_root,
)
from kvquant.datautils import get_loaders
from kvquant.model_parse import get_layers, parse_model

RESULTS_ROOT = resolve_results_root(Path(__file__))
EPS = 1e-8


# ──────────────────────────────────────────────────────────────
# Quantization helpers (same symmetric int-n as single-layer script)
# ──────────────────────────────────────────────────────────────

def _sym_quant(tensor, scale, num_bits):
    qmax = (1 << (num_bits - 1)) - 1
    scale = torch.clamp(scale, min=EPS)
    return torch.round(tensor / scale).clamp(-qmax, qmax) * scale


def quant_per_tensor(v, num_bits):
    scale = v.abs().amax() / ((1 << (num_bits - 1)) - 1)
    return _sym_quant(v, scale, num_bits)


def quant_per_token(v, num_bits):
    # v: [seq, hidden]  -> scale per row
    scale = v.abs().amax(dim=1, keepdim=True) / ((1 << (num_bits - 1)) - 1)
    return _sym_quant(v, scale, num_bits)


def quant_per_channel(v, num_bits):
    # v: [seq, hidden]  -> scale per column
    scale = v.abs().amax(dim=0, keepdim=True) / ((1 << (num_bits - 1)) - 1)
    return _sym_quant(v, scale, num_bits)


def quant_tile(v, tile_size, num_bits):
    """Tile quantization: each (tile_size x tile_size) block gets its own scale."""
    seq, hid = v.shape
    seq_t = (seq // tile_size) * tile_size
    hid_t = (hid // tile_size) * tile_size
    v_t = v[:seq_t, :hid_t]
    blocks = v_t.view(seq_t // tile_size, tile_size,
                      hid_t // tile_size, tile_size).permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(-1, -2), keepdim=True) / ((1 << (num_bits - 1)) - 1)
    q_blocks = _sym_quant(blocks, scale, num_bits)
    reconstructed = q_blocks.permute(0, 2, 1, 3).reshape(seq_t, hid_t)
    # Paste back (trimmed area stays fp16 = effectively zero error there)
    out = v.clone()
    out[:seq_t, :hid_t] = reconstructed
    return out


# ──────────────────────────────────────────────────────────────
# Capture: Q (post-RoPE, multi-head) + K_post (post-RoPE) + V
# ──────────────────────────────────────────────────────────────

def capture_qkv_full(model, input_ids, layer_idx, dev):
    """
    Returns a dict with:
      q_heads : [batch, num_heads, seq, head_dim]   float32 cpu
      k_post  : [batch, num_heads, seq, head_dim]   float32 cpu
      v_heads : [batch, num_kv_heads, seq, head_dim] float32 cpu
      v_flat  : [seq, num_kv_heads * head_dim]       float32 cpu
    """
    model_type = parse_model(model)
    layers = get_layers(model, model_type)
    attn_module = layers[layer_idx].self_attn
    captured = {}
    original_forward = attn_module.forward

    def wrapped_forward(*args, **kwargs):
        hidden_states = args[0] if args else kwargs["hidden_states"]
        position_ids = kwargs.get("position_ids")
        position_embeddings = kwargs.get("position_embeddings")

        bsz, q_len, _ = hidden_states.shape
        num_heads, num_kv_heads, head_dim = get_attention_layout(attn_module)

        q_raw, k_raw, v_raw = project_qkv(attn_module, hidden_states)

        q = q_raw.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        k = k_raw.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v_raw.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = get_rope_cos_sin(attn_module, v, position_ids, kv_seq_len)

        q_post, k_post = apply_rope(q, k, cos, sin, position_ids)

        if not captured:
            captured["q_heads"]  = q_post.detach().float().cpu()
            captured["k_post"]   = k_post.detach().float().cpu()
            captured["v_heads"]  = v.detach().float().cpu()
            captured["v_flat"]   = flatten_heads(v)[0].detach().float().cpu()

        return original_forward(*args, **kwargs)

    attn_module.forward = wrapped_forward
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        with torch.no_grad():
            model(input_ids.to(dev))
    finally:
        attn_module.forward = original_forward
        model.config.use_cache = use_cache

    return captured


# ──────────────────────────────────────────────────────────────
# Attention-output RMSE
# ──────────────────────────────────────────────────────────────

def compute_attn_out(q_heads, k_post, v_heads_flat, num_heads, num_kv_heads, head_dim, device):
    """
    Compute attention output given flat value tensor [seq, num_kv_heads*head_dim].
    Returns attn_out: [seq, num_heads * head_dim]
    """
    bsz, nh, seq, hd = q_heads.shape
    q = q_heads.to(device)
    k = k_post.to(device)

    # Reshape v from flat [seq, nkvh * hd] -> [1, nkvh, seq, hd]
    v = v_heads_flat.to(device).view(seq, num_kv_heads, head_dim).permute(1, 0, 2).unsqueeze(0)

    # GQA repeat if needed
    if num_kv_heads < num_heads:
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    scale = 1.0 / math.sqrt(hd)
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scale  # [1, nh, seq, seq]
    causal_mask = torch.ones((seq, seq), dtype=torch.bool, device=device).triu(1)
    attn_weights = attn_weights.masked_fill(
        causal_mask,
        torch.finfo(attn_weights.dtype).min,
    )
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_weights, v)                      # [1, nh, seq, hd]
    # Flatten: [seq, nh * hd]
    return attn_out[0].permute(1, 0, 2).reshape(seq, num_heads * head_dim).cpu()


def rel_rmse_attnout(attn_out_fp16, attn_out_quant):
    diff = (attn_out_quant - attn_out_fp16).float()
    ref  = attn_out_fp16.float()
    rmse = diff.pow(2).mean().sqrt().item()
    ref_rms = ref.pow(2).mean().sqrt().item()
    return rmse / max(ref_rms, EPS)


# ──────────────────────────────────────────────────────────────
# Build report for one layer
# ──────────────────────────────────────────────────────────────

def build_attnout_report(q_heads, k_post, v_flat, num_heads, num_kv_heads, head_dim,
                         num_bits, tile_sizes, device):
    """
    Sweep Value quantization granularity and report attn-output rel_rmse.
    Also reports Value-tensor rel_rmse (for comparison).
    """
    # fp16 reference attn output
    attn_fp16 = compute_attn_out(q_heads, k_post, v_flat, num_heads, num_kv_heads, head_dim, device)

    EPS_loc = EPS

    def val_rel_rmse(v_q, v_ref):
        diff = (v_q - v_ref).float()
        rmse = diff.pow(2).mean().sqrt().item()
        rms  = v_ref.float().pow(2).mean().sqrt().item()
        return rmse / max(rms, EPS_loc)

    results = {}

    for gran_name, v_quant in [
        ("per_tensor",  quant_per_tensor(v_flat, num_bits)),
        ("per_token",   quant_per_token(v_flat, num_bits)),
        ("per_channel", quant_per_channel(v_flat, num_bits)),
    ]:
        attn_q = compute_attn_out(q_heads, k_post, v_quant, num_heads, num_kv_heads, head_dim, device)
        results[gran_name] = {
            "attnout_rel_rmse": rel_rmse_attnout(attn_fp16, attn_q),
            "value_rel_rmse":   val_rel_rmse(v_quant, v_flat),
        }

    for tile_size in tile_sizes:
        v_quant = quant_tile(v_flat, tile_size, num_bits)
        attn_q  = compute_attn_out(q_heads, k_post, v_quant, num_heads, num_kv_heads, head_dim, device)
        results[f"tile{tile_size}"] = {
            "attnout_rel_rmse": rel_rmse_attnout(attn_fp16, attn_q),
            "value_rel_rmse":   val_rel_rmse(v_quant, v_flat),
        }

    return results


# ──────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────

GRAN_ORDER = ["per_tensor", "per_token", "per_channel", "tile32", "tile64"]

def write_csv(results_by_layer, output_path):
    fieldnames = ["layer", "granularity", "attnout_rel_rmse", "value_rel_rmse"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer_idx, res in sorted(results_by_layer.items()):
            for gran, metrics in res.items():
                writer.writerow({
                    "layer": layer_idx,
                    "granularity": gran,
                    "attnout_rel_rmse": metrics["attnout_rel_rmse"],
                    "value_rel_rmse": metrics["value_rel_rmse"],
                })


def plot_single_layer(results, layer_idx, output_path):
    """Bar chart: attn-output rel_rmse vs value rel_rmse side-by-side."""
    grans = [g for g in GRAN_ORDER if g in results]
    attn_vals = [results[g]["attnout_rel_rmse"] for g in grans]
    val_vals  = [results[g]["value_rel_rmse"]   for g in grans]

    x = np.arange(len(grans))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, attn_vals, width, label="attn-output rel_rmse", color="steelblue")
    bars2 = ax.bar(x + width/2, val_vals,  width, label="value-tensor rel_rmse", color="darkorange", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(grans)
    ax.set_ylabel("Relative RMSE")
    ax.set_title(f"Layer {layer_idx}: Value quantization — attn-output vs value-tensor RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multilayer(results_by_layer, output_path):
    """Line chart across layers for each granularity — attn-output rel_rmse."""
    layer_indices = sorted(results_by_layer.keys())
    grans_present = [g for g in GRAN_ORDER if g in results_by_layer[layer_indices[0]]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top: attn-output rel_rmse
    ax = axes[0]
    for gran in grans_present:
        vals = [results_by_layer[li][gran]["attnout_rel_rmse"] for li in layer_indices]
        ax.plot(layer_indices, vals, marker="o", linewidth=1.8, label=gran)
    ax.set_title("Attention-output rel_rmse by Value granularity (across layers)")
    ax.set_ylabel("Attn-output rel_rmse")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom: value-tensor rel_rmse (reference)
    ax = axes[1]
    for gran in grans_present:
        vals = [results_by_layer[li][gran]["value_rel_rmse"] for li in layer_indices]
        ax.plot(layer_indices, vals, marker="o", linewidth=1.8, linestyle="--", label=gran)
    ax.set_title("Value-tensor rel_rmse by Value granularity (across layers) [reference]")
    ax.set_ylabel("Value-tensor rel_rmse")
    ax.set_xlabel("Layer index")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_gran_comparison(results_by_layer, output_path):
    """
    For each layer show which granularity wins under each metric.
    Stacked bar of layer counts.
    """
    layer_indices = sorted(results_by_layer.keys())
    grans_present = [g for g in GRAN_ORDER if g in results_by_layer[layer_indices[0]]]

    best_attn = {}
    best_val  = {}
    for li in layer_indices:
        r = results_by_layer[li]
        best_attn[li] = min(grans_present, key=lambda g: r[g]["attnout_rel_rmse"])
        best_val[li]  = min(grans_present, key=lambda g: r[g]["value_rel_rmse"])

    attn_counts = {g: sum(1 for li in layer_indices if best_attn[li] == g) for g in grans_present}
    val_counts  = {g: sum(1 for li in layer_indices if best_val[li]  == g) for g in grans_present}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, counts, title in [
        (axes[0], attn_counts, "Best granularity by attn-output rel_rmse"),
        (axes[1], val_counts,  "Best granularity by value-tensor rel_rmse"),
    ]:
        grans = [g for g in grans_present if counts.get(g, 0) > 0]
        vals  = [counts[g] for g in grans]
        bars = ax.bar(grans, vals, color=plt.cm.tab10.colors[:len(grans)])
        ax.set_title(title)
        ax.set_ylabel("# layers")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle("Which Value granularity wins most layers?", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(results_by_layer, output_path):
    """
    Heatmap: (attnout_rel_rmse - value_rel_rmse) / value_rel_rmse
    Shows where the two metrics diverge most.
    """
    layer_indices = sorted(results_by_layer.keys())
    grans_present = [g for g in GRAN_ORDER if g in results_by_layer[layer_indices[0]]]

    data = np.zeros((len(grans_present), len(layer_indices)))
    for j, li in enumerate(layer_indices):
        r = results_by_layer[li]
        for i, g in enumerate(grans_present):
            val_r = r[g]["value_rel_rmse"]
            atn_r = r[g]["attnout_rel_rmse"]
            # positive = attn metric is harder (higher error); negative = easier
            data[i, j] = (atn_r - val_r) / max(val_r, EPS)

    fig, ax = plt.subplots(figsize=(max(12, len(layer_indices) * 0.5), 4))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_yticks(range(len(grans_present)))
    ax.set_yticklabels(grans_present)
    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels([str(li) for li in layer_indices], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_title("(attnout_rel_rmse − value_rel_rmse) / value_rel_rmse\nBlue = attn metric is EASIER (lower); Red = attn metric is HARDER")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Block analysis using attention-output RMSE (single layer or all layers)."
    )
    parser.add_argument("model", type=str)
    parser.add_argument("--seqlen",       type=int, default=2048)
    parser.add_argument("--maxseqlen",    type=int, default=2048)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--layer-idx",    type=int, default=10,
                        help="Layer to analyze (ignored if --all-layers is set).")
    parser.add_argument("--all-layers",   action="store_true",
                        help="Run analysis on all layers.")
    parser.add_argument("--layer-indices", type=int, nargs="*", default=None,
                        help="Explicit layer indices (overrides --all-layers).")
    parser.add_argument("--tile-sizes",   type=int, nargs="+", default=[32, 64])
    parser.add_argument("--num-bits",     type=int, default=4)
    parser.add_argument("--device",       type=str, default="cuda:0")
    parser.add_argument("--output-dir",   type=str, default=None)
    args = parser.parse_args()

    dev = torch.device(args.device)

    # ── Determine which layers to run ──
    if args.layer_indices is not None:
        layer_indices = sorted(set(args.layer_indices))
        mode = "multilayer"
    elif args.all_layers:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        layer_indices = list(range(cfg.num_hidden_layers))
        mode = "multilayer"
    else:
        layer_indices = [args.layer_idx]
        mode = "layer10" if args.layer_idx == 10 else f"layer{args.layer_idx}"

    # ── Output dir ──
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        tag = "multilayer-all" if mode == "multilayer" else mode
        output_dir = RESULTS_ROOT / f"attnout-block-analysis-llama7b-{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[attnout block analysis] layers={layer_indices}, output_dir={output_dir}")

    # ── Load model once ──
    print("Loading model …")
    model = get_model_longseqlen(args.model, args.seqlen, args.maxseqlen)
    model = model.half().eval().to(dev)

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    num_heads    = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim     = cfg.hidden_size // num_heads

    _, testenc = get_loaders("wikitext2", nsamples=1, seed=0,
                              model=args.model, seqlen=args.seqlen)
    input_ids = build_sample(testenc, args.seqlen, args.sample_index)

    # ── Capture all needed layers in ONE forward pass ──
    print(f"Capturing {len(layer_indices)} layers …")
    model_type = parse_model(model)
    layers = get_layers(model, model_type)
    all_captured = {li: {} for li in layer_indices}
    original_forwards = {}

    for layer_idx in layer_indices:
        attn = layers[layer_idx].self_attn
        orig = attn.forward
        original_forwards[layer_idx] = orig

        def make_hook(li, _attn, _orig):
            def wrapped(*args, **kwargs):
                hidden_states = args[0] if args else kwargs["hidden_states"]
                position_ids = kwargs.get("position_ids")
                position_embeddings = kwargs.get("position_embeddings")

                if not all_captured[li]:
                    bsz, q_len, _ = hidden_states.shape
                    q_raw, k_raw, v_raw = project_qkv(_attn, hidden_states)

                    q = q_raw.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                    k = k_raw.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                    v = v_raw.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

                    kv_len = k.shape[-2]
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                    else:
                        cos, sin = get_rope_cos_sin(_attn, v, position_ids, kv_len)

                    q_post, k_post = apply_rope(q, k, cos, sin, position_ids)

                    all_captured[li]["q_heads"] = q_post.detach().float().cpu()
                    all_captured[li]["k_post"]  = k_post.detach().float().cpu()
                    all_captured[li]["v_flat"]  = flatten_heads(v)[0].detach().float().cpu()

                return _orig(*args, **kwargs)
            return wrapped

        attn.forward = make_hook(layer_idx, attn, orig)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    with torch.no_grad():
        model(input_ids.to(dev))
    for li in layer_indices:
        layers[li].self_attn.forward = original_forwards[li]
    model.config.use_cache = use_cache
    print("Capture done.")

    # ── Run attn-output block analysis ──
    results_by_layer = {}
    for layer_idx in layer_indices:
        print(f"  Analyzing layer {layer_idx} …")
        cap = all_captured[layer_idx]
        results_by_layer[layer_idx] = build_attnout_report(
            cap["q_heads"], cap["k_post"], cap["v_flat"],
            num_heads, num_kv_heads, head_dim,
            args.num_bits, args.tile_sizes, dev,
        )

    # ── Write outputs ──
    write_csv(results_by_layer, output_dir / "attnout_layerwise_summary.csv")

    # Save JSON
    with (output_dir / "attnout_report.json").open("w") as f:
        json.dump({str(li): res for li, res in results_by_layer.items()}, f, indent=2)

    # Plots
    if len(layer_indices) == 1:
        li = layer_indices[0]
        plot_single_layer(results_by_layer[li], li, output_dir / "attnout_layer_barchart.png")
        print(f"[Layer {li}] results:")
        for gran, m in results_by_layer[li].items():
            print(f"  {gran:14s}  attnout_rel_rmse={m['attnout_rel_rmse']:.6f}  "
                  f"value_rel_rmse={m['value_rel_rmse']:.6f}")
    else:
        plot_multilayer(results_by_layer, output_dir / "attnout_multilayer_linechart.png")
        plot_best_gran_comparison(results_by_layer, output_dir / "attnout_best_gran_counts.png")
        plot_delta_heatmap(results_by_layer, output_dir / "attnout_delta_heatmap.png")

        # Also per-layer bar charts
        for li in layer_indices:
            plot_single_layer(results_by_layer[li], li,
                              output_dir / f"attnout_layer{li:02d}_barchart.png")

        # Console summary
        print("\nCross-layer summary (attnout_rel_rmse):")
        grans = list(next(iter(results_by_layer.values())).keys())
        for g in grans:
            vals = [results_by_layer[li][g]["attnout_rel_rmse"] for li in layer_indices]
            print(f"  {g:14s}: mean={np.mean(vals):.6f}  std={np.std(vals):.6f}  "
                  f"min={np.min(vals):.6f}  max={np.max(vals):.6f}")

        print("\nBest granularity per layer (attnout):")
        for li in layer_indices:
            r = results_by_layer[li]
            best = min(r, key=lambda g: r[g]["attnout_rel_rmse"])
            print(f"  layer {li:2d}: {best}  ({r[best]['attnout_rel_rmse']:.6f})")

    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
