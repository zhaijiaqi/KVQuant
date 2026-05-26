import argparse
import csv
import gc
import json
import math
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import torch


def add_quant_path(repo_root: Path):
    quant_path = repo_root / "kvquant" / "quant"
    if str(quant_path) not in sys.path:
        sys.path.insert(0, str(quant_path))


def get_model(model_name, seqlen, maxseqlen):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            use_flash_attention_2=True,
            torch_dtype=torch.half,
        )
    except (ImportError, ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.half,
        )
    model.seqlen = seqlen
    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)
    return model


@torch.no_grad()
def capture_fp16_hidden_states(model, dataloader, nsamples, dev, cache_path):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    h0 = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            h0[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    token_batches = []
    for batch in dataloader:
        if len(token_batches) >= nsamples:
            break
        token_batches.append(batch[0].cpu())
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if cache["i"] != nsamples:
        raise RuntimeError(f"Captured {cache['i']} samples, expected {nsamples}.")

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    hidden_states = [h0.cpu()]
    inps = h0
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i, layer in enumerate(layers):
        print(f"[cache] fp16 layer {i}", flush=True)
        layer = layer.to(dev)
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        hidden_states.append(outs.cpu().clone())
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    payload = {
        "tokens": torch.cat(token_batches, dim=0),
        "hidden_states": hidden_states,
        "attention_mask": attention_mask.cpu() if attention_mask is not None else None,
        "position_ids": position_ids.cpu() if position_ids is not None else None,
        "num_sequences": nsamples,
        "seq_len": model.seqlen,
        "num_layers": len(layers),
        "hidden_size": model.config.hidden_size,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def filter_quantizers(quantizers, layer_idx, module_suffix):
    key = f"model.layers.{layer_idx}.{module_suffix}"
    if key not in quantizers:
        raise KeyError(f"Missing quantizer key {key}")
    return {key: quantizers[key]}


def apply_layer_quantizers(layer, layer_idx, bit, granularity, q_baseline, q_candidate, args):
    from kvquant.simquant_module_quantizer import make_quant_sim

    key_q = filter_quantizers(q_baseline, layer_idx, "self_attn.k_proj")
    make_quant_sim(
        layer,
        key_q,
        bit,
        name=f"model.layers.{layer_idx}",
        perchannel=True,
        include_sparse=True,
        sparsity_threshold=args.sparsity_threshold,
        dynamicquantization=False,
        nuq=True,
        nf_nuq=False,
        norm=args.norm,
        cap_outliers=-1,
        first_few_fp16=-1,
        clamp=False,
    )

    value_q = filter_quantizers(q_candidate, layer_idx, "self_attn.v_proj")
    if granularity == "per-token":
        make_quant_sim(
            layer,
            value_q,
            bit,
            name=f"model.layers.{layer_idx}",
            perchannel=False,
            include_sparse=True,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=True,
            nuq=True,
            nf_nuq=False,
            norm=args.norm,
            cap_outliers=-1,
            first_few_fp16=-1,
            clamp=False,
        )
    elif granularity == "per-channel":
        make_quant_sim(
            layer,
            value_q,
            bit,
            name=f"model.layers.{layer_idx}",
            perchannel=True,
            include_sparse=True,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=False,
            nuq=True,
            nf_nuq=False,
            norm=args.norm,
            cap_outliers=-1,
            first_few_fp16=-1,
            clamp=False,
        )
    elif granularity == "tile32":
        make_quant_sim(
            layer,
            value_q,
            bit,
            name=f"model.layers.{layer_idx}",
            perchannel=False,
            include_sparse=True,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=True,
            nuq=True,
            nf_nuq=False,
            norm=args.norm,
            cap_outliers=-1,
            first_few_fp16=-1,
            clamp=False,
            tile_size=32,
        )
    else:
        raise ValueError(granularity)


@torch.no_grad()
def evaluate_candidate(model, payload, layer_idx, bit, granularity, qmaps, args, dev):
    num_layers = payload["num_layers"]
    k_eff = min(args.k, num_layers - layer_idx)
    inps = payload["hidden_states"][layer_idx].to(dev)
    outs = torch.empty_like(inps)
    attention_mask = payload["attention_mask"].to(dev) if payload["attention_mask"] is not None else None
    position_ids = payload["position_ids"].to(dev) if payload["position_ids"] is not None else None

    baseline = qmaps[(bit, "per-token")]
    candidate = qmaps[(bit, granularity)]

    for cur_layer in range(layer_idx, layer_idx + k_eff):
        layer = deepcopy(model.model.layers[cur_layer]).half()
        cand_granularity = granularity if cur_layer == layer_idx else "per-token"
        cand_qmap = candidate if cur_layer == layer_idx else baseline
        apply_layer_quantizers(layer, cur_layer, bit, cand_granularity, baseline, cand_qmap, args)
        layer = layer.to(dev)
        for j in range(payload["num_sequences"]):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        layer = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    ref = payload["hidden_states"][layer_idx + k_eff].to(dev).float()
    cand = inps.float()
    diff = cand - ref
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    ref_rms = torch.sqrt(torch.mean(ref * ref)).item()
    rel_rmse = rmse / (ref_rms + 1e-12)

    del inps, outs, ref, cand, diff
    torch.cuda.empty_cache()
    return k_eff, rmse, ref_rms, rel_rmse


def write_strategy_outputs(rows, result_dir, strategy_dir, k, num_layers):
    result_dir.mkdir(parents=True, exist_ok=True)
    strategy_dir.mkdir(parents=True, exist_ok=True)

    strategy_rows = []
    for bit in [2, 3, 4]:
        selected = {}
        for layer in range(num_layers):
            choices = [r for r in rows if r["layer"] == layer and r["bit"] == bit]
            best = min(choices, key=lambda r: r["rel_rmse"])
            selected[str(layer)] = {"bit": bit, "granularity": best["granularity"]}

        name = f"k4_fixed{bit}_value_strategy"
        path = strategy_dir / f"{name}.json"
        payload = {
            "name": name,
            "mode": "fixed-bit",
            "bit": bit,
            "K": k,
            "value_strategy": selected,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

        buckets = {"per-token": [], "per-channel": [], "tile32": []}
        for layer_s, spec in selected.items():
            buckets[spec["granularity"]].append(int(layer_s))
        strategy_rows.append(
            {
                "strategy": name,
                "avg_bit": bit,
                "num_per_token": len(buckets["per-token"]),
                "num_per_channel": len(buckets["per-channel"]),
                "num_tile32": len(buckets["tile32"]),
                "per_token_layers": json.dumps(buckets["per-token"]),
                "per_channel_layers": json.dumps(buckets["per-channel"]),
                "tile32_layers": json.dumps(buckets["tile32"]),
            }
        )
        print(f"[strategy] {name}: {strategy_rows[-1]}", flush=True)

    with (result_dir / "k4_strategy_summary.csv").open("w", newline="") as handle:
        fieldnames = [
            "strategy",
            "avg_bit",
            "num_per_token",
            "num_per_channel",
            "num_tile32",
            "per_token_layers",
            "per_channel_layers",
            "tile32_layers",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(strategy_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/LLaMA-7B")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--result-dir", default="results/k4-lookahead")
    parser.add_argument("--strategy-dir", default="strategies")
    parser.add_argument("--quantizer-dir", default="/data/kvquant/quantizers/k4_lookahead")
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--maxseqlen", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--sparsity-threshold", type=float, default=0.99)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    add_quant_path(repo_root)
    from kvquant.datautils import get_loaders

    result_dir = repo_root / args.result_dir
    strategy_dir = repo_root / args.strategy_dir
    quantizer_dir = Path(args.quantizer_dir)
    cache_path = result_dir / "cache" / "fp16_hidden_states_layer0_to_32.pt"
    csv_path = result_dir / "k4_lookahead_layer_candidate_rmse.csv"

    dev = torch.device("cuda:0")
    model = get_model(args.model, args.seqlen, args.maxseqlen).half().eval()
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=args.seqlen,
    )

    if args.reuse_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
    else:
        payload = capture_fp16_hidden_states(model, dataloader, args.nsamples, dev, cache_path)

    qmaps = {}
    for bit in [2, 3, 4]:
        for granularity in ["per-token", "per-channel", "tile32"]:
            path = quantizer_dir / f"lookahead_nuq{bit}_{granularity}.pkl"
            if not path.exists():
                raise FileNotFoundError(path)
            qmaps[(bit, granularity)] = load_pickle(path)

    rows = []
    fieldnames = [
        "layer",
        "K_eff",
        "bit",
        "granularity",
        "rel_rmse",
        "rmse",
        "ref_rms",
        "num_sequences",
        "seq_len",
    ]
    result_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer_idx in range(payload["num_layers"]):
            for bit in [2, 3, 4]:
                args.norm = bit == 2
                for granularity in ["per-token", "per-channel", "tile32"]:
                    print(
                        f"[eval] layer={layer_idx} bit={bit} granularity={granularity}",
                        flush=True,
                    )
                    k_eff, rmse, ref_rms, rel_rmse = evaluate_candidate(
                        model, payload, layer_idx, bit, granularity, qmaps, args, dev
                    )
                    row = {
                        "layer": layer_idx,
                        "K_eff": k_eff,
                        "bit": bit,
                        "granularity": granularity,
                        "rel_rmse": rel_rmse,
                        "rmse": rmse,
                        "ref_rms": ref_rms,
                        "num_sequences": payload["num_sequences"],
                        "seq_len": payload["seq_len"],
                    }
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
                    gc.collect()

    write_strategy_outputs(rows, result_dir, strategy_dir, args.k, payload["num_layers"])
    print(f"[done] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
