#!/usr/bin/env python3
"""Sparsity sweep with ORIGINAL n16 historical quantizers."""
import pickle, sys, gc, torch, time
from pathlib import Path

REPO = Path.home() / "program" / "rlkvq"
sys.path.insert(0, str(REPO / "kvquant" / "quant"))
sys.path.insert(0, str(REPO / "scripts" / "k4_lookahead_experiment"))
import llama_simquant as sim
from kvquant.datautils import get_loaders
from kvquant.simquant_module_quantizer import QuantLinearSim

DEV = torch.device("cuda:0")
MODEL = "/data/models/LLaMA-7B"
QDIR = Path("/data/kvquant/quantizers")

bits_4 = [31, 9, 7, 23, 1, 6, 8, 13]
bits_2 = [27, 25, 24, 16, 26, 21, 0, 22]
def bfl(l): return 4 if l in bits_4 else (2 if l in bits_2 else 3)
# ORIGINAL n16 historical quantizers
qmaps = {b: pickle.loads((QDIR / f"nuq{b}_s1.pkl").read_bytes()) for b in [2, 3, 4]}

def do_eval(sparsity, uniform=False):
    model = sim.get_model(MODEL, 2048, 2048).half().eval()
    model.seqlen = 2048
    for l in range(32):
        bit = 3 if uniform else bfl(l)
        for attr in ["k_proj", "v_proj"]:
            name = f"model.layers.{l}.self_attn.{attr}"
            parent = model.get_submodule(f"model.layers.{l}.self_attn")
            old = getattr(parent, attr)
            w = old.weight.data.clone()
            b = old.bias.data.clone() if old.bias is not None else None
            in_f, out_f = w.shape[1], w.shape[0]
            qp = qmaps[3][name] if uniform else qmaps[bit][name]
            new_q = QuantLinearSim(name, bit, qp, in_f, out_f,
                w.cuda(), b.cuda() if b is not None else None,
                perchannel=(attr == "k_proj"), sparsity_threshold=sparsity,
                include_sparse=True,
                dynamicquantization=(attr == "v_proj"),
                nuq=True, nf_nuq=False, norm=(bit == 2 and not uniform))
            setattr(parent, attr, new_q)
    model.cuda(); torch.cuda.synchronize(); gc.collect()
    _, tl = get_loaders("wikitext2", nsamples=16, seed=0, model=MODEL, seqlen=2048)
    t0 = time.time()
    ppl = sim.llama_eval(model, tl, DEV)
    dt = time.time() - t0
    model.cpu(); del model; gc.collect(); torch.cuda.empty_cache()
    return ppl, dt

print("SWEEP_N16", flush=True)
# Uniform 3-bit baseline (n16 original)
pu, du = do_eval(0.99, uniform=True)
print(f"U 0.99: {pu:.6f} ({du:.1f}s)", flush=True)

# Mixed with original n16 quantizers
for st in [0.99, 0.95, 0.9, 0.8, 0.5]:
    ppl, dt = do_eval(st)
    d = ppl - pu
    f = "BETTER" if ppl < pu else "WORSE"
    print(f"M {st:.3f}: {ppl:.6f} d={d:+.6f} ({dt:.1f}s) {f}", flush=True)
print("DONE", flush=True)
