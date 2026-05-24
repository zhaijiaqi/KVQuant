"""
bench_kernel_value_opt.py
Benchmark nuq4-1% Value kernel using the _opt2 API.
Corresponds to Table 6: "Value nuq4-1%" column.

Usage (run from benchmarking/):
  python scripts/bench_kernel_value_opt.py --seqlen 2048
"""
import argparse, pickle
import torch
import quant_cuda
from torch.distributions import Normal

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def get_nf4_signposts(bits=4):
    dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    offsets = [0.5*(1/32 + 1/30), 1 - 0.5*(1/32 + 1/30)]
    list1 = [offsets[0]]
    spacing = (0.5 - offsets[0]) / (2 ** (bits - 1) - 1)
    add = offsets[0]
    for i in range(2 ** (bits-1) - 1):
        add += spacing; list1.append(add)
    list2 = []
    spacing = (offsets[1] - 0.5) / (2 ** (bits - 1))
    add = 0.5
    for i in range(2 ** (bits-1)):
        list2.append(add); add += spacing
    list2.append(offsets[-1])
    nf4_neg = [dist.icdf(torch.tensor([v])).item() for v in list1]
    nf4_pos = [dist.icdf(torch.tensor([v])).item() for v in list2]
    off = abs(nf4_neg[-1]); rng = abs(nf4_neg[0]) - abs(nf4_neg[-1])
    nf4_neg = [(x + off) / rng for x in nf4_neg]
    off = abs(nf4_pos[0]); rng = abs(nf4_pos[-1]) - abs(nf4_pos[0])
    nf4_pos = [(x - off) / rng for x in nf4_pos]
    del nf4_pos[0]
    sp = nf4_neg + nf4_pos
    assert len(sp) == 2**bits
    return sp

parser = argparse.ArgumentParser()
parser.add_argument('--seqlen', type=int, default=2048)
parser.add_argument('--num_iters', type=int, default=1000)
args = parser.parse_args()

N          = args.seqlen
num_iters  = args.num_iters
num_heads  = 32
head_dim   = 128
bits       = 4
hidden_size = num_heads * head_dim
threshold_k = int(((1 - 0.99) / 2) * hidden_size) + 1  # 1% outliers
DEV        = torch.device('cuda:0')

print(f'Benchmarking Value nuq4-1% kernel, seqlen={N} ...')

with open(f'activations-seqlen{N}.pickle', 'rb') as f:
    activations = pickle.load(f)

nf4_signposts = get_nf4_signposts(bits)
lut_1d    = torch.tensor(nf4_signposts, dtype=torch.float, device=DEV)  # (16,) for scalar math
lut_1d_2d = lut_1d.unsqueeze(0)  # (1, 16) required by vecquant4appendvecVsparseorig

d = {}
for l in range(32):
    v_act = activations[f'self_attn.v_proj.{l}']

    # vcache: (num_heads, (head_dim // 32) * bits, N) = (32, 16, N) for 4-bit
    # Same packed layout as kcache — see deployment/transformers/modeling_llama.py:1012
    vcache = torch.zeros((num_heads, (head_dim // 32) * bits, N), dtype=torch.int, device=DEV)
    outliers_store     = torch.zeros((N, 42), dtype=torch.float, device=DEV)
    outlier_idx_store  = torch.zeros((N, 42), dtype=torch.int,   device=DEV)

    # Per-token lookup table for Value: (N, 2^bits)
    lut_pertok = torch.zeros((N, 2**bits), dtype=torch.float, device=DEV)

    rows2 = torch.tensor([], device=DEV)
    cols2 = torch.tensor([], device=DEV)
    vals2 = torch.tensor([], device=DEV)
    start_cols = torch.tensor([], device=DEV)

    for i in range(N):
        newv = v_act[i % v_act.shape[0]].float()
        sorted_v, _ = newv.sort()
        minval = sorted_v[threshold_k]
        maxval = sorted_v[-threshold_k]
        sf = (maxval - minval) / 2
        offset = (maxval + minval) / 2
        zeropoint_tmp = lut_1d[7] * sf.item() + offset.item()
        lut_pertok[i] = lut_1d * sf.item() + offset.item()

        rows2, cols2, vals2, start_cols, num_threads, outlier_count = \
            quant_cuda.vecquant4appendvecVsparseorig(
                vcache, lut_1d_2d, newv, zeropoint_tmp.item(),
                rows2, cols2, vals2, start_cols,
                minval.item(), maxval.item(), i)

    score = torch.ones((1, num_heads, N), dtype=torch.float, device=DEV)
    mul   = torch.zeros((1, num_heads, head_dim), dtype=torch.float, device=DEV)

    d[f'l{l}_score']    = score
    d[f'l{l}_vcache']   = vcache
    d[f'l{l}_mul']      = mul
    d[f'l{l}_lut']      = lut_pertok  # (N, 16) per-token LUT
    d[f'l{l}_N']        = N
    d[f'l{l}_outliers'] = outliers_store
    d[f'l{l}_oidx']     = outlier_idx_store

# Test call once to verify shapes
try:
    quant_cuda.vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
        d['l0_score'], d['l0_vcache'], d['l0_mul'], d['l0_lut'],
        d['l0_N'], d['l0_outliers'], d['l0_oidx'])
    print('  shape test OK')
except Exception as e:
    print(f'  shape test FAILED: {e}')
    raise

# warmup
for _ in range(num_iters):
    quant_cuda.vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
        d['l0_score'], d['l0_vcache'], d['l0_mul'], d['l0_lut'],
        d['l0_N'], d['l0_outliers'], d['l0_oidx'])
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for l in range(32):
        for _ in range(num_iters):
            quant_cuda.vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
                d[f'l{l}_score'], d[f'l{l}_vcache'], d[f'l{l}_mul'], d[f'l{l}_lut'],
                d[f'l{l}_N'], d[f'l{l}_outliers'], d[f'l{l}_oidx'])
            torch.cuda.synchronize()

print(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=-1))
