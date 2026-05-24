"""
bench_kernel_key_opt.py
Benchmark nuq4-1% Key kernel using the _opt2 API.
Corresponds to Table 6: "Key nuq4-1%" column.

Usage (run from benchmarking/):
  python scripts/bench_kernel_key_opt.py --seqlen 2048
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
DEV        = torch.device('cuda:0')

print(f'Benchmarking Key nuq4-1% kernel, seqlen={N} ...')

with open(f'activations-seqlen{N}.pickle', 'rb') as f:
    activations = pickle.load(f)
with open(f'quantizers.pickle', 'rb') as f:
    quantizers = pickle.load(f)

nf4_signposts = get_nf4_signposts(bits)

d = {}
for l in range(32):
    quantizer = quantizers[f'model.layers.{l}.self_attn.k_proj']
    maxval = torch.tensor(quantizer[0]).cuda().half().squeeze(0)  # (num_heads*head_dim,)
    minval = torch.tensor(quantizer[1]).cuda().half().squeeze(0)
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2

    # Per-channel lookup table: (num_heads, head_dim, 2^bits)
    lut = torch.zeros((num_heads, head_dim, 2**bits), dtype=torch.float, device=DEV)
    for i in range(num_heads):
        for j in range(head_dim):
            idx = i * head_dim + j
            lut[i, j] = torch.tensor(nf4_signposts) * rangeval[idx].item() + offset[idx].item()

    # Build kcache and sparse outlier tensors via append kernel
    kcache = torch.zeros((num_heads, (head_dim // 32) * bits, N), dtype=torch.int, device=DEV)
    outliers_store = torch.zeros((N, 42), dtype=torch.float, device=DEV)
    outlier_idx_store = torch.zeros((N, 42), dtype=torch.int, device=DEV)

    zeropoint = offset.float().cuda()
    outlier_threshold_lower = minval.float()
    outlier_threshold_upper = maxval.float()

    k_act = activations[f'self_attn.k_proj.{l}']
    rows2 = torch.tensor([], device=DEV)
    cols2 = torch.tensor([], device=DEV)
    vals2 = torch.tensor([], device=DEV)
    start_rows = torch.tensor([], device=DEV)
    for i in range(N):
        newk = k_act[i % k_act.shape[0]].float()
        rows2, cols2, vals2, start_rows, num_threads, outlier_count = \
            quant_cuda.vecquant4appendvecKsparseorig(
                kcache, lut, newk, zeropoint,
                rows2, cols2, vals2, start_rows,
                outlier_threshold_lower, outlier_threshold_upper, i)

    q   = torch.ones((1, num_heads, head_dim), dtype=torch.float, device=DEV)
    mul = torch.zeros((1, num_heads, N), dtype=torch.float, device=DEV)

    d[f'l{l}_q']        = q
    d[f'l{l}_kcache']   = kcache
    d[f'l{l}_mul']      = mul
    d[f'l{l}_lut']      = lut
    d[f'l{l}_N']        = N
    d[f'l{l}_outliers'] = outliers_store
    d[f'l{l}_oidx']     = outlier_idx_store

# warmup
for _ in range(num_iters):
    quant_cuda.vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
        d['l0_q'], d['l0_kcache'], d['l0_mul'], d['l0_lut'],
        d['l0_N'], d['l0_outliers'], d['l0_oidx'], 10000.0, 0)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for l in range(32):
        for _ in range(num_iters):
            quant_cuda.vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
                d[f'l{l}_q'], d[f'l{l}_kcache'], d[f'l{l}_mul'], d[f'l{l}_lut'],
                d[f'l{l}_N'], d[f'l{l}_outliers'], d[f'l{l}_oidx'], 10000.0, 0)
            torch.cuda.synchronize()

print(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=-1))
