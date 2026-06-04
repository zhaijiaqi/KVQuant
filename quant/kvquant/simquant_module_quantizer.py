import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.cluster import KMeans

import torch
from torch.distributions import Normal

def round_to_nearest_pole_sim(w, poles, return_freq=False):
    """
    w: weight/act values (1d vector)
    poles: tuple of values

    Round the numbers in w to the nearest value in poles.
    """
    stack = []
    for c in poles:
        diff = (w - c).abs()
        stack.append(diff)
    diff = torch.stack(stack)
    idx = diff.argmin(axis=0)
    aug = 0
    freq = []
    for i, c in enumerate(poles):
        mask = idx == i
        aug += mask * c
        if return_freq:
            freq.append(mask.sum())

    if return_freq:
        return aug, freq
    return aug



def _token_window_slices(num_tokens, sink_tokens=-1, recent_tokens=-1, seq_len=2048):
    """Return row slices to preserve in fp16 for flattened [tokens, hidden] activations."""
    sink_tokens = 0 if sink_tokens is None or sink_tokens < 0 else int(sink_tokens)
    recent_tokens = 0 if recent_tokens is None or recent_tokens < 0 else int(recent_tokens)
    if sink_tokens == 0 and recent_tokens == 0:
        return []
    if seq_len is None or seq_len <= 0:
        seq_len = num_tokens
    seq_len = int(seq_len)
    spans = []
    for seq_start in range(0, num_tokens, seq_len):
        seq_end = min(seq_start + seq_len, num_tokens)
        if sink_tokens > 0:
            sink_end = min(seq_start + sink_tokens, seq_end)
            if sink_end > seq_start:
                spans.append((seq_start, sink_end))
        if recent_tokens > 0:
            recent_start = max(seq_start, seq_end - recent_tokens)
            if seq_end > recent_start:
                spans.append((recent_start, seq_end))
    return spans


def _mark_fp16_token_window(mask, sink_tokens=-1, recent_tokens=-1, seq_len=2048):
    for start, end in _token_window_slices(mask.shape[0], sink_tokens, recent_tokens, seq_len):
        mask[start:end, :] = True
    return mask


def _restore_fp16_token_window(qinp_out, orig, sink_tokens=-1, recent_tokens=-1, seq_len=2048):
    for start, end in _token_window_slices(qinp_out.shape[0], sink_tokens, recent_tokens, seq_len):
        qinp_out[start:end, :] = orig[start:end, :]
    return qinp_out



def _apply_head_rotation(inp, rotation, transpose=False):
    if rotation is None:
        return inp
    orig_dtype = inp.dtype
    hidden = inp.shape[-1]
    rot = rotation.to(device=inp.device, dtype=torch.float32)
    if rot.dim() == 2:
        out = inp.float().matmul(rot.t() if transpose else rot)
        return out.to(orig_dtype)
    if rot.dim() != 3:
        raise ValueError(f"rotation must have shape [hidden, hidden] or [heads, dim, dim], got {tuple(rot.shape)}")
    heads, head_dim, head_dim2 = rot.shape
    if head_dim != head_dim2 or heads * head_dim != hidden:
        raise ValueError(f"rotation shape {tuple(rot.shape)} is incompatible with hidden dim {hidden}")
    x = inp.float().reshape(-1, heads, head_dim)
    if transpose:
        out = torch.einsum('nhd,hed->nhe', x, rot)
    else:
        out = torch.einsum('nhd,hde->nhe', x, rot)
    return out.reshape(inp.shape).to(orig_dtype)



def _clip_activation(inp, clip_ratio=-1, qchannel=-1, method="maxabs"):
    if clip_ratio is None or clip_ratio <= 0 or clip_ratio >= 1:
        return inp
    work = inp.float()
    if method == "percentile":
        # OSCAR-style clipping: keep a fixed percentile mass per token row.
        # kthvalue is much faster than torch.quantile in the full-PPL loop.
        abs_work = work.abs()
        k = int(math.ceil(float(clip_ratio) * abs_work.shape[-1]))
        k = max(1, min(k, abs_work.shape[-1]))
        limit = torch.kthvalue(abs_work, k, dim=-1, keepdim=True).values
    elif method == "maxabs":
        limit = work.abs().amax(dim=qchannel, keepdim=True) * float(clip_ratio)
    else:
        raise ValueError(f"Unknown clip method: {method}")
    limit = torch.clamp(limit, min=1e-8)
    return torch.clamp(work, min=-limit, max=limit).to(inp.dtype)


KEY_TOKEN_SCALE_STATS = {}


def reset_key_token_scale_stats():
    KEY_TOKEN_SCALE_STATS.clear()


def get_key_token_scale_stats():
    result = {}
    for name, stat in KEY_TOKEN_SCALE_STATS.items():
        count = max(1, int(stat["count"]))
        cv_count = max(1, int(stat.get("cv_count", 0)))
        result[name] = {
            "count": int(stat["count"]),
            "norm_mean": float(stat["norm_sum"] / count),
            "norm_min": float(stat["norm_min"]),
            "norm_max": float(stat["norm_max"]),
            "scale_mean": float(stat["scale_sum"] / count),
            "scale_min": float(stat["scale_min"]),
            "scale_max": float(stat["scale_max"]),
            "norm_cv_mean": float(stat.get("norm_cv_sum", 0.0) / cv_count),
            "nan_count": int(stat.get("nan_count", 0)),
            "inf_count": int(stat.get("inf_count", 0)),
        }
    return result


def _accum_key_token_scale_stats(name, norm, scale, valid_mask):
    work_norm = norm.detach().float()
    work_scale = scale.detach().float()
    valid = valid_mask.detach().bool()
    if valid.any():
        n = work_norm[valid]
        s = work_scale[valid]
    else:
        n = work_norm.reshape(-1)
        s = work_scale.reshape(-1)
    nan_count = torch.isnan(n).sum().item() + torch.isnan(s).sum().item()
    inf_count = torch.isinf(n).sum().item() + torch.isinf(s).sum().item()
    n = torch.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    count = int(n.numel())
    if count == 0:
        return
    # CV is computed per call over quantized tokens only; this is a lightweight sanity statistic.
    mean_n = n.mean()
    cv = (n.std(unbiased=False) / torch.clamp(mean_n.abs(), min=1e-8)).item()
    cur = KEY_TOKEN_SCALE_STATS.setdefault(name, {
        "count": 0,
        "norm_sum": 0.0,
        "norm_min": float("inf"),
        "norm_max": float("-inf"),
        "scale_sum": 0.0,
        "scale_min": float("inf"),
        "scale_max": float("-inf"),
        "norm_cv_sum": 0.0,
        "cv_count": 0,
        "nan_count": 0,
        "inf_count": 0,
    })
    cur["count"] += count
    cur["norm_sum"] += float(n.sum().item())
    cur["norm_min"] = min(cur["norm_min"], float(n.min().item()))
    cur["norm_max"] = max(cur["norm_max"], float(n.max().item()))
    cur["scale_sum"] += float(s.sum().item())
    cur["scale_min"] = min(cur["scale_min"], float(s.min().item()))
    cur["scale_max"] = max(cur["scale_max"], float(s.max().item()))
    cur["norm_cv_sum"] += float(cv)
    cur["cv_count"] += 1
    cur["nan_count"] += int(nan_count)
    cur["inf_count"] += int(inf_count)


def _key_token_scale_mask(num_tokens, device, sink_tokens=-1, recent_tokens=-1, seq_len=2048):
    mask = torch.ones((num_tokens,), dtype=torch.bool, device=device)
    for start, end in _token_window_slices(num_tokens, sink_tokens, recent_tokens, seq_len):
        mask[start:end] = False
    return mask


def _apply_key_token_scaling(inp, target="none", eps=1e-6, sink_tokens=-1, recent_tokens=-1, seq_len=2048, name=""):
    if target is None or target == "none":
        return inp, None
    orig_dtype = inp.dtype
    work = inp.float()
    hidden = work.shape[-1]
    num_tokens = work.shape[0]
    if hidden % 128 == 0:
        head_dim = 128
        heads = hidden // head_dim
    else:
        heads = 1
        head_dim = hidden
    x = work.reshape(num_tokens, heads, head_dim)
    norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + float(eps))
    valid_mask = _key_token_scale_mask(num_tokens, work.device, sink_tokens, recent_tokens, seq_len)
    valid_h = valid_mask[:, None, None]
    if target == "unit":
        target_norm = torch.ones_like(norm)
    elif target == "mean-norm":
        valid_count = torch.clamp(valid_h.sum().float(), min=1.0)
        mean_norm = (norm * valid_h.float()).sum(dim=0, keepdim=True) / valid_count
        target_norm = mean_norm.expand_as(norm)
    else:
        raise ValueError(f"Unknown key token scale target: {target}")
    scale = target_norm / torch.clamp(norm, min=float(eps))
    scale = torch.where(valid_h, scale, torch.ones_like(scale))
    _accum_key_token_scale_stats(name, norm.squeeze(-1), scale.squeeze(-1), valid_mask[:, None].expand(-1, heads))
    y = (x * scale).reshape_as(work)
    return y.to(orig_dtype), scale


def _invert_key_token_scaling(inp, scale):
    if scale is None:
        return inp
    orig_dtype = inp.dtype
    work = inp.float()
    num_tokens, hidden = work.shape
    heads = scale.shape[1]
    head_dim = hidden // heads
    x = work.reshape(num_tokens, heads, head_dim)
    y = x / torch.clamp(scale.to(work.device, dtype=torch.float32), min=1e-8)
    return y.reshape_as(work).to(orig_dtype)

def get_outliers(
    w,
    channel=-1,
    outlier_threshold_upper=-1,
    outlier_threshold_lower=-1,
    cap_outliers=-1,
    first_few_fp16=-1,
    recent_fp16=-1,
    fp16_seq_len=2048
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    outlier_threshold_upper: upper outlier thresholds
    outlier_threshold_lower: lower outlier thresholds
    first_few_fp16: number of initial tokens to keep in fp16
    recent_fp16: number of sequence-tail tokens to keep in fp16

    Detect outliers above upper threshold / below lower threshold
    """
    # only use either per-channel or per-token outlier
    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)

    under_lower = w < outlier_threshold_lower
    above_upper = w > outlier_threshold_upper

    outlier_mask = torch.logical_or(under_lower, above_upper)

    if cap_outliers > -1:
        outlier_mask_tmp = outlier_mask.clone()

        zero_point = (outlier_threshold_upper + outlier_threshold_lower) / 2
        distance = (outlier_threshold_upper - outlier_threshold_lower) / 2
        outliers = w * outlier_mask

        values = torch.zeros_like(outliers)
        values[outlier_mask] = ((w - zero_point) / distance)[outlier_mask]

        upper_values, upper_indices = torch.topk(values, 21, dim=-1)
        lower_values, lower_indices = torch.topk(values, 21, dim=-1, largest=False)
        indices_combined = torch.cat((upper_indices, lower_indices), dim=-1)
        values_combined = torch.cat((upper_values, lower_values), dim=-1)

        values2 = torch.zeros_like(outliers)
        values2.scatter_(-1, indices_combined, values_combined)
        outlier_mask = values2 != 0

    _mark_fp16_token_window(
        outlier_mask,
        sink_tokens=first_few_fp16,
        recent_tokens=recent_fp16,
        seq_len=fp16_seq_len,
    )

    return outlier_mask

def get_outliers_dynamic(
    w,
    channel=-1,
    thresh=0.999,
    first_few_fp16=-1,
    recent_fp16=-1,
    fp16_seq_len=2048
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    thresh: percentile for outlier threshold computation
    first_few_fp16: number of initial tokens to keep in fp16
    recent_fp16: number of sequence-tail tokens to keep in fp16

    Detect outliers above upper threshold / below lower threshold
    """

    t = 1-((1-thresh)/2)
    w = w.float()

    # only use either per-channel or per-token outlier
    outlier_threshold_upper = torch.quantile(w, t, dim=channel)
    outlier_threshold_lower = torch.quantile(w, 1-t, dim=channel)

    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)

    under_lower = w <= outlier_threshold_lower
    above_upper = w >= outlier_threshold_upper

    outlier_mask = torch.logical_or(under_lower, above_upper)

    _mark_fp16_token_window(
        outlier_mask,
        sink_tokens=first_few_fp16,
        recent_tokens=recent_fp16,
        seq_len=fp16_seq_len,
    )

    return outlier_mask

# integer quantization function
def quant_fn_zp(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    clamp=False
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    clamp: whether to round and clamp the zeropoint

    Performs simulated integer quantization
    """

    # set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using median to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    rangeval = (maxval - minval)
    qx = (2**bits - 1) / rangeval

    # set offset
    if clamp:
        offset = torch.round(minval * qx)
        offset = offset.clamp(-(2**bits - 1), 0)
    else: # improves accuracy with per-channel key quantization
        offset = minval * qx

    offset = offset.unsqueeze(qchannel)
    qx = qx.unsqueeze(qchannel)

    # need to handle outlier removal
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    # scale and subtract offset
    qinp = torch.round(qx * inp - offset)

    #clipping (just for debugging purposes)
    qinp = torch.clip(qinp, min=0, max=2**bits - 1)

    #rescale
    qinp_out = (qinp + offset) / qx

    # add outliers back
    if include_sparse:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers

    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)
    return qinp_out


def quant_fn_oscar_affine(
    inp,
    bits=8,
    qchannel=-1,
    first_few_fp16=-1,
    recent_fp16=-1,
    fp16_seq_len=2048,
):
    """OSCAR-style dynamic affine activation quantization.

    OSCAR rotates KV into a friendlier basis, clips outliers, and then applies
    standard affine INT quantization. Unlike KVQuant NUQ, this does not reuse
    KMeans centroids or static thresholds learned in a different basis.
    """
    if first_few_fp16 > -1 or recent_fp16 > -1:
        orig = inp

    work = inp.float()
    maxval = torch.amax(work, dim=qchannel, keepdim=True)
    minval = torch.amin(work, dim=qchannel, keepdim=True)
    qmax = float(2 ** bits - 1)
    scale = torch.clamp((maxval - minval) / qmax, min=1e-8)
    zero_point = torch.round(-minval / scale).clamp(0, qmax)
    q = torch.round(work / scale + zero_point).clamp(0, qmax)
    qinp_out = (q - zero_point) * scale
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)

    if first_few_fp16 > -1 or recent_fp16 > -1:
        _restore_fp16_token_window(
            qinp_out,
            orig,
            sink_tokens=first_few_fp16,
            recent_tokens=recent_fp16,
            seq_len=fp16_seq_len,
        )

    return qinp_out.float()

def quant_fn_nf(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    nf_lut=None
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    nf_lut: NormalFloat signpost values

    Performs simulated NormalFloat quantization
    """

    # set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using mean to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2
    offset = offset.unsqueeze(qchannel)
    rangeval = rangeval.unsqueeze(qchannel)

    # subtract offset
    inp = inp - offset

    # need to handle outlier removal here due to issues with zeroing out non-outliers
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    #dividing by range to normalize to [-1,1]
    inp_scaled = inp / rangeval

    Q = round_to_nearest_pole_sim(inp_scaled.flatten(), nf_lut)
    qinp_out = Q.reshape(inp.shape).half().cuda()
    qinp_out = qinp_out * rangeval

    # add outliers back
    if include_sparse:
        qinp_out = qinp_out + outliers

    #shift by offset
    qinp_out = qinp_out + offset
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0) #TODO: debug (shouldn't be necessary)

    return qinp_out

def quant_fn_nuq_recon(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    lut=None,
    norm=False,
    normscale=None,
    normoffset=None,
    first_few_fp16=-1,
    recent_fp16=-1,
    fp16_seq_len=2048
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    lut: NUQ signpost values
    norm: whether to use Q-Norm
    normscale: scaling for Q-Norm
    normoffset: shift for Q-Norm
    first_few_fp16: number of initial tokens to keep in fp16
    recent_fp16: number of sequence-tail tokens to keep in fp16

    Performs simulated NUQ quantization
    """

    if first_few_fp16 > -1 or recent_fp16 > -1:
        orig = inp

    # set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using mean to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2
    offset = offset.unsqueeze(qchannel)
    rangeval = rangeval.unsqueeze(qchannel)

    # subtract offset
    inp = inp - offset

    # need to handle outlier removal here due to issues with zeroing out non-outliers
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    #dividing by range to normalize to [-1,1]
    inp_scaled = inp / rangeval

    # round to nearest LUT entry
    lut_cuda = torch.tensor(lut[0]).to(inp_scaled.device)
    Q = round_to_nearest_pole_sim(inp_scaled.flatten(), lut_cuda)
    qinp_out = Q.reshape(inp.shape).float().to(inp_scaled.device)

    if norm:
        normscale = normscale.to(inp_scaled.device)
        normoffset = normoffset.to(inp_scaled.device)
        qinp_out = qinp_out*normscale + normoffset

    # un-normalize
    qinp_out = qinp_out * rangeval

    # add outliers back
    if include_sparse:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers

    #shift by offset
    qinp_out = qinp_out + offset
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0) #TODO: debug (shouldn't be necessary)

    # leave first few in fp16
    # leave this here for now -> avoids any small perturbations from rescaling
    if first_few_fp16 > -1 or recent_fp16 > -1:
        _restore_fp16_token_window(
            qinp_out,
            orig,
            sink_tokens=first_few_fp16,
            recent_tokens=recent_fp16,
            seq_len=fp16_seq_len,
        )

    return qinp_out.float()

# simquant quantizer (calibration)
class SimQuant:
    def __init__(
                    self,
                    layer,
                    bits,
                    perchannel=True,
                    qchannel=0,
                    include_rope=False
                ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.perchannel = perchannel
        self.qchannel = qchannel
        self.bits = bits

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.nsamples = 0

        self.out = None

    def add_batch(self, inp, out):
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        tmp = out.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(out.shape) == 3:
                out = out.reshape((-1, self.rows))
        self.nsamples += tmp

        if self.out == None:
            self.out = out.clone()
        else:
            self.out = torch.cat((self.out, out.clone()), dim=0)

    def quantize(
        self,
        include_sparse=False,
        sparsity_threshold=0.999,
        nuq=False,
        fisher=False,
        norm=False,
        cap_outliers=False,
        first_few_fp16=-1,
    ):

        # for now, just update threshold here
        if include_sparse:
            t = 1-((1-sparsity_threshold)/2)
        else:
            t = 1 #use min-max quantization

        #TODO - if not using sparsity, use a different threshold for min-max quant?
        data = self.out.float().cpu().numpy()


        if self.perchannel and cap_outliers:
            #per-channel - remove tokenwise outliers and normalize range to [-1,1]
            data = torch.tensor(data)

            outlier_threshold_upper = torch.tensor(np.percentile(data, t*100, axis=self.qchannel)).unsqueeze(self.qchannel)
            outlier_threshold_lower = torch.tensor(np.percentile(data, (1-t)*100, axis=self.qchannel)).unsqueeze(self.qchannel)
            zero_point = (outlier_threshold_upper + outlier_threshold_lower) / 2
            distance = (outlier_threshold_upper - outlier_threshold_lower) / 2
            data2 = ((data - zero_point) / distance).abs()

            outlier_mask = torch.zeros_like(data2, dtype=torch.bool)
            hidden_dim = data.shape[-1]
            num_elems = math.ceil((1-t) * hidden_dim)
            upper_indices = torch.topk(data2, num_elems).indices
            lower_indices = torch.topk(data2, num_elems, largest=False).indices

            true_mask = torch.ones_like(upper_indices, dtype=torch.bool)
            outlier_mask.scatter_(-1, lower_indices, true_mask)
            outlier_mask.scatter_(-1, upper_indices, true_mask)

            if first_few_fp16 > -1 :
                # remove first few tokens
                for i in range(0,self.nsamples):
                    start = i*2048
                    end = i*2048 + first_few_fp16
                    outlier_mask[start:end,:] = True

            med = torch.median(data, dim=0).values.unsqueeze(0).repeat(32768,1)
            data_trimmed = data.clone()
            data_trimmed[outlier_mask] = med[outlier_mask] 

            outlier_threshold_upper = torch.max(data_trimmed, axis=self.qchannel).values
            outlier_threshold_lower = torch.min(data_trimmed, axis=self.qchannel).values

            # recomputing outlier mask here before doing k-means fitting
            zero_point = (outlier_threshold_upper + outlier_threshold_lower) / 2
            distance = (outlier_threshold_upper - outlier_threshold_lower) / 2
            zero_point = zero_point.unsqueeze(0)
            distance = distance.unsqueeze(0)
            data_shifted_normalized = ((data - zero_point) / distance).abs()
            outlier_mask = torch.logical_or((data_shifted_normalized > 1), (data_shifted_normalized < -1))

        if self.perchannel:
            #per-channel - remove tokenwise outliers and normalize range to [-1,1]
            outlier_threshold_upper = np.percentile(data, t*100, axis=self.qchannel)
            outlier_threshold_lower = np.percentile(data, (1-t)*100, axis=self.qchannel)
        else:
            #per-token - remove tokenwise outliers and normalize range to [-1,1]
            assert(False) # not currently supported

        # convert to torch
        data = torch.tensor(data)
        outlier_threshold_upper = torch.tensor(outlier_threshold_upper).unsqueeze(self.qchannel)
        outlier_threshold_lower = torch.tensor(outlier_threshold_lower).unsqueeze(self.qchannel)

        # range and offset
        rangeval = (outlier_threshold_upper - outlier_threshold_lower) / 2
        zeropoint = (outlier_threshold_upper + outlier_threshold_lower) / 2

        # shift by offset
        data_shifted = data - zeropoint

        # normalize by rangeval into [-1,1]
        data_shifted_normalized = data_shifted / rangeval

        #get outliers (need to mask out for kmeans)
        if not cap_outliers:
            outlier_mask = torch.logical_or((data_shifted_normalized > 1), (data_shifted_normalized < -1))

        # remove first few tokens
        if first_few_fp16 > -1:
            for i in range(0,self.nsamples):
                start = i*2048
                end = i*2048 + first_few_fp16
                outlier_mask[start:end,:] = True

        if nuq:
            centroids = []
            act_distn_np = data_shifted_normalized.flatten()
            n_cluster = 2 ** self.bits

            outlier_mask_unflattened = outlier_mask
            outlier_mask = outlier_mask.flatten()
            act_distn_np_without_outliers = act_distn_np[~outlier_mask]
            act_distn_np_without_outliers = act_distn_np_without_outliers.float().cpu().numpy().reshape(-1, 1)

            # load fisher info
            if fisher is not None:
                fisher_info = fisher.flatten()
                fisher_info_tmp_without_outliers = fisher_info[~outlier_mask]
                kmeans = KMeans(
                    n_clusters=n_cluster,
                    random_state=0,
                    n_init="auto",
                    max_iter=50,
                ).fit(
                    act_distn_np_without_outliers,
                    sample_weight=fisher_info_tmp_without_outliers,
                )
            else:
                kmeans = KMeans(
                    n_clusters=n_cluster,
                    random_state=0,
                    n_init="auto",
                    max_iter=50,
                ).fit(
                    act_distn_np_without_outliers
                )

            centroids.append(kmeans.cluster_centers_)

            #Q-Norm
            if norm:
                centroid = torch.tensor(centroids[0])
                aug = torch.tensor(data_shifted_normalized)
                not_outlier_mask_unflattened = ~outlier_mask_unflattened

                m1 = (aug*not_outlier_mask_unflattened).sum()/not_outlier_mask_unflattened.sum()
                not_outlier_mask_unqueeze = not_outlier_mask_unflattened.sum()
                stdev1 = torch.sqrt(torch.sum(((aug - m1)*not_outlier_mask_unflattened)**2) / not_outlier_mask_unqueeze)

                aug, freq = round_to_nearest_pole_sim(aug, centroid, return_freq=True)

                m2 = (aug*not_outlier_mask_unflattened).sum()/not_outlier_mask_unflattened.sum()
                stdev2 = torch.sqrt(torch.sum(((aug - m2)*not_outlier_mask_unflattened)**2) / not_outlier_mask_unqueeze)

                normscale = (stdev1 / stdev2)
                normoffset = (- m2) * (stdev1 / stdev2) + m1

                return outlier_threshold_upper, outlier_threshold_lower, centroids, normscale, normoffset
            if norm:
                result = (outlier_threshold_upper, outlier_threshold_lower, centroids, normscale, normoffset)
            else:
                result = (outlier_threshold_upper, outlier_threshold_lower, centroids)
            return result
        else:
            # not using NUQ
            return outlier_threshold_upper, outlier_threshold_lower

    def free(self):
        self.out = None
        self.qout = None
        torch.cuda.empty_cache()

# drop-in layer replacement class
class QuantLinearSim(nn.Module):
    def __init__(
                    self,
                    name,
                    bits,
                    quantizer,
                    infeatures,
                    outfeatures,
                    weight,
                    bias,
                    perchannel=True,
                    include_sparse=False,
                    sparsity_threshold=0.999,
                    dynamicquantization=False,
                    nuq=False,
                    nf_nuq=True,
                    norm=False,
                    first_few_fp16=-1,
                    recent_fp16=-1,
                    fp16_seq_len=2048,
                    rotation=None,
                    clip_ratio=-1,
                    clip_method="maxabs",
                    oscar_affine=False,
                    key_token_scale_target="none",
                    key_token_scale_eps=1e-6,
                    cap_outliers=-1,
                    clamp=False,
                ):

        super().__init__()
        if bits not in [2,3,4,5]:
            raise NotImplementedError("Only 3, 4, 5 bits are supported.")
        self.name = name
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        self.weight = weight.T.detach().cpu()
        if bias:
            self.bias = bias.detach().cpu()
        else:
            self.bias = None

        self.perchannel = perchannel
        self.dynamicquantization = dynamicquantization
        self.clamp = clamp

        if perchannel:
            self.qchannel = 0
        else: #per-token quant
            self.qchannel = -1

        self.ochannel = self.qchannel

        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.outlier_threshold_upper = torch.tensor(quantizer[0]).cuda().flatten().half()
        self.outlier_threshold_lower = torch.tensor(quantizer[1]).cuda().flatten().half()

        self.nuq = nuq
        self.nf_nuq = nf_nuq
        if self.nuq and not self.nf_nuq:
            self.lut = quantizer[2]
        else:
            self.lut = None

        if norm:
            self.normscale = quantizer[3]
            self.normoffset = quantizer[4]
            self.norm = True
        else:
            self.norm = False
            self.normscale = None
            self.normoffset = None

        self.cap_outliers = cap_outliers
        self.first_few_fp16 = first_few_fp16
        self.recent_fp16 = recent_fp16
        self.fp16_seq_len = fp16_seq_len
        self.rotation = rotation
        self.clip_ratio = clip_ratio
        self.clip_method = clip_method
        self.oscar_affine = oscar_affine
        self.key_token_scale_target = key_token_scale_target
        self.key_token_scale_eps = key_token_scale_eps

        # for normalfloat support - compute NF signposts
        if self.nf_nuq:
            dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            # get evenly spaced percentile values

            num_signposts_pos = (2 ** (self.bits - 1)) + 1 # for pos half
            num_signposts_neg = (2 ** (self.bits - 1)) # for neg half

            self.nf_signposts_negative = []
            self.nf_signposts_positive = []

            # from https://arxiv.org/pdf/2306.06965.pdf
            offsets = [0.5*(1/32 + 1/30), 1 - 0.5*(1/32 + 1/30)]
            list1 = [offsets[0]]
            spacing = (0.5 - offsets[0]) / (2 ** (self.bits - 1) - 1)

            add = offsets[0]
            for i in range(num_signposts_neg - 1):
                add += spacing
                list1.append(add)

            list2 = []
            spacing = (offsets[1] - 0.5) / (2 ** (self.bits - 1)) #1 extra space
            add = 0.5
            for i in range(num_signposts_pos - 1):
                list2.append(add)
                add += spacing
            list2.append(offsets[-1])

            # first do negative part [0->0.5]
            for i in range(num_signposts_neg):
                v1 = list1[i]
                val = dist.icdf(torch.tensor([v1])).data.numpy()
                self.nf_signposts_negative.append(torch.tensor(val).item())

            # next do positive part [0.5->1]
            for i in range(num_signposts_pos):
                v1 = list2[i]
                val = dist.icdf(torch.tensor([v1])).data.numpy()
                self.nf_signposts_positive.append(torch.tensor(val).item())

            signpost_neg_min = self.nf_signposts_negative[0]
            signpost_neg_max = self.nf_signposts_negative[-1]
            rangeval = abs(signpost_neg_min)-abs(signpost_neg_max)
            off = abs(signpost_neg_max)
            for s in range(len(self.nf_signposts_negative)):
                self.nf_signposts_negative[s] = (self.nf_signposts_negative[s] + off) / rangeval

            signpost_pos_min = self.nf_signposts_positive[0]
            signpost_pos_max = self.nf_signposts_positive[-1]
            rangeval = abs(signpost_pos_max)-abs(signpost_pos_min)
            off = abs(signpost_pos_min)

            for s in range(len(self.nf_signposts_positive)):
                self.nf_signposts_positive[s] = (self.nf_signposts_positive[s] - off) / rangeval

            del self.nf_signposts_positive[0]

            # delete last negative value and merge
            self.nf_signposts = self.nf_signposts_negative + self.nf_signposts_positive

            assert (len(self.nf_signposts) == (2 ** self.bits))

    #replacement forward pass
    def forward(self, x, other_mat=None):

        out_shape = x.shape[:-1] + (self.outfeatures, )
        x = x.reshape(-1,x.shape[-1])

        # copying weight to / from device during evaluation lets us evaluate
        # a large model with limitted memory usage

        self.weight = self.weight.to(x.device)
        if self.bias is not None:
            self.bias = self.bias.to(x.device)

        x = x.half() # for now cast to fp16 and back (quantization code assumes fp32)
        y = x @ self.weight
        y = y + self.bias if self.bias is not None else y
        y = y.float()
        y = _apply_head_rotation(y, self.rotation, transpose=False)
        effective_qchannel = -1 if self.oscar_affine else self.qchannel
        effective_ochannel = -1 if self.oscar_affine else self.ochannel
        y = _clip_activation(y, self.clip_ratio, qchannel=effective_qchannel, method=self.clip_method)
        y, key_token_scale = _apply_key_token_scaling(
            y,
            target=self.key_token_scale_target,
            eps=self.key_token_scale_eps,
            sink_tokens=self.first_few_fp16,
            recent_tokens=self.recent_fp16,
            seq_len=self.fp16_seq_len,
            name=self.name,
        )

        # if using dense-and-sparse quantization, detect outliers in output tensor.
        # OSCAR affine clips outliers instead of preserving them sparsely.
        if self.include_sparse and not self.oscar_affine:
            if self.dynamicquantization:
                outlier_mask = get_outliers_dynamic(
                    y,
                    channel=effective_ochannel,
                    thresh=self.sparsity_threshold,
                    first_few_fp16=self.first_few_fp16,
                    recent_fp16=self.recent_fp16,
                    fp16_seq_len=self.fp16_seq_len
                )
            else:
                self.outlier_threshold_upper = self.outlier_threshold_upper.to(y.device)
                self.outlier_threshold_lower = self.outlier_threshold_lower.to(y.device)
                outlier_mask = get_outliers(
                    y,
                    channel=effective_ochannel,
                    outlier_threshold_upper=self.outlier_threshold_upper,
                    outlier_threshold_lower=self.outlier_threshold_lower,
                    cap_outliers=self.cap_outliers,
                    first_few_fp16=self.first_few_fp16,
                    recent_fp16=self.recent_fp16,
                    fp16_seq_len=self.fp16_seq_len
                )
        else:
            outlier_mask = None


        # quantize output tensor
        if self.oscar_affine:
            y = quant_fn_oscar_affine(
                y,
                bits=self.bits,
                qchannel=effective_qchannel,
                first_few_fp16=self.first_few_fp16,
                recent_fp16=self.recent_fp16,
                fp16_seq_len=self.fp16_seq_len,
            )
        elif self.nuq:
            if self.nf_nuq:
                y = quant_fn_nf(
                    y,
                    bits=self.bits,
                    qchannel=self.qchannel,
                    maxval=self.outlier_threshold_upper,
                    minval=self.outlier_threshold_lower,
                    include_sparse=self.include_sparse,
                    outlier_mask=outlier_mask,
                    dynamicquantization=self.dynamicquantization,
                    nf_lut=self.nf_signposts
                )
            else:
                y = quant_fn_nuq_recon(
                    y,
                    bits=self.bits,
                    qchannel=self.qchannel,
                    maxval=self.outlier_threshold_upper,
                    minval=self.outlier_threshold_lower,
                    include_sparse=self.include_sparse,
                    outlier_mask=outlier_mask,
                    dynamicquantization=self.dynamicquantization,
                    lut=self.lut,
                    norm=self.norm,
                    normscale=self.normscale,
                    normoffset=self.normoffset,
                    first_few_fp16=self.first_few_fp16,
                    recent_fp16=self.recent_fp16,
                    fp16_seq_len=self.fp16_seq_len
                )

        else:
            # low-bit uniform simulated quant
            y = quant_fn_zp(
                y,
                bits=self.bits,
                qchannel=self.qchannel,
                maxval=self.outlier_threshold_upper,
                minval=self.outlier_threshold_lower,
                include_sparse=self.include_sparse,
                outlier_mask=outlier_mask,
                dynamicquantization=self.dynamicquantization,
                clamp=self.clamp
            )

        y = _invert_key_token_scaling(y, key_token_scale)
        y = _apply_head_rotation(y, self.rotation, transpose=True)

        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

        y = y.reshape(out_shape)


        y = y.half()
        return y

# update modules
def make_quant_sim(
                    module,
                    quantizers,
                    bits,
                    name='',
                    perchannel=True,
                    include_sparse=False,
                    sparsity_threshold=0.999,
                    dynamicquantization=False,
                    nuq=False,
                    nf_nuq=True,
                    norm=False,
                    cap_outliers=-1,
                    first_few_fp16=-1,
                    recent_fp16=-1,
                    fp16_seq_len=2048,
                    rotations=None,
                    clip_ratios=None,
                    clip_methods=None,
                    oscar_affine=False,
                    key_token_scale_targets=None,
                    key_token_scale_eps=1e-6,
                    clamp=False,
                  ):
    if isinstance(module, QuantLinearSim):
        return
    rotations = rotations or {}
    clip_ratios = clip_ratios or {}
    clip_methods = clip_methods or {}
    key_token_scale_targets = key_token_scale_targets or {}
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantizers.keys():
            delattr(module, attr)
            setattr(module, attr, QuantLinearSim(
                                                    name1,
                                                    bits,
                                                    quantizers[name1],
                                                    tmp.in_features,
                                                    tmp.out_features,
                                                    tmp.weight,
                                                    tmp.bias is not None,
                                                    perchannel=perchannel,
                                                    include_sparse=include_sparse,
                                                    sparsity_threshold=sparsity_threshold,
                                                    dynamicquantization=dynamicquantization,
                                                    nuq=nuq,
                                                    nf_nuq=nf_nuq,
                                                    norm=norm,
                                                    cap_outliers=cap_outliers,
                                                    first_few_fp16=first_few_fp16,
                                                    recent_fp16=recent_fp16,
                                                    fp16_seq_len=fp16_seq_len,
                                                    rotation=rotations.get(name1),
                                                    clip_ratio=clip_ratios.get(name1, -1),
                                                    clip_method=clip_methods.get(name1, "maxabs"),
                                                    oscar_affine=oscar_affine,
                                                    key_token_scale_target=key_token_scale_targets.get(name1, "none"),
                                                    key_token_scale_eps=key_token_scale_eps,
                                                    clamp=clamp,
                                                ))
        del tmp
    for name1, child in module.named_children():
        make_quant_sim(
                        child,
                        quantizers,
                        bits,
                        name + '.' + name1 if name != '' else name1,
                        perchannel=perchannel,
                        include_sparse=include_sparse,
                        sparsity_threshold=sparsity_threshold,
                        dynamicquantization=dynamicquantization,
                        nuq=nuq,
                        nf_nuq=nf_nuq,
                        norm=norm,
                        cap_outliers=cap_outliers,
                        first_few_fp16=first_few_fp16,
                        recent_fp16=recent_fp16,
                        fp16_seq_len=fp16_seq_len,
                        rotations=rotations,
                        clip_ratios=clip_ratios,
                        clip_methods=clip_methods,
                        oscar_affine=oscar_affine,
                        key_token_scale_targets=key_token_scale_targets,
                        key_token_scale_eps=key_token_scale_eps,
                        clamp=clamp
                      )
