import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans

import torch
from torch.distributions import Normal

def round_to_nearest_pole_sim(w, poles, return_freq=False, chunk_size=2_000_000):
    """
    w: weight/act values (1d vector)
    poles: tuple of values

    Round the numbers in w to the nearest value in poles.
    """
    if not torch.is_tensor(w):
        w = torch.as_tensor(w)

    poles_t = torch.as_tensor(poles, device=w.device, dtype=w.dtype).flatten()
    if poles_t.numel() == 0:
        raise ValueError("poles must contain at least one value")

    flat_w = w.reshape(-1)
    flat_out = torch.empty_like(flat_w)
    freq = torch.zeros(poles_t.numel(), device=flat_w.device, dtype=torch.long) if return_freq else None

    if chunk_size <= 0:
        chunk_size = flat_w.numel()

    for start in range(0, flat_w.numel(), chunk_size):
        end = min(start + chunk_size, flat_w.numel())
        chunk = flat_w[start:end]

        best_idx = torch.zeros(chunk.shape, device=chunk.device, dtype=torch.long)
        best_diff = (chunk - poles_t[0]).abs()

        for i in range(1, poles_t.numel()):
            diff = (chunk - poles_t[i]).abs()
            better = diff < best_diff
            best_diff = torch.where(better, diff, best_diff)
            best_idx = torch.where(better, torch.full_like(best_idx, i), best_idx)

        flat_out[start:end] = poles_t.index_select(0, best_idx)
        if return_freq:
            freq += torch.bincount(best_idx, minlength=poles_t.numel())

    aug = flat_out.reshape(w.shape)
    if return_freq:
        return aug, [count for count in freq]
    return aug


def _tile_flatten(inp, tile_size):
    rows, cols = inp.shape
    row_pad = (tile_size - rows % tile_size) % tile_size
    col_pad = (tile_size - cols % tile_size) % tile_size

    if row_pad or col_pad:
        padded = F.pad(inp, (0, col_pad, 0, row_pad))
    else:
        padded = inp

    padded_rows, padded_cols = padded.shape
    tiles = padded.reshape(
        padded_rows // tile_size,
        tile_size,
        padded_cols // tile_size,
        tile_size,
    ).permute(0, 2, 1, 3).contiguous()

    return tiles.reshape(tiles.shape[0], tiles.shape[1], -1), {
        "tile_rows": tiles.shape[0],
        "tile_cols": tiles.shape[1],
        "tile_size": tile_size,
        "orig_rows": rows,
        "orig_cols": cols,
    }


def _tile_restore(flat, meta):
    tiles = flat.reshape(
        meta["tile_rows"],
        meta["tile_cols"],
        meta["tile_size"],
        meta["tile_size"],
    )
    out = tiles.permute(0, 2, 1, 3).contiguous().reshape(
        meta["tile_rows"] * meta["tile_size"],
        meta["tile_cols"] * meta["tile_size"],
    )
    return out[: meta["orig_rows"], : meta["orig_cols"]]


def _build_first_few_mask(flat_shape, meta, first_few_fp16, nsamples, device=None):
    if first_few_fp16 < 0 or nsamples <= 0:
        return None

    sample_seqlen = meta["orig_rows"] // nsamples
    if sample_seqlen <= 0:
        return None

    mask_2d = torch.zeros((meta["orig_rows"], meta["orig_cols"]), dtype=torch.bool, device=device)
    for sample_idx in range(nsamples):
        start = sample_idx * sample_seqlen
        end = min(start + first_few_fp16, (sample_idx + 1) * sample_seqlen)
        mask_2d[start:end, :] = True

    mask_flat, _ = _tile_flatten(mask_2d.float(), meta["tile_size"])
    return mask_flat.bool().reshape(flat_shape)


def _tile_dynamic_stats(flat, include_sparse, sparsity_threshold, first_few_mask=None):
    if include_sparse:
        t = 1 - ((1 - sparsity_threshold) / 2)
        upper = torch.quantile(flat, t, dim=-1)
        lower = torch.quantile(flat, 1 - t, dim=-1)
        outlier_mask = torch.logical_or(
            flat >= upper.unsqueeze(-1),
            flat <= lower.unsqueeze(-1),
        )
        if first_few_mask is not None:
            outlier_mask = torch.logical_or(outlier_mask, first_few_mask)
        median = torch.median(flat, dim=-1).values
        tmp = torch.where(outlier_mask, median.unsqueeze(-1), flat)
        maxval = tmp.max(dim=-1).values
        minval = tmp.min(dim=-1).values
    else:
        outlier_mask = first_few_mask
        maxval = flat.max(dim=-1).values
        minval = flat.min(dim=-1).values

    rangeval = (maxval - minval) / 2
    rangeval = torch.where(rangeval == 0, torch.ones_like(rangeval), rangeval)
    offset = (maxval + minval) / 2
    return offset, rangeval, outlier_mask

def get_outliers(
    w,
    channel=-1,
    outlier_threshold_upper=-1,
    outlier_threshold_lower=-1,
    cap_outliers=-1,
    first_few_fp16=-1
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    outlier_threshold_upper: upper outlier thresholds
    outlier_threshold_lower: lower outlier thresholds
    first_few_fp16: number of initial tokens to keep in fp16

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

    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16,:] = True

    return outlier_mask

def get_outliers_dynamic(
    w,
    channel=-1,
    thresh=0.999,
    first_few_fp16=-1
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    thresh: percentile for outlier threshold computation
    first_few_fp16: number of initial tokens to keep in fp16

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

    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16,:] = True

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
    first_few_fp16=-1
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

    Performs simulated NUQ quantization
    """

    if first_few_fp16 > -1:
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
    if first_few_fp16 > -1:
        qinp_out[:first_few_fp16,:] = orig[:first_few_fp16,:]

    return qinp_out.float()


def quant_fn_nuq_recon_tile(
    inp,
    bits=8,
    tile_size=32,
    dynamicquantization=True,
    include_sparse=False,
    sparsity_threshold=0.999,
    lut=None,
    norm=False,
    normscale=None,
    normoffset=None,
    first_few_fp16=-1,
):
    if lut is None:
        raise NotImplementedError("Tile quantization currently requires a LUT-backed NUQ path.")

    orig = inp if first_few_fp16 > -1 else None
    flat, meta = _tile_flatten(inp, tile_size)
    first_few_mask = _build_first_few_mask(
        flat.shape,
        meta,
        first_few_fp16,
        nsamples=1,
        device=flat.device,
    )
    offset, rangeval, outlier_mask = _tile_dynamic_stats(
        flat,
        include_sparse=include_sparse,
        sparsity_threshold=sparsity_threshold,
        first_few_mask=first_few_mask,
    )

    shifted = flat - offset.unsqueeze(-1)
    if include_sparse and outlier_mask is not None:
        outliers = torch.where(outlier_mask, shifted, torch.zeros_like(shifted))
        shifted = shifted - outliers
    else:
        outliers = None

    normalized = shifted / rangeval.unsqueeze(-1)
    lut_cuda = torch.as_tensor(lut[0], device=normalized.device, dtype=normalized.dtype)
    quantized = round_to_nearest_pole_sim(normalized.reshape(-1), lut_cuda).reshape(normalized.shape).float()

    if norm:
        normscale = normscale.to(normalized.device)
        normoffset = normoffset.to(normalized.device)
        quantized = quantized * normscale + normoffset

    restored = quantized * rangeval.unsqueeze(-1)
    if include_sparse and outliers is not None:
        restored[outlier_mask] = 0
        restored = restored + outliers
    restored = restored + offset.unsqueeze(-1)
    restored = torch.nan_to_num(restored, nan=0.0, posinf=0.0, neginf=0.0)
    out = _tile_restore(restored, meta).float()

    if orig is not None:
        out[:first_few_fp16, :] = orig[:first_few_fp16, :]

    return out

# simquant quantizer (calibration)
class SimQuant:
    def __init__(
                    self,
                    layer,
                    bits,
                    perchannel=True,
                    qchannel=0,
                    include_rope=False,
                    tile_size=0
                ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.perchannel = perchannel
        self.qchannel = qchannel
        self.bits = bits
        self.tile_size = tile_size

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
        first_few_fp16=-1
    ):

        if self.tile_size > 0:
            data = self.out.float()
            flat, meta = _tile_flatten(data, self.tile_size)
            first_few_mask = _build_first_few_mask(
                flat.shape,
                meta,
                first_few_fp16,
                self.nsamples,
                device=flat.device,
            )
            offset, rangeval, outlier_mask = _tile_dynamic_stats(
                flat,
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                first_few_mask=first_few_mask,
            )

            shifted = flat - offset.unsqueeze(-1)
            normalized = shifted / rangeval.unsqueeze(-1)
            if outlier_mask is None:
                outlier_mask = torch.zeros_like(normalized, dtype=torch.bool)

            act_distn_np_without_outliers = normalized[~outlier_mask].float().cpu().numpy().reshape(-1, 1)

            if fisher is not None:
                fisher_data = fisher.float()
                if fisher_data.dim() > 2:
                    fisher_data = fisher_data.reshape(-1, fisher_data.shape[-1])
                fisher_flat, _ = _tile_flatten(fisher_data, self.tile_size)
                fisher_mask = outlier_mask
                if fisher_mask.device != fisher_flat.device:
                    fisher_mask = fisher_mask.to(fisher_flat.device)
                fisher_info_tmp_without_outliers = fisher_flat[~fisher_mask].float().cpu().numpy()
                kmeans = KMeans(
                    n_clusters=2 ** self.bits,
                    random_state=0,
                    n_init="auto",
                    max_iter=50,
                ).fit(
                    act_distn_np_without_outliers,
                    sample_weight=fisher_info_tmp_without_outliers,
                )
            else:
                kmeans = KMeans(
                    n_clusters=2 ** self.bits,
                    random_state=0,
                    n_init="auto",
                    max_iter=50,
                ).fit(act_distn_np_without_outliers)

            centroids = [kmeans.cluster_centers_]

            upper = offset + rangeval
            lower = offset - rangeval

            if norm:
                centroid = torch.as_tensor(centroids[0], device=normalized.device, dtype=normalized.dtype)
                aug = normalized
                not_outlier_mask = ~outlier_mask
                m1 = (aug * not_outlier_mask).sum() / not_outlier_mask.sum()
                denom = not_outlier_mask.sum()
                stdev1 = torch.sqrt(torch.sum(((aug - m1) * not_outlier_mask) ** 2) / denom)

                aug, freq = round_to_nearest_pole_sim(aug, centroid, return_freq=True)

                m2 = (aug * not_outlier_mask).sum() / not_outlier_mask.sum()
                stdev2 = torch.sqrt(torch.sum(((aug - m2) * not_outlier_mask) ** 2) / denom)

                normscale = stdev1 / stdev2
                normoffset = (-m2) * (stdev1 / stdev2) + m1
                return upper, lower, centroids, normscale, normoffset

            return upper, lower, centroids

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
            else:
                return outlier_threshold_upper, outlier_threshold_lower, centroids
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
                    cap_outliers=-1,
                    clamp=False,
                    tile_size=0
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
        self.tile_size = tile_size

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
        if self.tile_size > 0 and self.nf_nuq:
            raise NotImplementedError("Tile quantization for Value activations does not support NormalFloat yet.")
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

        # quantize output tensor
        if self.tile_size > 0:
            if self.nuq:
                y = quant_fn_nuq_recon_tile(
                    y,
                    bits=self.bits,
                    tile_size=self.tile_size,
                    dynamicquantization=self.dynamicquantization,
                    include_sparse=self.include_sparse,
                    sparsity_threshold=self.sparsity_threshold,
                    lut=self.lut,
                    norm=self.norm,
                    normscale=self.normscale,
                    normoffset=self.normoffset,
                    first_few_fp16=self.first_few_fp16,
                )
            else:
                raise NotImplementedError("Tile quantization is only implemented for NUQ evaluation.")
        else:
            # if using dense-and-sparse quantization, detect outliers in output tensor
            if self.include_sparse:
                if self.dynamicquantization:
                    outlier_mask = get_outliers_dynamic(
                        y,
                        channel=self.ochannel,
                        thresh=self.sparsity_threshold,
                        first_few_fp16=self.first_few_fp16
                    )
                else:
                    self.outlier_threshold_upper = self.outlier_threshold_upper.to(y.device)
                    self.outlier_threshold_lower = self.outlier_threshold_lower.to(y.device)
                    outlier_mask = get_outliers(
                        y,
                        channel=self.ochannel,
                        outlier_threshold_upper=self.outlier_threshold_upper,
                        outlier_threshold_lower=self.outlier_threshold_lower,
                        cap_outliers=self.cap_outliers,
                        first_few_fp16=self.first_few_fp16
                    )
            else:
                outlier_mask = None

            if self.nuq:
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
                        first_few_fp16=self.first_few_fp16
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
                    clamp=False,
                    tile_size=0
                  ):
    if isinstance(module, QuantLinearSim):
        return
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
                                                    clamp=clamp,
                                                    tile_size=tile_size
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
                        clamp=clamp,
                        tile_size=tile_size
                      )
