import math

import torch

from kvquant.datautils import get_loaders
from kvquant.model_parse import get_layers, parse_model


def get_model_longseqlen(model_name, seqlen, maxseqlen):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    context_size = maxseqlen
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
    )
    model.seqlen = seqlen
    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)
    return model


def project_qkv(attn_module, hidden_states):
    config = attn_module.config
    if getattr(config, "pretraining_tp", 1) > 1:
        import torch.nn.functional as F

        key_value_slicing = (attn_module.num_key_value_heads * attn_module.head_dim) // config.pretraining_tp
        query_slices = attn_module.q_proj.weight.split(
            (attn_module.num_heads * attn_module.head_dim) // config.pretraining_tp,
            dim=0,
        )
        key_slices = attn_module.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = attn_module.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(config.pretraining_tp)]
        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(config.pretraining_tp)]
        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(config.pretraining_tp)]

        return (
            torch.cat(query_states, dim=-1),
            torch.cat(key_states, dim=-1),
            torch.cat(value_states, dim=-1),
        )

    return (
        attn_module.q_proj(hidden_states),
        attn_module.k_proj(hidden_states),
        attn_module.v_proj(hidden_states),
    )


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(query_states, key_states, cos, sin, position_ids):
    if position_ids is not None:
        cos = cos.squeeze(0).squeeze(0)
        sin = sin.squeeze(0).squeeze(0)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos[:, :, : key_states.shape[-2], :]
        sin = sin[:, :, : key_states.shape[-2], :]

    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    return query_states, key_states


def get_rope_cos_sin(attn_module, value_states, position_ids, kv_seq_len):
    rotary_emb = attn_module.rotary_emb
    calls = (
        lambda: rotary_emb(value_states, seq_len=kv_seq_len),
        lambda: rotary_emb(value_states, position_ids),
        lambda: rotary_emb(value_states, position_ids=position_ids),
        lambda: rotary_emb(value_states, seq_len=kv_seq_len, position_ids=position_ids),
    )
    for call in calls:
        try:
            out = call()
            if isinstance(out, tuple) and len(out) == 2:
                return out
        except TypeError:
            continue
    raise RuntimeError("Unable to query RoPE embeddings from this transformers version.")


def flatten_heads(tensor):
    return tensor.transpose(1, 2).contiguous().reshape(tensor.shape[0], tensor.shape[2], -1)


def reduce_blocks(tensor_2d, token_block, channel_block, reduction):
    seq_len, hidden_size = tensor_2d.shape
    seq_trim = (seq_len // token_block) * token_block
    hidden_trim = (hidden_size // channel_block) * channel_block
    trimmed = tensor_2d[:seq_trim, :hidden_trim]

    blocks = trimmed.view(
        seq_trim // token_block,
        token_block,
        hidden_trim // channel_block,
        channel_block,
    ).permute(0, 2, 1, 3)

    abs_blocks = blocks.abs().float()
    if reduction == "max_abs":
        reduced = abs_blocks.amax(dim=(-1, -2))
    elif reduction == "mean_abs":
        reduced = abs_blocks.mean(dim=(-1, -2))
    elif reduction == "p99_abs":
        reduced = torch.quantile(abs_blocks.reshape(*abs_blocks.shape[:2], -1), 0.99, dim=-1)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    return reduced.cpu(), seq_trim, hidden_trim


def build_sample(testenc, seqlen, sample_index):
    start = sample_index * seqlen
    end = start + seqlen
    total = testenc.input_ids.shape[1]
    if end > total:
        raise ValueError(
            f"sample_index={sample_index} with seqlen={seqlen} exceeds tokenized test set length {total}."
        )
    return testenc.input_ids[:, start:end]


def capture_layer_tensors(model, input_ids, layer_idx, dev):
    model_type = parse_model(model)
    layer = get_layers(model, model_type)[layer_idx]
    attn_module = layer.self_attn

    captured = {}
    original_forward = attn_module.forward

    def wrapped_forward(*args, **kwargs):
        hidden_states = args[0] if args else kwargs["hidden_states"]
        position_ids = kwargs.get("position_ids")
        past_key_value = kwargs.get("past_key_value")

        if not captured:
            bsz, q_len, _ = hidden_states.shape
            if position_ids is None:
                position_ids_local = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
            else:
                position_ids_local = position_ids

            query_states, key_states_pre, value_states = project_qkv(attn_module, hidden_states)

            query_states = query_states.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
            key_states = key_states_pre.view(
                bsz,
                q_len,
                attn_module.num_key_value_heads,
                attn_module.head_dim,
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz,
                q_len,
                attn_module.num_key_value_heads,
                attn_module.head_dim,
            ).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, attn_module.layer_idx)

            cos, sin = get_rope_cos_sin(attn_module, value_states, position_ids_local, kv_seq_len)
            _, key_states_post = apply_rope(query_states, key_states, cos, sin, position_ids_local)

            captured["k_pre_rope"] = key_states_pre[0].detach().float().cpu()
            captured["k_post_rope"] = flatten_heads(key_states_post)[0].detach().float().cpu()
            captured["values"] = flatten_heads(value_states)[0].detach().float().cpu()

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

    if not captured:
        raise RuntimeError(f"Failed to capture tensors from layer {layer_idx}.")
    return captured


def load_wikitext_layer_tensors(
    model_name,
    seqlen,
    maxseqlen,
    sample_index,
    layer_idx,
    device,
):
    dev = torch.device(device)
    model = get_model_longseqlen(model_name, seqlen, maxseqlen)
    model = model.half()
    model.eval()
    model.to(dev)

    _, testenc = get_loaders(
        "wikitext2",
        nsamples=1,
        seed=0,
        model=model_name,
        seqlen=seqlen,
    )
    input_ids = build_sample(testenc, seqlen, sample_index)
    return capture_layer_tensors(model, input_ids, layer_idx, dev)
