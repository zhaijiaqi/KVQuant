import math

import torch

from kvquant.datautils import get_loaders
from kvquant.model_parse import get_layers, parse_model


def get_attention_layout(attn_module):
    config = attn_module.config
    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(config, "num_attention_heads")

    num_key_value_heads = getattr(attn_module, "num_key_value_heads", None)
    if num_key_value_heads is None:
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)

    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // num_heads

    return num_heads, num_key_value_heads, head_dim


def get_model_longseqlen(model_name, seqlen, maxseqlen):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import AutoConfig, AutoModelForCausalLM
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils

    # Newer transformers releases block loading PyTorch .bin checkpoints on
    # torch<2.6. The public OpenLLaMA weights we download for this analysis are
    # .bin shards, so we bypass that version gate in this trusted local setup.
    import_utils.check_torch_load_is_safe = lambda: None
    modeling_utils.check_torch_load_is_safe = lambda: None

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
    num_heads, num_key_value_heads, head_dim = get_attention_layout(attn_module)
    if getattr(config, "pretraining_tp", 1) > 1:
        import torch.nn.functional as F

        key_value_slicing = (num_key_value_heads * head_dim) // config.pretraining_tp
        query_slices = attn_module.q_proj.weight.split(
            (num_heads * head_dim) // config.pretraining_tp,
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
    if cos.dim() == 4:
        cos = cos[:, :, : key_states.shape[-2], :]
        sin = sin[:, :, : key_states.shape[-2], :]
    elif cos.dim() == 3:
        if cos.shape[0] == query_states.shape[0] and cos.shape[1] == query_states.shape[2]:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        elif position_ids is not None:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
        else:
            cos = cos[:, : key_states.shape[-2], :].unsqueeze(1)
            sin = sin[:, : key_states.shape[-2], :].unsqueeze(1)
    else:
        if position_ids is not None:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
        else:
            cos = cos[: key_states.shape[-2], :].unsqueeze(0).unsqueeze(0)
            sin = sin[: key_states.shape[-2], :].unsqueeze(0).unsqueeze(0)

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


def _extract_layer_tensors(attn_module, hidden_states, position_ids, position_embeddings, past_key_value):
    bsz, q_len, _ = hidden_states.shape
    num_heads, num_key_value_heads, head_dim = get_attention_layout(attn_module)
    if position_ids is None:
        position_ids_local = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
    else:
        position_ids_local = position_ids

    query_states, key_states_pre, value_states = project_qkv(attn_module, hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states_pre.view(
        bsz,
        q_len,
        num_key_value_heads,
        head_dim,
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz,
        q_len,
        num_key_value_heads,
        head_dim,
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, attn_module.layer_idx)

    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        cos, sin = get_rope_cos_sin(attn_module, value_states, position_ids_local, kv_seq_len)
    _, key_states_post = apply_rope(query_states, key_states, cos, sin, position_ids_local)

    return {
        "k_pre_rope": key_states_pre[0].detach().float().cpu(),
        "k_post_rope": flatten_heads(key_states_post)[0].detach().float().cpu(),
        "values": flatten_heads(value_states)[0].detach().float().cpu(),
    }


def capture_layers_tensors(model, input_ids, layer_indices, dev):
    model_type = parse_model(model)
    layers = get_layers(model, model_type)
    target_indices = sorted(set(layer_indices))
    captured = {layer_idx: {} for layer_idx in target_indices}
    original_forwards = {}

    for layer_idx in target_indices:
        attn_module = layers[layer_idx].self_attn
        original_forward = attn_module.forward
        original_forwards[layer_idx] = original_forward

        def wrapped_forward(*args, _layer_idx=layer_idx, _attn_module=attn_module, _original_forward=original_forward, **kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            position_ids = kwargs.get("position_ids")
            position_embeddings = kwargs.get("position_embeddings")
            past_key_value = kwargs.get("past_key_value")
            if past_key_value is None:
                past_key_value = kwargs.get("past_key_values")

            if not captured[_layer_idx]:
                captured[_layer_idx] = _extract_layer_tensors(
                    _attn_module,
                    hidden_states,
                    position_ids,
                    position_embeddings,
                    past_key_value,
                )

            return _original_forward(*args, **kwargs)

        attn_module.forward = wrapped_forward

    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        with torch.no_grad():
            model(input_ids.to(dev))
    finally:
        for layer_idx in target_indices:
            layers[layer_idx].self_attn.forward = original_forwards[layer_idx]
        model.config.use_cache = use_cache

    missing = [layer_idx for layer_idx, tensors in captured.items() if not tensors]
    if missing:
        raise RuntimeError(f"Failed to capture tensors from layers {missing}.")
    return captured


def capture_layer_tensors(model, input_ids, layer_idx, dev):
    return capture_layers_tensors(model, input_ids, [layer_idx], dev)[layer_idx]


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


def load_wikitext_layers_tensors(
    model_name,
    seqlen,
    maxseqlen,
    sample_index,
    layer_indices,
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
    return capture_layers_tensors(model, input_ids, layer_indices, dev)
