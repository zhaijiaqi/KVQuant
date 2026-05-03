"""
Passkey Retrieval evaluation using simulated quantization.
Patches applied vs. original:
  - Removed deepspeed / evaluate / torch.distributed imports (not needed for single-GPU simquant)
  - math import added (needed for rope_scaling calculation)
  - config._flash_attn_2_enabled removed (use standard attention, no flash-attn required)
  - model.model.set_devices() guarded (only present in deployment transformers fork)
  - generate() called with use_cache=True and model moved to GPU correctly
  - Output dir created if not exists
"""

import random
import argparse
import re
import math
from pathlib import Path
import jsonlines

import torch
import transformers
import numpy as np
from transformers import LlamaTokenizer
from tqdm import tqdm

from transformers import LlamaForCausalLM
from transformers import LlamaConfig

from kvquant.modelutils import *
from kvquant.datautils import *
from kvquant.simquant_module_quantizer import *

from kvquant.model_parse import (
    parse_model,
    get_layers,
    get_embedding,
    get_norm,
)

gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_prompt(max_tokens=16384):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_total = (max_tokens - 32 - 26 - 11) // 25
    n_garbage_prefix = random.randint(0, n_garbage_total)
    n_garbage_suffix = n_garbage_total - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."  # 32 tokens
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."  # 25 tokens
    garbage_prefix = garbage * n_garbage_prefix
    garbage_suffix = garbage * n_garbage_suffix
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."  # 26 tokens
    final_question = "What is the pass key? The pass key is"  # 11 tokens
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    return "\n".join(lines), pass_key


def test_model(model, tokenizer, prompt_text, pass_key):
    model_input = tokenizer.encode(
        prompt_text, return_tensors="pt", max_length=100000, truncation=True
    ).to(gpu_device)

    with torch.no_grad():
        response = model.generate(model_input, num_return_sequences=1, max_new_tokens=10)
    response = tokenizer.batch_decode(
        response[:, model_input.shape[1]:], skip_special_tokens=True
    )[0]
    print(response)

    assert f"The pass key is {pass_key}" in prompt_text

    try:
        pred = int(re.search(r'\d+', response).group())
    except Exception:
        pred = response[:20]

    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxseqlen", type=int, default=32768)
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--path_to_ckp", type=str, default="/data/llama-7b")
    parser.add_argument("--path_to_output_dir", type=str, default="results/passkey")
    parser.add_argument("--simquant", action='store_true')
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--quantizer-path", type=str, default=None, help='quantizer path')
    parser.add_argument("--norm", action='store_true')
    args = parser.parse_args()

    model_name_or_path = args.path_to_ckp
    scaled_max_position_embeddings = args.maxseqlen

    config = LlamaConfig.from_pretrained(model_name_or_path)
    context_size = args.maxseqlen
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Standard attention (no flash-attn required)
    config.use_cache = True

    try:
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_flash_attention_2=True,
        )
    except (ImportError, ValueError, TypeError):
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)

    print('load tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    import pickle
    if args.simquant:
        with open(args.quantizer_path, 'rb') as handle:
            quantizers = pickle.load(handle)

        # replace layers
        perchannelquant = {}
        pertokenquant = {}

        for k in quantizers.keys():
            if "k_proj" in k:
                perchannelquant[k] = quantizers[k]
            if "v_proj" in k:
                pertokenquant[k] = quantizers[k]

        # per-channel quant (K)
        make_quant_sim(
            model,
            perchannelquant,
            args.abits,
            perchannel=True,
            include_sparse=True,
            sparsity_threshold=0.99,
            nuq=True,
            nf_nuq=False,
            norm=args.norm,
        )

        # per-token quant (V)
        make_quant_sim(
            model,
            pertokenquant,
            args.abits,
            perchannel=False,
            dynamicquantization=True,
            include_sparse=True,
            sparsity_threshold=0.99,
            nuq=True,
            nf_nuq=False,
            norm=args.norm,
        )

    # Move model to GPU (single-GPU path)
    model = model.to(gpu_device)
    model.eval()

    # Guard: set_devices only exists in deployment transformers fork
    if hasattr(model, 'model') and hasattr(model.model, 'set_devices'):
        model.model.set_devices()

    # Create output directory
    root_dir = Path(__file__).parent.parent
    out_dir = root_dir / args.path_to_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path_to_output_fn = (out_dir / f"{args.model_name}.jsonl").as_posix()

    result_list = list()
    length_list = [2048, 4096, 8192, 16384, 32768]
    for context_size in length_list:
        if context_size == scaled_max_position_embeddings:
            context_size -= 100
        print(f"context_size: {context_size}")
        correct_cnt = 0
        result_dict = {
            "scaled_length": scaled_max_position_embeddings,
            "context_size": context_size,
        }
        iter_nums = 50
        for i in tqdm(range(iter_nums)):
            prompt_text, pass_key = generate_prompt(context_size)
            pred = test_model(model, tokenizer, prompt_text, pass_key)
            result = "Pass!" if pred == pass_key else "Fail!"
            correct_cnt += 1 if pred == pass_key else 0
            case_report = f"pred: {pred}, ans: {pass_key}, result: {result}"
            result_dict[f"case{i}"] = case_report
        print(f"correct_rate: {correct_cnt / iter_nums}")
        result_dict["correct_rate"] = correct_cnt / iter_nums
        result_list.append(result_dict)

    with jsonlines.open(path_to_output_fn, "w") as writer:
        writer.write_all(result_list)
    print(f"Results written to {path_to_output_fn}")


if __name__ == "__main__":
    main()
