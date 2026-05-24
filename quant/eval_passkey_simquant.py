"""
Passkey Retrieval evaluation using simulated quantization.

Patches applied vs. original:
  - Removed deepspeed / evaluate / torch.distributed imports (not needed for single-GPU simquant)
  - Uses AutoConfig / AutoModelForCausalLM / AutoTokenizer for better compatibility with 32K variants
  - Adds smoke-test controls via --context-lengths and --num-samples
  - Falls back cleanly when flash-attn is unavailable
  - Guards model.model.set_devices() for non-KVQuant model variants
  - Moves the model to GPU explicitly and writes results to an ensured output directory
"""

import argparse
import gc
import math
import random
import re
from pathlib import Path

import jsonlines
import torch
from tqdm import tqdm

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
DEFAULT_CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768]


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


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_tokenizer(model_name_or_path, use_fast=True):
    attempts = [
        {"use_fast": use_fast, "trust_remote_code": True},
        {"use_fast": use_fast},
    ]
    if not use_fast:
        attempts.extend(
            [
                {"use_fast": False, "trust_remote_code": True, "legacy": True},
                {"use_fast": False, "legacy": True},
            ]
        )
    else:
        attempts.extend(
            [
                {"use_fast": False, "trust_remote_code": True, "legacy": True},
                {"use_fast": False, "legacy": True},
            ]
        )

    last_error = None
    for kwargs in attempts:
        try:
            return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        except (ImportError, TypeError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to load tokenizer for {model_name_or_path}")


def load_model_and_tokenizer(model_name_or_path, maxseqlen):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    config.use_cache = True

    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_flash_attention_2=True,
        )
    except (ImportError, ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)

    tokenizer = _load_tokenizer(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = maxseqlen

    model = model.to(gpu_device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model, "model") and hasattr(model.model, "set_devices"):
        model.model.set_devices()

    return model, tokenizer


def test_model(model, tokenizer, prompt_text, pass_key, max_new_tokens, use_cache=True):
    model_input = tokenizer.encode(
        prompt_text,
        return_tensors="pt",
        max_length=100000,
        truncation=True,
    ).to(gpu_device)

    generated_ids = None
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                model_input,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                do_sample=False,
            )
        response = tokenizer.batch_decode(
            generated_ids[:, model_input.shape[1]:],
            skip_special_tokens=True,
        )[0]
    finally:
        del model_input
        if generated_ids is not None:
            del generated_ids
        cleanup_cuda()

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
    parser.add_argument("--context-lengths", type=int, nargs="+", default=DEFAULT_CONTEXT_LENGTHS)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--context-headroom", type=int, default=100)
    parser.add_argument("--disable-cache-above", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name_or_path = args.path_to_ckp
    scaled_max_position_embeddings = args.maxseqlen
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, scaled_max_position_embeddings)

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

    root_dir = Path(__file__).parent.parent
    out_dir = root_dir / args.path_to_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path_to_output_fn = out_dir / f"{args.model_name}.jsonl"
    completed_contexts = set()
    if args.resume and path_to_output_fn.exists():
        existing_rows = []
        with jsonlines.open(path_to_output_fn.as_posix(), "r") as reader:
            for row in reader:
                if row.get("error") is None and row.get("correct_rate") is not None:
                    existing_rows.append(row)
                    completed_contexts.add(row.get("requested_context_size"))
        with jsonlines.open(path_to_output_fn.as_posix(), "w") as writer:
            for row in existing_rows:
                writer.write(row)
    elif path_to_output_fn.exists():
        path_to_output_fn.unlink()

    result_list = []
    for requested_context_size in args.context_lengths:
        if requested_context_size in completed_contexts:
            print(f"Skipping completed context_size {requested_context_size}")
            continue
        context_size = requested_context_size
        if context_size >= scaled_max_position_embeddings:
            context_size = scaled_max_position_embeddings - args.max_new_tokens - args.context_headroom
        use_cache_for_context = not (
            args.disable_cache_above is not None and requested_context_size >= args.disable_cache_above
        )
        print(f"context_size: {context_size}, use_cache: {use_cache_for_context}")
        correct_cnt = 0
        result_dict = {
            "scaled_length": scaled_max_position_embeddings,
            "requested_context_size": requested_context_size,
            "context_size": context_size,
            "num_samples": args.num_samples,
            "seed": args.seed,
        }
        for i in tqdm(range(args.num_samples)):
            prompt_text, pass_key = generate_prompt(context_size)
            try:
                pred = test_model(
                    model,
                    tokenizer,
                    prompt_text,
                    pass_key,
                    args.max_new_tokens,
                    use_cache=use_cache_for_context,
                )
            except torch.OutOfMemoryError:
                cleanup_cuda()
                result_dict["error"] = "torch.OutOfMemoryError"
                result_dict["failed_case"] = i
                result_dict["correct_rate"] = correct_cnt / i if i > 0 else None
                with jsonlines.open(path_to_output_fn.as_posix(), "a") as writer:
                    writer.write(result_dict)
                print(f"OOM at context_size {context_size}, case {i}. Partial results written to {path_to_output_fn}.")
                raise
            result = "Pass!" if pred == pass_key else "Fail!"
            correct_cnt += 1 if pred == pass_key else 0
            case_report = f"pred: {pred}, ans: {pass_key}, result: {result}"
            result_dict[f"case{i}"] = case_report
        result_dict["correct_rate"] = correct_cnt / args.num_samples
        print(f"correct_rate: {result_dict['correct_rate']}")
        result_list.append(result_dict)
        with jsonlines.open(path_to_output_fn.as_posix(), "a") as writer:
            writer.write(result_dict)
        cleanup_cuda()

    print(f"Results written to {path_to_output_fn}")

if __name__ == "__main__":
    main()
