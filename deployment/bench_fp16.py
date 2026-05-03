"""
fp16 baseline latency benchmark (token-by-token, standard transformers).
Measures the same metric as deployment/llama.py benchmark() but for fp16,
using standard past_key_values instead of the KVQuant quantized KV cache.
"""
import time, argparse, numpy as np
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvquant.datautils import get_loaders

DEV = torch.device("cuda:0")

def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(DEV)
    torch.cuda.synchronize()

    loss = nn.CrossEntropyLoss()
    tot = 0.0

    def sync():
        torch.cuda.synchronize()

    print("Benchmarking ...")
    times = []
    past_key_values = None
    max_memory = 0
    with torch.no_grad():
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i:i+1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            past_key_values = out.past_key_values
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i+1)].to(DEV)).float()
            del out
        sync()
    print("Median:", np.median(times))
    if check:
        print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
        print("max memory(MiB):", max_memory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dataset", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=128)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seqlen", type=int, default=2048)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.half, device_map="cpu"
    )
    model.seqlen = args.seqlen
    model.eval()
    model = model.to(DEV)

    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        model=args.model, seqlen=model.seqlen
    )

    input_ids = next(iter(dataloader))[0][:, :args.benchmark]
    benchmark(model, input_ids, check=args.check)
