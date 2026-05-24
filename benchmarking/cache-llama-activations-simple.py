"""
cache-llama-activations-simple.py
Simpler activation caching: uses raw forward hooks on k_proj/v_proj
without make_quant_sim, avoiding OOM on large seqlen.

Usage (run from benchmarking/):
  python cache-llama-activations-simple.py <model_path> \
      --seqlen 16384 --output-path activations-seqlen16384.pickle
"""
import argparse, pickle, sys
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '.')
from kvquant.datautils import get_loaders
from kvquant.model_parse import parse_model, get_layers, get_embedding, get_norm

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--seqlen', type=int, default=2048)
parser.add_argument('--maxseqlen', type=int, default=None)
parser.add_argument('--nsamples', type=int, default=1)
parser.add_argument('--output-path', type=str, required=True)
args = parser.parse_args()

if args.maxseqlen is None:
    args.maxseqlen = args.seqlen

DEV = torch.device('cuda:0')

print(f'Loading model {args.model} (seqlen={args.seqlen})...')
# Load fp16 model directly without quantization
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.half, device_map='cpu')
model.seqlen = args.seqlen
model.eval()

print('Loading data...')
_, testloader = get_loaders(
    'wikitext2', nsamples=args.nsamples, seed=0,
    model=args.model, seqlen=args.seqlen)

model_type = parse_model(model)
layers = get_layers(model, model_type)
embeddings = get_embedding(model, model_type)
for emb in embeddings:
    emb.to(DEV)
layers[0].to(DEV)

inps = torch.zeros((args.nsamples, args.seqlen, model.config.hidden_size),
                    dtype=torch.half, device=DEV)
cache = {'i': 0, 'attention_mask': None}

class Catcher(nn.Module):
    def __init__(self, m): super().__init__(); self.module = m
    def forward(self, inp, **kwargs):
        inps[cache['i']] = inp; cache['i'] += 1
        cache['attention_mask'] = kwargs.get('attention_mask')
        cache['position_ids']   = kwargs.get('position_ids')
        raise ValueError

layers[0] = Catcher(layers[0])
testenc_ids = testloader.input_ids
for i in range(args.nsamples):
    try:
        model(testenc_ids[:, i*args.seqlen:(i+1)*args.seqlen].to(DEV))
    except ValueError:
        pass
layers[0] = layers[0].module
layers[0].cpu()
for emb in embeddings: emb.cpu()
torch.cuda.empty_cache()

attention_mask = cache['attention_mask']
position_ids   = cache.get('position_ids')

activations = {}
outs = torch.zeros_like(inps)

for layer_idx, layer in enumerate(layers):
    print(f'Layer {layer_idx}')
    layer = layer.to(DEV)

    # Find k_proj and v_proj
    k_proj = v_proj = None
    for name, m in layer.named_modules():
        if name == 'self_attn.k_proj':
            k_proj = m
        elif name == 'self_attn.v_proj':
            v_proj = m

    k_acts, v_acts = [], []

    def hook_k(module, inp, out):
        # out: (batch=1, seqlen, hidden)
        k_acts.append(out.float().detach().cpu())
    def hook_v(module, inp, out):
        v_acts.append(out.float().detach().cpu())

    hk = k_proj.register_forward_hook(hook_k)
    hv = v_proj.register_forward_hook(hook_v)

    for j in range(args.nsamples):
        outs[j] = layer(
            inps[j].unsqueeze(0),
            attention_mask=attention_mask,
            position_ids=position_ids
        )[0]

    hk.remove(); hv.remove()

    # Concatenate all samples: (nsamples × seqlen, hidden)
    k_cat = torch.cat(k_acts, dim=0).squeeze(0)  # (seqlen, hidden)
    v_cat = torch.cat(v_acts, dim=0).squeeze(0)
    activations[f'self_attn.k_proj.{layer_idx}'] = k_cat
    activations[f'self_attn.v_proj.{layer_idx}'] = v_cat

    del k_acts, v_acts, k_cat, v_cat
    layer.cpu()
    torch.cuda.empty_cache()

    inps, outs = outs, inps

print(f'Saving activations to {args.output_path} ...')
with open(args.output_path, 'wb') as f:
    pickle.dump(activations, f, protocol=pickle.HIGHEST_PROTOCOL)
print('Done.')
