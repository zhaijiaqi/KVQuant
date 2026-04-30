# KVQuant Deployment Code

The code in this folder can be used to run the inference experiments from the paper (for benchmarking kernel runtime).

## Installation

1. Create a conda environment
```
conda create --name benchmark python=3.9 -y
conda activate benchmark
```

2. Clone and install the dependencies (including the local transformers environment)
```
pip install transformers matplotlib
pip install torch datasets sentencepiece scikit-learn protobuf
pip install accelerate -U
pip install -e .
cd kvquant
python setup_cuda.py install
cd ..
pip install flash-attn --no-build-isolation
```

3. Run kernel benchmarking

Note that the quantizer is obtained from steps in the quant directory.

```
cp ../quant/quantizers.pickle .
CUDA_VISIBLE_DEVICES=0 python cache-llama-activations.py <path-to-llama-7b-hf> --wbits 4 --nsamples 1 --seqlen 2048 --quantizer-path quantizers.pickle --output-path activations-seqlen2048.pickle;
```

Assuming the activations and the quantizers are stored in "activations.pickle" and "quantizers.pickle", you can run the kernel benchmarks by running the python scripts in the "scripts" folder.

## Profile key/value distributions

To profile the Layer 10 key/value activation magnitudes on a 2K Wikitext-2 sample and aggregate them at `64x64` token/channel granularity, run:

```
CUDA_VISIBLE_DEVICES=0 python profile-llama-kv-distribution.py <path-to-llama-7b-hf> \
  --seqlen 2048 \
  --layer-idx 10 \
  --token-block 64 \
  --channel-block 64 \
  --reduction max_abs \
  --output-data kv-profile-layer10-64x64.pkl \
  --output-plot kv-profile-layer10-64x64.png
```

This script saves both the aggregated tensors and a 3-panel figure for `Keys pre-RoPE`, `Keys post-RoPE`, and `Values`.

## Analyze per-block quantization

To test whether `64`, `128`, or larger blocks are a reasonable quantization granularity, run:

```
CUDA_VISIBLE_DEVICES=0 python analyze-llama-kv-blocks.py <path-to-llama-7b-hf> \
  --seqlen 2048 \
  --layer-idx 10 \
  --block-sizes 64 128 256 \
  --num-bits 4 \
  --output-dir block-analysis-layer10
```

This script compares `per-tensor`, `per-block`, `per-token`, and `per-channel` quantization error, and also reports block homogeneity statistics such as block CV (`std(|x|) / mean(|x|)`). Lower block quantization error and lower block CV indicate that block-wise quantization is a better fit.
