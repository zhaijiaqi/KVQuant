#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
MODEL="${MODEL:-/data/models/LLaMA-7B}"
RESULT_DIR="${RESULT_DIR:-results/k4-lookahead}"
QDIR="${QDIR:-/data/kvquant/quantizers/k4_lookahead}"
FULL_QDIR="${FULL_QDIR:-/data/kvquant/quantizers/k4_lookahead_full_ppl}"
LOOKAHEAD_NSAMPLES="${LOOKAHEAD_NSAMPLES:-4}"
QUANT_NSAMPLES="${QUANT_NSAMPLES:-16}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-42}"

mkdir -p "$RESULT_DIR/logs" "$QDIR" "$FULL_QDIR" strategies

timestamp() {
  date '+[%Y-%m-%d %H:%M:%S %Z]'
}

run_conda() {
  /home/ubuntu/miniconda3/bin/conda run --no-capture-output -n rlkv "$@"
}

calibrate_quantizer() {
  local bit="$1"
  local granularity="$2"
  local out="$QDIR/lookahead_nuq${bit}_${granularity}.pkl"
  if [[ -f "$out" ]]; then
    echo "$(timestamp) quantizer exists: $out"
    return
  fi

  local extra=()
  if [[ "$bit" == "2" ]]; then
    extra+=(--norm)
  fi
  if [[ "$granularity" == "per-channel" ]]; then
    extra+=(--value-perchannel-layers {0..31})
  elif [[ "$granularity" == "tile32" ]]; then
    extra+=(--value-tile-size 32 --value-tile-all-layers)
  fi

  echo "$(timestamp) calibrate bit=${bit} granularity=${granularity} -> ${out}"
  run_conda python -u kvquant/quant/llama_simquant.py "$MODEL" \
    --abits "$bit" \
    --nsamples "$QUANT_NSAMPLES" \
    --seed "$SEED" \
    --seqlen "$SEQLEN" \
    --nuq \
    --fisher /data/kvquant/fisher-llama-7b \
    --include_sparse \
    --sparsity-threshold 0.99 \
    "${extra[@]}" \
    --quantize \
    --quantizer-path "$out"
}

main() {
  cd "$REPO_ROOT"
  echo "$(timestamp) starting K4 lookahead experiment in $REPO_ROOT"

  for bit in 2 3 4; do
    for granularity in per-token per-channel tile32; do
      calibrate_quantizer "$bit" "$granularity"
    done
  done

  echo "$(timestamp) run K4 candidate RMSE and strategy selection"
  run_conda python -u scripts/k4_lookahead_experiment/k4_lookahead_eval.py \
    --repo-root "$REPO_ROOT" \
    --model "$MODEL" \
    --result-dir "$RESULT_DIR" \
    --strategy-dir strategies \
    --quantizer-dir "$QDIR" \
    --nsamples "$LOOKAHEAD_NSAMPLES" \
    --seed "$SEED" \
    --seqlen "$SEQLEN" \
    --maxseqlen "$SEQLEN" \
    --k 4

  echo "$(timestamp) run full PPL for fixed-bit K4 strategies"
  run_conda python -u scripts/k4_lookahead_experiment/run_k4_full_ppl.py \
    --repo-root "$REPO_ROOT" \
    --result-dir "$RESULT_DIR" \
    --quantizer-dir "$FULL_QDIR" \
    strategies/k4_fixed2_value_strategy.json \
    strategies/k4_fixed3_value_strategy.json \
    strategies/k4_fixed4_value_strategy.json

  echo "$(timestamp) write README report"
  run_conda python -u scripts/k4_lookahead_experiment/write_k4_readme.py \
    --repo-root "$REPO_ROOT" \
    --result-dir "$RESULT_DIR" \
    --output README_k4_lookahead_experiment.md

  echo "$(timestamp) completed K4 lookahead experiment"
}

main "$@"
