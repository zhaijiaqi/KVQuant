import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path


BASELINES = {
    2: 6.069353103637695,
    3: 5.759615421295166,
    4: 5.700958728790283,
}


def parse_strategy(path):
    data = json.loads(Path(path).read_text())
    value_strategy = data["value_strategy"]
    bit = int(data["bit"])
    buckets = {"per-token": [], "per-channel": [], "tile32": []}
    for layer_s, spec in value_strategy.items():
        buckets[spec["granularity"]].append(int(layer_s))
    for value in buckets.values():
        value.sort()
    return data["name"], bit, buckets


def run_command(cmd, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return log_path.read_text(errors="ignore")


def extract_ppl(text):
    values = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?", stripped, flags=re.I):
            values.append(float(stripped))
    if not values:
        raise RuntimeError("Could not find final PPL scalar in eval log.")
    return values[-1]


def build_quant_args(bit, buckets):
    args = [
        sys.executable,
        "-u",
        "kvquant/quant/llama_simquant.py",
        "/data/models/LLaMA-7B",
        "--abits",
        str(bit),
        "--nsamples",
        "16",
        "--seed",
        "0",
        "--seqlen",
        "2048",
        "--nuq",
        "--fisher",
        "/data/kvquant/fisher-llama-7b",
        "--include_sparse",
        "--sparsity-threshold",
        "0.99",
    ]
    if bit == 2:
        args.append("--norm")
    if buckets["per-channel"]:
        args += ["--value-perchannel-layers"] + [str(x) for x in buckets["per-channel"]]
    if buckets["tile32"]:
        args += ["--value-tile-size", "32", "--value-tile-layers"] + [str(x) for x in buckets["tile32"]]
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--result-dir", default="results/k4-lookahead")
    parser.add_argument("--quantizer-dir", default="/data/kvquant/quantizers/k4_lookahead_full_ppl")
    parser.add_argument("strategies", nargs="+")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    result_dir = repo_root / args.result_dir
    quantizer_dir = Path(args.quantizer_dir)
    logs_dir = result_dir / "full_ppl_logs"
    quantizer_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for strategy_path in args.strategies:
        name, bit, buckets = parse_strategy(strategy_path)
        quantizer_path = quantizer_dir / f"{name}.pkl"
        calib_log = logs_dir / f"{name}.calib.log"
        eval_log = logs_dir / f"{name}.eval.log"

        base_cmd = build_quant_args(bit, buckets)
        if not quantizer_path.exists():
            print(f"[full-ppl] calibrating {name} -> {quantizer_path}", flush=True)
            run_command(base_cmd + ["--quantize", "--quantizer-path", str(quantizer_path)], calib_log, repo_root)

        print(f"[full-ppl] evaluating {name}", flush=True)
        text = run_command(base_cmd + ["--quantizer-path", str(quantizer_path)], eval_log, repo_root)
        ppl = extract_ppl(text)
        nll = math.log(ppl)
        baseline = BASELINES[bit]
        rows.append(
            {
                "strategy": name,
                "bit_mode": f"fixed{bit}",
                "avg_bit": bit,
                "full_ppl": ppl,
                "full_nll": nll,
                "baseline_ppl": baseline,
                "delta_ppl_vs_baseline": ppl - baseline,
                "kv_memory_ratio": bit / 16.0,
                "comment": "",
            }
        )

        out_path = result_dir / "k4_full_ppl_summary.csv"
        with out_path.open("w", newline="") as handle:
            fieldnames = [
                "strategy",
                "bit_mode",
                "avg_bit",
                "full_ppl",
                "full_nll",
                "baseline_ppl",
                "delta_ppl_vs_baseline",
                "kv_memory_ratio",
                "comment",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"[done] wrote {result_dir / 'k4_full_ppl_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
