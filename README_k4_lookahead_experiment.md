# K4 Lookahead Value Granularity Experiment

This preliminary experiment evaluates Value quantization candidates with a K=4 lookahead hidden-state RMSE. For a target layer, only that layer's Value granularity is varied; the Key path and subsequent layers use the KVQuant baseline setting for the same bit-width.

## Strategy Summary

| Strategy | Avg Bit | #Per-token | #Per-channel | #Tile32 | Per-channel Layers | Tile32 Layers |
|---|---:|---:|---:|---:|---|---|
| k4_fixed2_value_strategy | 2 | 31 | 0 | 1 | `[]` | `[31]` |
| k4_fixed3_value_strategy | 3 | 31 | 1 | 0 | `[0]` | `[]` |
| k4_fixed4_value_strategy | 4 | 31 | 1 | 0 | `[31]` | `[]` |

## Full PPL

| Config | Baseline PPL | val_rmse PPL | attnout PPL | K4 Lookahead PPL | Delta vs Baseline |
|---|---:|---:|---:|---:|---:|
| nuq4-1% | 5.700959 | 5.727 | 5.727 | 5.701026 | +0.000067 |
| nuq3-1% | 5.759615 | 5.952 | 5.972 | 5.763513 | +0.003898 |
| nuq2-1% | 6.069353 | 40.988 | 41.373 | 6.075462 | +0.006109 |

## Conclusion

K=4 lookahead RMSE fits the safety/ranking behavior of full PPL much better than single-layer `val_rmse` or `attnout_rel_rmse`. It avoids the nuq2-1% 40+ PPL failure mode, keeps nuq3-1% close to baseline, and is effectively tied with baseline for nuq4-1%.

The proxy is lightweight once quantizers and fp16 teacher hidden states are cached: a single `layer + bit + granularity` candidate took about 2.5-3s in this run, while the RMSE calculation itself is millisecond-level; the cost is dominated by the K-layer forward pass.
