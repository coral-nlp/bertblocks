`modal run scripts/run_benchmark_modal.py --batch-size 32 --max-length 4096 --uneven --num-iterations 20 --num-warmup 5`

```
Device       : cuda
Model        : answerdotai/ModernBERT-base
Batch size   : 32
Max length   : 4096
Uneven docs  : True
Warmup iters : 5
Timed iters  : 20
Load weights : True
Attn backend : flash_attention_2
Compile      : False
```

============================================================
RESULTS
============================================================

  HuggingFace ModernBERT
    Mean latency :  2127.17 ms
    p50  latency :  2128.41 ms
    p95  latency :  2141.69 ms
    p99  latency :  2150.34 ms
    Throughput   :       15.0 seq/s  |         32590 tok/s

  BertBlocks  ModernBERT
    Mean latency :   691.16 ms
    p50  latency :   692.55 ms
    p95  latency :   698.00 ms
    p99  latency :   699.12 ms
    Throughput   :       46.3 seq/s  |        100302 tok/s

Speedup (BertBlocks / HuggingFace) : 3.08x
============================================================

```
Device       : cuda
Model        : answerdotai/ModernBERT-base
Batch size   : 32
Max length   : 4096
Uneven docs  : True
Warmup iters : 5
Timed iters  : 20
Load weights : True
Attn backend : flash_attention_2
Compile      : True
```

============================================================
RESULTS
============================================================

  HuggingFace ModernBERT
    Mean latency :  1055.80 ms
    p50  latency :  1056.68 ms
    p95  latency :  1062.15 ms
    p99  latency :  1062.22 ms
    Throughput   :       30.3 seq/s  |         64349 tok/s

  BertBlocks  ModernBERT
    Mean latency :   587.07 ms
    p50  latency :   587.99 ms
    p95  latency :   593.43 ms
    p99  latency :   596.11 ms
    Throughput   :       54.5 seq/s  |        115728 tok/s

Speedup (BertBlocks / HuggingFace) : 1.80x
============================================================

```
Device       : cuda
Model        : answerdotai/ModernBERT-base
Batch size   : 32
Max length   : 4096
Uneven docs  : False
Warmup iters : 5
Timed iters  : 20
Load weights : True
Attn backend : flash_attention_2
Compile      : True
```

============================================================
RESULTS
============================================================

  HuggingFace ModernBERT
    Mean latency :  1392.71 ms
    p50  latency :  1397.90 ms
    p95  latency :  1409.83 ms
    p99  latency :  1412.84 ms
    Throughput   :       23.0 seq/s  |         94113 tok/s

  BertBlocks  ModernBERT
    Mean latency :  1367.63 ms
    p50  latency :  1374.13 ms
    p95  latency :  1386.49 ms
    p99  latency :  1389.94 ms
    Throughput   :       23.4 seq/s  |         95839 tok/s

Speedup (BertBlocks / HuggingFace) : 1.02x
============================================================

```
Device       : cuda
Model        : answerdotai/ModernBERT-base
Batch size   : 32
Max length   : 4096
Uneven docs  : False
Warmup iters : 5
Timed iters  : 20
Load weights : True
Attn backend : flash_attention_2
Compile      : False
```

============================================================
RESULTS
============================================================

  HuggingFace ModernBERT
    Mean latency :  2183.20 ms
    p50  latency :  2182.54 ms
    p95  latency :  2191.46 ms
    p99  latency :  2192.60 ms
    Throughput   :       14.7 seq/s  |         60037 tok/s

  BertBlocks  ModernBERT
    Mean latency :  1292.24 ms
    p50  latency :  1291.62 ms
    p95  latency :  1300.13 ms
    p99  latency :  1306.79 ms
    Throughput   :       24.8 seq/s  |        101430 tok/s

Speedup (BertBlocks / HuggingFace) : 1.69x
============================================================
