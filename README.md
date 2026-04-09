# mlx-memory-bench

MLX benchmark suite -- measures KV cache memory scaling, inference throughput, and context length limits on Apple Silicon. One script, one dependency.

## Setup

```bash
pip install mlx mlx-lm
```

## Quick start

```bash
# KV cache memory sweep
python3 mlx_memory_bench.py kv-sweep \
    --model mlx-community/Qwen3.5-35B-A3B-4bit \
    --ctx 2048,4096,8192,16384,32768,65536,131072 \
    --output results/kv_sweep_qwen35_35b.json

# E2E inference at fixed context
python3 mlx_memory_bench.py inference \
    --model mlx-community/Qwen3.5-35B-A3B-4bit \
    --ctx 16384 --output-tokens 128 --repeats 3 \
    --output results/inference_qwen35_35b.json

# Speculative decoding comparison
python3 mlx_memory_bench.py spec-decode \
    --model mlx-community/Qwen3.5-35B-A3B-4bit \
    --draft-model mlx-community/Qwen3.5-4B-4bit \
    --ctx 16384 --output-tokens 128 --repeats 3 \
    --output results/specdec_qwen35.json
```

See `benchmarks/` for model-specific configs. Run `python3 mlx_memory_bench.py <command> --help` for all options.

## Benchmarks

### KV cache memory sweep

Measures TTFT, decode speed, and GPU memory at increasing context lengths. Finds the practical context limit for a given model on 128GB.

| Metric | What it measures |
|--------|-----------------|
| TTFT | Time to first token (prefill + prompt processing) |
| Prefill t/s | Input tokens processed per second |
| Decode t/s | Output tokens generated per second |
| GPU peak | Total GPU memory (model weights + KV cache + overhead) |
| KV cache | Memory used by KV cache alone (peak minus model weights) |

### E2E inference

Runs direct MLX inference at a fixed context length with configurable repeats for statistical confidence.

### Speculative decoding

Loads target + draft model simultaneously, compares baseline decode speed vs speculative decoding.

## Hardware

| Spec | M5 Max |
|------|--------|
| Chip | Apple M5 Max |
| GPU cores | 40 (with Neural Accelerators) |
| Memory | 128 GB unified |
| Bandwidth | ~546 GB/s |
| macOS | 26.4 |
