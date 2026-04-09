# Qwen3.5-35B-A3B (MoE, 3B active)

MoE model, 3B active parameters per token. 19GB at 4-bit.

## KV cache sweep

```bash
python3 mlx_memory_bench.py kv-sweep \
    --model mlx-community/Qwen3.5-35B-A3B-4bit \
    --ctx 2048,4096,8192,16384,32768,65536,131072 \
    --output results/kv_sweep_qwen35_35b.json
```

## E2E inference (16K context)

```bash
python3 mlx_memory_bench.py inference \
    --model mlx-community/Qwen3.5-35B-A3B-4bit \
    --ctx 16384 --output-tokens 128 --repeats 5 \
    --output results/inference_qwen35_35b.json
```

## Notes
- MLX community 4-bit quantization
- Qwen3.5 is the successor to Qwen3, ranked top-3 open model on Artificial Analysis
- MoE architecture: 35B total params, only ~3B active per token
