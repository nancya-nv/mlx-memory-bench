#!/bin/bash
# Run all methodology review tests in order.
# Usage: bash run_methodology_review.sh [model_path_35b] [model_path_122b] [model_path_72b]
#
# Defaults assume models are in ~/models/. Override with positional args.

set -e

MODEL_35B="${1:-mlx-community/Qwen3.5-35B-A3B-4bit}"
MODEL_122B="${2:-mlx-community/Qwen3.5-122B-A10B-4bit}"
MODEL_72B="${3:-mlx-community/Qwen2.5-72B-Instruct-4bit}"

SCRIPT="python3 mlx_memory_bench.py"

echo "================================================================"
echo "Methodology Review — Full Test Suite"
echo "================================================================"
echo "35B model:  $MODEL_35B"
echo "122B model: $MODEL_122B"
echo "72B model:  $MODEL_72B"
echo "================================================================"
echo ""

# --- Test 1: Repeated KV sweep, 35B ---
echo ">>> TEST 1/7: KV sweep 35B, repeats=3"
$SCRIPT kv-sweep \
    --model "$MODEL_35B" \
    --ctx 2048,4096,8192,16384,32768,65536,131072 \
    --repeats 3 \
    --output results/kv_sweep_35b_repeated.json
echo ""

# --- Test 2: Repeated KV sweep, 122B ---
echo ">>> TEST 2/7: KV sweep 122B, repeats=3"
$SCRIPT kv-sweep \
    --model "$MODEL_122B" \
    --ctx 2048,4096,8192,16384,32768,65536,131072 \
    --repeats 3 \
    --output results/kv_sweep_122b_repeated.json
echo ""

# --- Test 3: Spike investigation, 122B ---
echo ">>> TEST 3/7: Spike investigation 122B (16K,32K,64K x5, warmup)"
$SCRIPT kv-sweep \
    --model "$MODEL_122B" \
    --ctx 16384,32768,65536 \
    --repeats 5 \
    --warmup \
    --output results/kv_sweep_122b_spike_investigation.json
echo ""

# --- Test 4: Memory overhead ---
# Run both models at 3 context lengths each, extract baseline_mem_mb from results.
# Uses repeats=1 since we care about memory values, not variance.
echo ">>> TEST 4/7: Memory overhead — 35B at 8K,32K,128K"
$SCRIPT kv-sweep \
    --model "$MODEL_35B" \
    --ctx 8192,32768,131072 \
    --repeats 1 \
    --output results/baseline_memory_overhead_35b.json
echo ""
echo ">>> TEST 4/7: Memory overhead — 122B at 8K,32K,128K"
$SCRIPT kv-sweep \
    --model "$MODEL_122B" \
    --ctx 8192,32768,131072 \
    --repeats 1 \
    --output results/baseline_memory_overhead_122b.json
echo ""

# --- Test 5: Ollama comparison ---
# This test requires Ollama to be installed and running.
# Skip if ollama is not available.
echo ">>> TEST 5/7: Ollama comparison"
if command -v ollama &> /dev/null; then
    echo "Ollama found. Running comparison..."
    echo "NOTE: You need to pull models first:"
    echo "  ollama pull qwen3.5:35b"
    echo "  ollama pull qwen3.5:122b"
    echo "Skipping automated run — run manually with local-agent-bench."
else
    echo "Ollama not installed. Skipping test 5."
    echo '{"status": "skipped", "reason": "ollama not installed"}' > results/ollama_comparison.json
fi
echo ""

# --- Test 6: 72B dense sweep ---
echo ">>> TEST 6/7: KV sweep 72B dense"
$SCRIPT kv-sweep \
    --model "$MODEL_72B" \
    --ctx 2048,4096,8192,16384,32768,65536,131072 \
    --repeats 1 \
    --output results/kv_sweep_72b_dense.json
echo ""

# --- Test 7: Extended context, 35B ---
echo ">>> TEST 7/7: Extended context 35B (256K, 512K)"
$SCRIPT kv-sweep \
    --model "$MODEL_35B" \
    --ctx 262144,524288 \
    --repeats 1 \
    --output results/kv_sweep_35b_extended.json
echo ""

echo "================================================================"
echo "All tests complete. Results in results/"
echo "================================================================"
ls -la results/*.json
