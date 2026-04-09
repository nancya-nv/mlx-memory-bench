#!/usr/bin/env python3
"""
MLX Memory Benchmark Suite

Measures KV cache memory scaling, inference performance, and context length
limits on Apple Silicon. Direct MLX measurement — no server overhead.

Requires: pip install mlx mlx-lm

Usage:
  # KV cache memory sweep (context length scaling)
  python3 mlx_memory_bench.py kv-sweep \
    --model mlx-community/Qwen3.5-35B-A3B-4bit \
    --ctx 2048,4096,8192,16384,32768,65536,131072 \
    --output results/kv_sweep_qwen35_35b.json

  # E2E inference benchmark (single context, multiple repeats)
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
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import mlx.core as mx
import mlx_lm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_mem():
    """GPU memory in MB."""
    return {
        "active_mb": mx.get_active_memory() / (1024**2),
        "peak_mb": mx.get_peak_memory() / (1024**2),
    }


def make_prompt(n_tokens, tokenizer):
    """Generate a prompt of approximately n_tokens."""
    base = "The quick brown fox jumps over the lazy dog. "
    base_toks = len(tokenizer.encode(base, add_special_tokens=False))
    text = base * ((n_tokens // base_toks) + 2)
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    return tokenizer.decode(ids)


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def system_info():
    """Collect system metadata."""
    info = {"hostname": platform.node(), "platform": platform.platform()}
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True)
        info["chip"] = r.stdout.strip()
    except Exception:
        info["chip"] = "unknown"
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True)
        info["ram_gb"] = int(r.stdout.strip()) // (1024**3)
    except Exception:
        info["ram_gb"] = 0
    try:
        r = subprocess.run(["sw_vers", "-productVersion"],
                           capture_output=True, text=True)
        info["macos"] = r.stdout.strip()
    except Exception:
        pass
    return info


def timed_generate(model, tokenizer, prompt, max_tokens, draft_model=None):
    """Run generation, return timing and memory metrics."""
    # Record baseline memory before inference (model loaded, no KV cache)
    mx.eval(mx.zeros(1))
    baseline_mem = get_mem()
    mx.reset_peak_memory()
    actual_tokens = count_tokens(prompt, tokenizer)

    t0 = time.perf_counter()
    t_first = None
    n_out = 0
    text = ""

    kwargs = {"max_tokens": max_tokens}
    if draft_model is not None:
        kwargs["draft_model"] = draft_model

    for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, **kwargs):
        if t_first is None:
            t_first = time.perf_counter()
        n_out += 1
        text += resp.text

    t_end = time.perf_counter()
    mx.eval(mx.zeros(1))

    ttft_ms = (t_first - t0) * 1000 if t_first else 0
    decode_ms = (t_end - t_first) * 1000 if t_first else 0
    decode_tps = n_out / (decode_ms / 1000) if decode_ms > 0 else 0
    prefill_tps = actual_tokens / (ttft_ms / 1000) if ttft_ms > 0 else 0

    mem = get_mem()

    return {
        "input_tokens": actual_tokens,
        "output_tokens": n_out,
        "ttft_ms": round(ttft_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "decode_tps": round(decode_tps, 1),
        "prefill_tps": round(prefill_tps, 1),
        "gpu_peak_mb": round(mem["peak_mb"]),
        "baseline_mem_mb": round(baseline_mem["active_mb"]),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_kv_sweep(args):
    """Sweep context lengths, measure KV cache memory growth."""
    ctx_list = [int(x) for x in args.ctx.split(",")]
    repeats = args.repeats
    warmup = args.warmup

    print(f"KV Cache Memory Sweep")
    print(f"Model: {args.model}")
    print(f"Contexts: {ctx_list}")
    print(f"Output tokens per run: {args.output_tokens}")
    print(f"Repeats: {repeats} | Warmup: {warmup}")
    print("=" * 80)

    print(f"Loading model...")
    model, tokenizer = mlx_lm.load(args.model)
    mx.eval(model.parameters())
    model_mem = get_mem()["active_mb"]
    print(f"Model memory: {model_mem:,.0f} MB")

    if warmup:
        print("Warmup pass...")
        timed_generate(model, tokenizer, "hello", 1)
        print("Warmup done.")

    print("=" * 80)

    results = []
    failed = False
    for ctx in ctx_list:
        print(f"\nContext: {ctx:,} tokens")
        prompt = make_prompt(ctx, tokenizer)

        runs = []
        for rep in range(repeats):
            if repeats > 1:
                print(f"  Run {rep+1}/{repeats}:")
            try:
                r = timed_generate(model, tokenizer, prompt, args.output_tokens)
                r["ctx"] = ctx
                r["run"] = rep + 1
                r["kv_cache_mb"] = round(r["gpu_peak_mb"] - model_mem)
                r["model_mb"] = round(model_mem)
                r["overhead_mb"] = round(r["gpu_peak_mb"] - r["baseline_mem_mb"])
                r["status"] = "ok"

                prefix = "    " if repeats > 1 else "  "
                print(f"{prefix}TTFT:     {r['ttft_ms']:>10,.0f} ms  ({r['prefill_tps']:,.0f} prefill t/s)")
                print(f"{prefix}Decode:   {r['decode_tps']:>10.1f} t/s  ({r['output_tokens']} tokens)")
                print(f"{prefix}GPU peak: {r['gpu_peak_mb']:>10,.0f} MB  (baseline: {r['baseline_mem_mb']:,.0f} MB)")
                print(f"{prefix}KV cache: {r['kv_cache_mb']:>10,.0f} MB")

                runs.append(r)

            except Exception as e:
                print(f"  FAILED: {e}")
                runs.append({"ctx": ctx, "run": rep + 1, "status": "failed",
                             "error": str(e), "gpu_peak_mb": round(get_mem()["peak_mb"])})
                failed = True
                break

        if failed:
            # Store partial runs for this context length
            results.append({
                "ctx": ctx,
                "status": "failed",
                "runs": runs,
                "error": runs[-1].get("error", "unknown"),
                "gpu_peak_mb": runs[-1].get("gpu_peak_mb", 0),
            })
            print("  Stopping sweep.")
            break

        # Aggregate stats across runs
        ok_runs = [r for r in runs if r["status"] == "ok"]
        ttfts = [r["ttft_ms"] for r in ok_runs]
        decode_list = [r["decode_tps"] for r in ok_runs]
        peak_list = [r["gpu_peak_mb"] for r in ok_runs]
        baseline_list = [r["baseline_mem_mb"] for r in ok_runs]

        def mean(lst): return round(sum(lst) / len(lst), 1) if lst else 0
        def std(lst): return round(statistics.stdev(lst), 1) if len(lst) > 1 else 0

        agg = {
            "ctx": ctx,
            "status": "ok",
            "repeats": len(ok_runs),
            "model_mb": round(model_mem),
            # Means
            "ttft_ms": mean(ttfts),
            "decode_tps": mean(decode_list),
            "prefill_tps": mean([r["prefill_tps"] for r in ok_runs]),
            "gpu_peak_mb": round(mean(peak_list)),
            "kv_cache_mb": round(mean(peak_list) - model_mem),
            "baseline_mem_mb": round(mean(baseline_list)),
            # Stddev
            "ttft_ms_stddev": std(ttfts),
            "decode_tps_stddev": std(decode_list),
            "gpu_peak_mb_stddev": std(peak_list),
            # Individual runs (always included for transparency)
            "runs": runs,
        }

        # Flag unstable points (stddev > 5% of mean)
        unstable = []
        if agg["ttft_ms"] > 0 and agg["ttft_ms_stddev"] / agg["ttft_ms"] > 0.05:
            unstable.append("ttft_ms")
        if agg["decode_tps"] > 0 and agg["decode_tps_stddev"] / agg["decode_tps"] > 0.05:
            unstable.append("decode_tps")
        if unstable:
            agg["unstable"] = unstable

        results.append(agg)

        if repeats > 1:
            print(f"  Mean: TTFT={agg['ttft_ms']:,.0f}ms ±{agg['ttft_ms_stddev']:.0f}  "
                  f"Decode={agg['decode_tps']:.1f}t/s ±{agg['decode_tps_stddev']:.1f}  "
                  f"Peak={agg['gpu_peak_mb']:,.0f}MB")
            if unstable:
                print(f"  ⚠ UNSTABLE: {', '.join(unstable)}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY — {os.path.basename(args.model)}")
    print(f"{'='*80}")
    print(f"Model: {model_mem:,.0f} MB | RAM: {system_info().get('ram_gb', '?')} GB | Repeats: {repeats}")
    if repeats > 1:
        print(f"{'Ctx':>10} {'TTFT mean':>12} {'±':>6} {'Decode mean':>12} {'±':>6} {'Peak':>10} {'KV':>10}")
        print("-" * 70)
        for r in results:
            if r["status"] == "ok":
                flag = " ⚠" if r.get("unstable") else ""
                print(f"{r['ctx']:>10,} {r['ttft_ms']:>10,.0f}ms {r['ttft_ms_stddev']:>5.0f} "
                      f"{r['decode_tps']:>10.1f}t/s {r['decode_tps_stddev']:>5.1f} "
                      f"{r['gpu_peak_mb']:>8,.0f}MB {r['kv_cache_mb']:>8,.0f}MB{flag}")
            else:
                print(f"{r['ctx']:>10,} {'--- FAILED ---':>50} {r.get('gpu_peak_mb',0):>8,.0f}MB")
    else:
        print(f"{'Ctx':>10} {'TTFT':>10} {'Prefill':>12} {'Decode':>10} {'Peak':>10} {'KV':>10}")
        print("-" * 64)
        for r in results:
            if r["status"] == "ok":
                print(f"{r['ctx']:>10,} {r['ttft_ms']:>9,.0f}ms {r['prefill_tps']:>10,.0f}t/s "
                      f"{r['decode_tps']:>8.1f}t/s {r['gpu_peak_mb']:>8,.0f}MB {r['kv_cache_mb']:>8,.0f}MB")
            else:
                print(f"{r['ctx']:>10,} {'--- FAILED ---':>42} {r.get('gpu_peak_mb',0):>8,.0f}MB")

    save(args.output, args.model, model_mem, results, "kv_sweep")


def cmd_inference(args):
    """E2E inference benchmark at a fixed context length."""
    print(f"E2E Inference Benchmark")
    print(f"Model: {args.model}")
    print(f"Context: {args.ctx} | Output: {args.output_tokens} | Repeats: {args.repeats}")
    print("=" * 80)

    model, tokenizer = mlx_lm.load(args.model)
    mx.eval(model.parameters())
    model_mem = get_mem()["active_mb"]
    print(f"Model memory: {model_mem:,.0f} MB")

    prompt = make_prompt(int(args.ctx), tokenizer)
    print(f"Actual tokens: {count_tokens(prompt, tokenizer):,}")
    print("=" * 80)

    # Warmup
    print("Warmup...")
    timed_generate(model, tokenizer, "hello", 1)

    results = []
    for i in range(args.repeats):
        print(f"\nRun {i+1}/{args.repeats}:")
        r = timed_generate(model, tokenizer, prompt, args.output_tokens)
        r["run"] = i + 1
        r["kv_cache_mb"] = round(r["gpu_peak_mb"] - model_mem)
        print(f"  TTFT:   {r['ttft_ms']:>10,.0f} ms  ({r['prefill_tps']:,.0f} prefill t/s)")
        print(f"  Decode: {r['decode_tps']:>10.1f} t/s  ({r['output_tokens']} tokens)")
        results.append(r)

    # Summary
    ttfts = [r["ttft_ms"] for r in results]
    tps_list = [r["decode_tps"] for r in results]
    print(f"\n{'='*80}")
    print(f"Median TTFT: {statistics.median(ttfts):,.0f} ms")
    print(f"Median decode: {statistics.median(tps_list):.1f} t/s")
    if len(ttfts) > 1:
        print(f"Stddev TTFT: {statistics.stdev(ttfts):,.0f} ms")

    save(args.output, args.model, model_mem, results, "inference")


def cmd_spec_decode(args):
    """Compare baseline vs speculative decoding."""
    print(f"Speculative Decoding Comparison")
    print(f"Target: {args.model}")
    print(f"Draft:  {args.draft_model}")
    print(f"Context: {args.ctx} | Output: {args.output_tokens} | Repeats: {args.repeats}")
    print("=" * 80)

    print("Loading target model...")
    model, tokenizer = mlx_lm.load(args.model)
    mx.eval(model.parameters())
    target_mem = get_mem()["active_mb"]
    print(f"Target memory: {target_mem:,.0f} MB")

    print("Loading draft model...")
    draft_model, _ = mlx_lm.load(args.draft_model)
    mx.eval(draft_model.parameters())
    total_mem = get_mem()["active_mb"]
    print(f"Total memory (both models): {total_mem:,.0f} MB")
    print("=" * 80)

    prompt = make_prompt(int(args.ctx), tokenizer)
    actual = count_tokens(prompt, tokenizer)
    print(f"Actual tokens: {actual:,}")

    # Warmup both paths
    print("Warmup...")
    timed_generate(model, tokenizer, "hello", 1)
    timed_generate(model, tokenizer, "hello", 1, draft_model=draft_model)

    baseline_results = []
    specdec_results = []

    for i in range(args.repeats):
        print(f"\n--- Run {i+1}/{args.repeats} ---")

        print(f"  Baseline:")
        r = timed_generate(model, tokenizer, prompt, args.output_tokens)
        r["mode"] = "baseline"
        r["run"] = i + 1
        print(f"    TTFT: {r['ttft_ms']:,.0f}ms | Decode: {r['decode_tps']:.1f} t/s")
        baseline_results.append(r)

        print(f"  Spec decode:")
        r = timed_generate(model, tokenizer, prompt, args.output_tokens,
                           draft_model=draft_model)
        r["mode"] = "speculative"
        r["run"] = i + 1
        print(f"    TTFT: {r['ttft_ms']:,.0f}ms | Decode: {r['decode_tps']:.1f} t/s")
        specdec_results.append(r)

    b_tps = statistics.median([r["decode_tps"] for r in baseline_results])
    s_tps = statistics.median([r["decode_tps"] for r in specdec_results])
    print(f"\n{'='*80}")
    print(f"Baseline decode:  {b_tps:.1f} t/s")
    print(f"SpecDec decode:   {s_tps:.1f} t/s")
    print(f"Speedup:          {s_tps/b_tps:.2f}x")

    save(args.output, args.model, target_mem,
         {"baseline": baseline_results, "speculative": specdec_results,
          "draft_model": args.draft_model},
         "spec_decode")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save(path, model, model_mem, results, bench_type):
    out = {
        "bench_type": bench_type,
        "model": model,
        "model_mem_mb": model_mem,
        "system": system_info(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MLX Memory Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # kv-sweep
    kv = sub.add_parser("kv-sweep", help="KV cache memory scaling sweep")
    kv.add_argument("--model", required=True)
    kv.add_argument("--ctx", default="2048,4096,8192,16384,32768,65536,131072")
    kv.add_argument("--output-tokens", type=int, default=64)
    kv.add_argument("--repeats", type=int, default=1,
                    help="Runs per context length (default: 1)")
    kv.add_argument("--warmup", action="store_true",
                    help="Run one silent inference pass before measurement")
    kv.add_argument("--output", default="results/kv_sweep.json")

    # inference
    inf = sub.add_parser("inference", help="E2E inference benchmark")
    inf.add_argument("--model", required=True)
    inf.add_argument("--ctx", default="16384")
    inf.add_argument("--output-tokens", type=int, default=128)
    inf.add_argument("--repeats", type=int, default=3)
    inf.add_argument("--output", default="results/inference.json")

    # spec-decode
    sd = sub.add_parser("spec-decode", help="Speculative decoding comparison")
    sd.add_argument("--model", required=True)
    sd.add_argument("--draft-model", required=True)
    sd.add_argument("--ctx", default="16384")
    sd.add_argument("--output-tokens", type=int, default=128)
    sd.add_argument("--repeats", type=int, default=3)
    sd.add_argument("--output", default="results/specdec.json")

    args = parser.parse_args()

    if args.command == "kv-sweep":
        cmd_kv_sweep(args)
    elif args.command == "inference":
        cmd_inference(args)
    elif args.command == "spec-decode":
        cmd_spec_decode(args)


if __name__ == "__main__":
    main()
