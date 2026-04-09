# Methodology Review

Open questions from the initial benchmark pass. Each section documents the concern, what to test, and what result closes it.

---

## 1. Repeatability — KV sweep runs once per context length

**Problem:** The KV sweep records a single measurement at each context length. No variance data means we can't tell whether numbers are stable or noisy.

**Test:** Re-run the full KV sweep for both models with `--repeats 3`. Report mean and stddev for TTFT, decode t/s, and GPU peak at each context length. Flag any point where stddev > 5% of mean as unstable.

**Done when:** Every row in the sweep table has mean +/- stddev. Unstable points are called out explicitly.

---

## 2. 122B decode speed anomaly at 32-64K

**Problem:** The 122B shows 62 t/s at 16K, then jumps to 133 t/s at 32K and 130 t/s at 64K before dropping to 39 t/s at 128K. Decode speed increasing with context length is not expected behavior and needs explanation before these numbers can be trusted.

**Test:**
- Run 16K, 32K, 64K with `--repeats 5`, log per-run decode speed (not just average)
- Add a warmup pass before measurement: one silent inference run, then measure. If the spike disappears after warmup, it's a Metal JIT/cache artifact. If it persists, it's real.

**Done when:** Either (a) spike is explained and attributed to warmup artifact with evidence, or (b) spike is confirmed real and a plausible mechanism is documented.

---

## 3. No baseline comparison

**Problem:** MLX decode speeds are reported in isolation. Without a comparison point, claims about relative performance are assertions, not findings.

**Test:** Run the same two models through Ollama at 2K, 16K, and 32K context. Record TTFT and decode t/s.

**Done when:** A comparison table exists with MLX vs Ollama side-by-side at the same context lengths.

---

## 4. Usable memory overhead is asserted, not measured

**Problem:** The ~90-95GB usable memory figure (128GB minus macOS/Metal overhead) is stated as fact but not derived from the benchmark data.

**Test:** At each run, record: (a) baseline memory after model load but before inference, (b) peak memory during inference. Overhead = peak minus (model weights + estimated KV cache). Do this at 3 context lengths per model and report actual overhead numbers.

**Done when:** Overhead is a measured range with data behind it, not a rule of thumb.

---

## 5. Only two models tested

**Problem:** 19GB and 65GB are the only data points. The gap between them is large and the "~300K+ context for 70B dense" claim has no supporting data.

**Test:** Add Qwen2.5-72B-Instruct-4bit (~38GB) to the sweep. Run the full KV sweep at the same context lengths as the existing models.

**Done when:** A third sweep table exists for the 72B dense model, and the 300K+ context claim is either validated or corrected.

---

## 6. 500K+ context extrapolation for the 35B

**Problem:** The claim that the 35B could handle 500K+ tokens is extrapolated from linear KV cache growth. Linear scaling assumptions may not hold at extreme context lengths, and MLX stability at that scale is untested.

**Test:** Run the 35B at 256K and 512K context. Record whether it completes, errors, or times out. Record TTFT if it completes.

**Done when:** Either (a) 256K and 512K runs complete with data, or (b) failure mode is documented.

---

## Output files

Each test should write results to `results/` using these filenames:

| Test | Output file |
|------|-------------|
| Repeated KV sweep -- 35B | `kv_sweep_35b_repeated.json` |
| Repeated KV sweep -- 122B | `kv_sweep_122b_repeated.json` |
| Spike investigation -- 122B | `kv_sweep_122b_spike_investigation.json` |
| Memory overhead measurement | `baseline_memory_overhead.json` |
| Ollama comparison | `ollama_comparison.json` |
| 72B dense sweep | `kv_sweep_72b_dense.json` |
| Extended context -- 35B | `kv_sweep_35b_extended.json` |
