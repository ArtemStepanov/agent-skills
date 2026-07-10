---
name: tuning-local-llms
description: Finds the fastest working configuration for running a local LLM (GGUF / llama.cpp family) on the user's specific hardware, using measured benchmarks instead of guessed flags. Use when a local model is slow, out-of-memory, crashes on load, or won't fit; when adding a new GGUF model; after a GPU, driver, or llama.cpp upgrade; or when the user asks to speed up, optimize, or tune local inference with llama.cpp, llama-server, Ollama, LM Studio, koboldcpp, Jan, or llama-swap — including choosing context size, GPU layer offload, KV cache quantization, batch sizes, threads, or MoE expert offload.
license: MIT
metadata:
  version: "1.0"
compatibility: Needs only a shell - the workflow degrades gracefully when tools are missing. Python 3 recommended (bundled scripts have documented manual fallbacks). Benchmarking uses whichever is available - a llama.cpp binary (llama-bench or llama-server), koboldcpp, or the HTTP endpoint of the user's existing runtime (Ollama, LM Studio, Jan).
---

# Tuning local LLMs

Find the best parameters by measuring on the user's machine, not by trusting
community numbers. Community advice is directional; this hardware is the ground
truth. Analytical estimates *seed* the search, a real model load *verifies* it,
and a benchmark *decides* it. Change one variable at a time, and record why each
winning flag won.

A full tune takes ~15–40 minutes of benchmarking for a dense model (more for a
large MoE). Tell the user this up front. If they say "just give me something
reasonable," do Steps 1–3, apply the analytical seed, verify it loads, and skip
the sweeps.

## Step 1 — Establish the goal

Ask one question: **"What will you use this model for?"** The goal decides what
you optimize. Also get their target context size and whether small quality
trades (KV cache quantization) are acceptable.

Gauge who you're talking to, and size the plan to them: for a non-technical
user, lead with the smallest set of safe, reversible changes (GUI paths where
they exist) and offer the rigorous sweep as an explicit opt-in — rigor the
user can't comfortably execute is not rigor. Same for an expert in a hurry:
depth is opt-in, not default.

| Goal | Optimize for | What changes |
|---|---|---|
| Interactive chat | tg (generation tok/s) at 2–8k context depth | GPU offload is nearly everything; batch sizes barely matter |
| Coding agent / RAG / long documents | pp (prompt tok/s) without tanking tg; large context is mandatory | `-ub` sweep becomes critical; benchmark at 16k+ depth; KV q8_0 usually needed to fit context |
| Maximum context | largest context that keeps tg above a floor the user names (e.g. ≥15 tok/s) | Invert the search: fix the tg floor, then trade GPU layers, KV quant, and `-ub` for context |
| Minimum memory / co-resident models | best speed inside a hard memory cap | Smaller quant, KV q8_0, consider `--no-kv-offload` |
| Serving several users | aggregate throughput at N parallel requests | Different instrument: `llama-batched-bench` — see [references/benchmarking.md](references/benchmarking.md) |

## Step 2 — Discover the environment

Never assume. Probe with read-only commands, then confirm the picture with the
user.

**Runtime:** find out what they run models with — `command -v llama-server
llama-bench ollama koboldcpp` (Windows: `where`), look for LM Studio / Jan
installs. This decides two things: the *measurement instrument* and the *final
config format*. Preference order for measuring: `llama-bench` (best: native
parameter sweeps) > koboldcpp `--benchmark` > timed HTTP requests via
`scripts/bench-api.py`. If the user's runtime is Ollama, LM Studio, or Jan, the
highest-leverage path is usually: benchmark with a raw llama.cpp binary, then
port the winning settings back — see [references/runtimes.md](references/runtimes.md)
for what ports and what can't.

**Hardware:** GPU model + free VRAM, physical CPU core count (P-cores only on
hybrid Intel), total RAM. Quick probes: `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv`
(NVIDIA, incl. WSL2); `lscpu` (Linux); `sysctl hw.perflevel0.physicalcpu hw.memsize`
(macOS). Full per-platform command table, including AMD, Windows, and the traps
(Windows WMI caps reported VRAM at 4GB; Apple Silicon has no VRAM — it's
unified memory): [references/platforms.md](references/platforms.md).

**Tool availability:** never hand the user a command without verifying the
tool exists (`command -v`, Windows `where`). A missing tool is a ladder, not a
wall: no llama-bench → koboldcpp `--benchmark` → `scripts/bench-api.py`
against the HTTP endpoint of the runtime the user already runs (needs only
Python and that server). No Python (common on Windows) → skip the scripts:
take model facts from llama-server's load log and compute the KV formula from
Step 3 by hand. Only suggest installing something (llama.cpp for proper
sweeps) when the degraded path genuinely can't answer the user's question —
install one-liners per platform are in [references/platforms.md](references/platforms.md).

**Model:** run `python3 <skill-dir>/scripts/gguf-meta.py <model.gguf>`, where
`<skill-dir>` is the directory containing this `SKILL.md`; it prints layer
count, KV heads, head dim, expert count (MoE or dense), and
the computed KV-cache cost per context size. KV heads drive context cost: 2 KV
heads means context is cheap (go big); 8+ means it's expensive (quantize KV,
cap context). Users often already have GGUFs on disk from another tool
(Ollama's blob store contains plain GGUFs) — reuse, don't re-download; paths in
[references/runtimes.md](references/runtimes.md).

## Step 3 — Compute the budget before benchmarking anything

Nothing else matters if the model is swapping or OOMing — paging destroys every
benchmark signal. Budget:

```
total = weights-on-GPU + KV cache + compute buffers + mmproj (vision models)
        + runtime overhead (~0.3–0.6 GB) + headroom (~0.5–1 GB)
```

- **Weights:** GGUF file size is ground truth, scaled by the fraction on GPU.
- **KV cache** = `2 × layers × kv_heads × head_dim × ctx × bytes` (f16=2,
  q8_0≈1, q4_0≈0.5). The leading `2` is K-and-V — it is the most commonly
  dropped factor; check your result against this worked example: 40 layers ×
  8 KV heads × 128 head_dim, f16 → 2×40×8×128×2 B ≈ 160 KiB/token ≈ 5.4 GB
  at 32k context. `gguf-meta.py` computes this for you. Overestimates for
  sliding-window (Gemma-class) and MLA (DeepSeek-class) models — treat as an
  upper bound there.
- **Compute buffers** grow with `-ub`; no reliable closed form — read the
  `compute buffer size` lines from the llama.cpp load log.
- **mmproj:** vision projectors can cost 0.5–2 GB VRAM; check the file size.

Modern llama.cpp (late 2025+) defaults to `-ngl auto` with `--fit on`, which
computes a *fitting* placement automatically — use `llama-fit-params` output as
your seed if available. But fit ≠ fast: it optimizes for loading, not
throughput, and MoE placement in particular usually beats it after manual
tuning. Two traps: `-c 0` now means *the model's full trained context* (often
128k+ — a VRAM bomb; always set `-c` explicitly), and low `--fit-target`
headroom OOMs long-running servers (keep ≥512–1024 MiB).

## Step 4 — Staged tuning

The parameter space decomposes; tune in this order. Each stage is typically ONE
`llama-bench` invocation using its comma-separated sweep syntax (see
[references/benchmarking.md](references/benchmarking.md) for exact commands,
hygiene rules, and what to do without llama-bench). Use short cheap runs while
exploring; long high-repetition runs only for the final confirmation.

1. **Memory layout** (dominates tg — effects up to 10×).
   - *Dense model:* highest `-ngl` that fits with headroom.
   - *MoE model:* keep `-ngl 99` and offload experts instead — start
     `--cpu-moe`, then sweep `--n-cpu-moe` downward (e.g. 30, 24, 20, 16, 12)
     until VRAM is full. Routinely 2–4× faster than lowering `-ngl`. The
     optimum is machine-specific and sometimes non-monotonic — measure.
   - Leave `-fa` on `auto` (NVIDIA: always beneficial; AMD Vulkan and old
     Pascal cards: A/B test it — see [references/flags.md](references/flags.md)).
2. **KV cache policy.** `q8_0/q8_0` ≈ half the KV memory, near-lossless — the
   default choice when context doesn't fit at f16. Quantized V-cache *requires*
   flash attention. Avoid q4_0 unless desperate: it costs quality on reasoning
   models and gets *slower* at long context. Freed VRAM may now admit more
   layers/experts → **re-run stage 1**. This is the strongest interaction in
   the whole system.
3. **Batch sizes** (dominates pp, barely touches tg). Sweep
   `-ub 256,512,1024,2048` (with `-b ≥ -ub`). CPU-offloaded MoE usually wants
   large equal batches (`-b 4096 -ub 4096`); VRAM-tight dense setups sometimes
   want *small* `-ub` (documented cases both of 10× gains from raising it and
   from lowering it). Larger `-ub` eats VRAM — if the winner no longer fits,
   evaluate that trade explicitly.
4. **Threads** — only if any weights run on CPU (partial offload or
   `--n-cpu-moe`); on a fully-offloaded model `-t` is nearly irrelevant. Sweep
   around the *physical* core count (P-cores only on hybrid Intel; SMT and
   E-cores hurt generation 20–30%). `-tb` (prefill threads) may use all cores.
5. **Residual toggles**, one cross-product run: `-mmp 0/1` (hybrid CPU+GPU
   inference often wants `--no-mmap` or `--mlock` to stop page-cache eviction
   stalls; pure-GPU doesn't care), speculative decoding (dense models with a
   same-family small draft only — skip for MoE; check for `--mtp` support on
   models with MTP heads instead), multi-GPU `--split-mode` (see
   [references/flags.md](references/flags.md)).
6. **Validate.** Re-run winner vs. the default config back-to-back, 5+
   repetitions, at the user's real context depth (`llama-bench -d 4096` or
   deeper — depth-0 numbers overstate everything). Then confirm through the
   real serving path with one actual request. If the winner's re-run regressed,
   thermal drift contaminated the sweep — redo the affected stage.

## Step 5 — Deliver and record

- Write the final configuration in the user's runtime format (llama-server
  flags, Ollama Modelfile + env vars, LM Studio settings, llama-swap
  config.yaml — translation tables in [references/runtimes.md](references/runtimes.md)).
- Annotate every non-default flag with its measured justification, e.g.
  `# q8_0 KV freed 1.9 GB → +4 GPU layers → +31% tg`. Untraceable flags rot.
- Save the benchmark log (llama-bench `-o jsonl` output or bench-api.py output)
  next to the config, with model file, quant, llama.cpp build, and driver
  version — the next retune (after any upgrade) diffs against it.
- State what was NOT tuned and why, so nobody mistakes defaults for measured
  choices.
- Performance tuning is not serving setup: finish by pointing the user at the
  model's recommended sampling parameters, chat template (`--jinja` where the
  model card calls for it), and reasoning/thinking-mode switches. A perfectly
  tuned server with the wrong sampling settings still serves bad tokens.

## Rules that generalize vs. things you must measure

**Encode these (universal):**
- Generation speed ceiling ≈ memory bandwidth ÷ active-weight bytes × ~0.6.
  This one formula predicts tg within ~30% on CUDA, Metal, ROCm, and CPU, and
  tells you when a config is already near optimal — stop tuning.
- MoE strategy: `-ngl 99` + expert offload beats lowering `-ngl`. Always.
- Threads = physical cores; V-cache quant requires flash attention; prefer
  symmetric KV quant types (asymmetric silently hits a slow path on ROCm).
- Speculative decoding is a dense-model, big-target technique.
- Set `-c` explicitly, sized to real usage, never the model max. Sizing above
  stated usage is fine only when VRAM is clearly abundant — and say so
  explicitly when you do it.

**Never encode — sweep every time (machine-specific):**
- The exact `--n-cpu-moe N`, `-ub`/`-b` values, thread count, multi-GPU split.
- AMD: ROCm vs Vulkan (the winner flips by GPU and by release). Intel Arc:
  SYCL vs Vulkan. Benchmark both backends.
- Whether flash attention helps on non-NVIDIA or pre-Ampere hardware.

## Pitfalls that produce wrong conclusions

| Pitfall | Symptom / rule |
|---|---|
| Windows NVIDIA sysmem fallback | VRAM overflow silently spills to RAM: model "loads fine", runs 10× slow. Detect by tok/s cliff, not by OOM errors. Fix: "Prefer No Sysmem Fallback" in NVIDIA Control Panel, then reduce offload |
| Cold first run | mmap page-in + cold GPU clocks skew results 12–18%. One discarded warm-up run before every sweep |
| Deterministic benchmark prompts | Hit EOS after 20 tokens → absurd tok/s. Fix `n_predict`, use `ignore_eos`, or a generative prompt |
| Prompt caching during pp measurement | `cache_prompt: true` (or prefix reuse) silently inflates pp numbers |
| Benchmarking only at depth 0 | Overstates real performance; a config that wins at 0 can lose at 32k. Use `-d` at the user's working depth |
| Background load | Browsers/indexers subtract from CPU-bound tg nearly 1:1. Check load first; quiet machine or don't trust the numbers |
| Thermal drift across a sweep | Later configs unfairly penalized. `--delay` between tests; re-run the winner at the end |
| Differences under ~3–5% | Usually noise (stddev on a quiet machine is 1–3%). Prefer the simpler config on a tie |
| Ollama silent context truncation | Past `num_ctx` the *head* of the prompt (system prompt) is dropped with no error |
| Apple Silicon "doesn't fit" | macOS caps GPU-wired memory at ~65–75% of RAM; raise `iogpu.wired_limit_mb` before giving up — see [references/platforms.md](references/platforms.md) |

## References (read when needed)

- [references/flags.md](references/flags.md) — per-flag deep dive: what each flag does, on which hardware it helps or hurts, evidence. Read when deciding a stage's sweep values or debugging a flag's effect.
- [references/benchmarking.md](references/benchmarking.md) — exact llama-bench sweep commands, server-path validation, statistics, and benchmarking without llama-bench. Read before running any sweep.
- [references/runtimes.md](references/runtimes.md) — per-runtime tunability (Ollama, LM Studio, koboldcpp, Jan, llama-swap, llamafile), setting translation tables, where GGUF files live on disk. Read when the user's runtime isn't raw llama.cpp.
- [references/platforms.md](references/platforms.md) — hardware detection commands per OS/vendor and platform gotchas (Windows, WSL2, macOS, AMD, iGPU, multi-GPU). Read during Step 2 on anything other than Linux+NVIDIA.

## Scripts (run, don't read)

- `scripts/gguf-meta.py <model.gguf>` — architecture facts + KV-cache budget from the GGUF header. Stdlib-only, reads the header only (fast on any file size). `--help` for options.
- `scripts/bench-api.py --url <endpoint>` — measures pp/tg through an HTTP API when llama-bench isn't available: exact timings from llama-server's `/completion`, approximate timings from any OpenAI-compatible endpoint (Ollama, LM Studio, Jan). `--help` for options.
