# llama.cpp flag reference: what generalizes, what doesn't

Contents: [Defaults changed in 2025–2026](#defaults) · [GPU offload](#gpu-offload) ·
[MoE expert offload](#moe-offload) · [Batch sizes](#batch-sizes) · [Flash attention](#flash-attention) ·
[KV cache quantization](#kv-quant) · [Context](#context) · [Threads](#threads) ·
[mmap/mlock](#mmap) · [Multi-GPU](#multi-gpu) · [Speculative decoding & MTP](#speculative) ·
[Vendor cheat sheet](#vendors)

Each entry states the rule, then its hardware-dependence class:
**universal** (encode it) / **vendor-specific** (branch on it) / **must-measure** (sweep it).

<a name="defaults"></a>
## Defaults that changed in 2025–2026 (stale-advice guard)

Old guides say "always pass `-ngl 99 -fa`". Since late 2025, llama.cpp defaults to
`-ngl auto` + `--fit on` (probes free VRAM, computes placement; `--fit-target` headroom
defaults to 1024 MiB) and `-fa auto`. `llama-fit-params` prints the derived parameters.
So the modern job is: (a) know when auto-fit's choice is suboptimal — it optimizes for
*fitting*, not throughput, and MoE placement usually beats it manually; (b) tune the
flags auto-fit does NOT touch: batch sizes, threads, KV quant, split mode, speculative
decoding. Also: `-c 0` now means the model's full trained context (VRAM bomb — always
set `-c`), and `--fit-target` below ~512 MiB OOMs long-running servers.

<a name="gpu-offload"></a>
## `-ngl` (GPU layers) — universal direction, measured value

Offload as many layers as fit; a single CPU-resident layer serializes the pipeline, so
gains are superlinear. Whole model fits → `-ngl 99`. Doesn't fit → dense: lower `-ngl`
(binary-search the max that loads with headroom); MoE: don't lower it — see next entry.
How much a partial offload hurts scales with CPU memory bandwidth and PCIe speed.

<a name="moe-offload"></a>
## `--cpu-moe` / `--n-cpu-moe N` / `-ot` — universal strategy, must-measure N

For MoE models that don't fully fit: `-ngl 99 --n-cpu-moe N`. Keeps attention, dense
FFN, shared experts, and KV cache on GPU (touched every token) while routed experts
(touched sparsely) live in RAM. Typically 2–4× faster than lowering `-ngl`.
`--cpu-moe` = all experts on CPU; `--n-cpu-moe N` moves the experts of the top N layers.
Sweep N downward until VRAM is full — but the optimum is sometimes non-monotonic
(more CPU experts occasionally wins via CPU/GPU overlap), so measure, don't assume
minimum-N-is-best. `-ot "exps=CPU"` (`--override-tensor` regex) is the general form for
finer placement (per-layer, per-GPU). GPU priority order (universal, by per-token access
frequency): attention + KV cache > shared experts + dense FFN > routed experts.
Canonical guide: https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide

<a name="batch-sizes"></a>
## `-b` / `-ub` (logical / physical batch) — must-measure

Affects prompt processing only, not generation. Defaults `-b 2048 -ub 512`. Raising
`-ub` speeds prefill at the cost of larger compute buffers (VRAM); in VRAM-tight dense
setups a big `-ub` can cost a GPU layer and thus *lower* tg. CPU-offloaded MoE wants
large equal batches (`-b 4096 -ub 4096`) — prefill batches big enough to justify the
PCIe transfer run experts on GPU. Documented extremes in both directions (59→582 pp
tok/s from *lowering* `-ub` to 64 on one machine; 2× prefill from *raising* it on
another). Always sweep; sweep at the already-chosen memory layout.

<a name="flash-attention"></a>
## `-fa` (flash attention) — vendor-specific

Leave on `auto`. Cuts attention memory, speeds prefill, and is a **prerequisite for
quantized V-cache**.
- NVIDIA Ampere+ (RTX 30/40/50): always beneficial (measured 2× tg at 8k ctx vs fallback).
- NVIDIA Pascal/Turing: fallback kernels, limited head sizes, long-context bugs — A/B test.
- AMD Vulkan: historically catastrophic (CPU-fallback path); improved, still A/B test.
- AMD ROCm: good only if built with rocWMMA, and K/V quant types must be **symmetric**
  (`-ctk q8_0 -ctv q4_0` silently falls to a slow path on HIP).
- Apple Metal: supported, beneficial, `auto` handles it.

<a name="kv-quant"></a>
## `-ctk` / `-ctv` (KV cache quantization) — universal math, measured speed

`q8_0/q8_0`: ~50% KV memory saved, near-lossless (perplexity deltas 0.002–0.05), keeps
the fast fused-FA path. The default choice when target context doesn't fit at f16.
`q4_0`: ~72–75% saved but measurable quality cost (worst on long-chain reasoning
models) **and a speed penalty that grows with depth** (measured: equal to f16 at 6k,
−12% at 24k, −37% at 110k, from per-token dequant). Last resort only; if used, re-bench
tg at the real depth. V is more quality-sensitive than K. Quantized V requires `-fa`;
without FA, quantized KV can be *slower* than f16. Freed KV memory may admit more GPU
layers — always redo the memory-layout stage after changing KV policy.

<a name="context"></a>
## `-c` (context) and sliding-window models — universal

Pure memory lever: size to the largest prompt actually served, never the model max.
Sliding-window-attention models (Gemma-class hybrids) keep only window-sized cache on
SWA layers — much cheaper than the naive formula; do not pass `--swa-full` (inflates
memory for a cache-restoration edge case). MLA models (DeepSeek lineage) use compressed
latent KV — the standard formula overestimates badly. `--cache-reuse N` enables prefix
cache trimming for agentic workloads but doesn't work on some hybrid-attention models.

<a name="threads"></a>
## `-t` / `-tb` (threads) — universal rule, machine-specific value

Only matters when weights run on CPU; nearly irrelevant fully-offloaded (small values
can even win — GPU-only case study improved by *lowering* threads). Rule: `-t` =
physical cores, minus 1–2 for the OS; SMT/hyperthreads hurt generation. Intel hybrid
CPUs: P-cores only (classic measurement: 3× faster with 8 P-cores than all 16 threads
on i5-12600K); E-cores drag the ring at the slowest core. `-tb` (prefill) is
compute-bound, not bandwidth-bound — can use all cores including E-cores. AMD dual-CCD
(Zen4/5): mild analog — prefer one CCD for small models. Generation on CPU is
memory-bandwidth-bound: `tg ≈ bandwidth ÷ active-weight-bytes × 0.5–0.7` — if measured
tg is already near that ceiling, threads won't help.

<a name="mmap"></a>
## `--mmap` / `--no-mmap` / `--mlock` — situation-specific, portable decision rule

Pure-GPU or ample-RAM: default mmap is fine, and it makes model *load/swap* far faster
(pages in lazily). Hybrid CPU+GPU (especially MoE experts in RAM): page-cache eviction
causes erratic tg — if CPU-resident weights fit in RAM, `--no-mmap` or `--mlock`.
Model bigger than RAM: mmap is mandatory (disk-bound). Comparing mmap on/off is
confounded by the OS page cache — compare steady-state, and measure load time
separately if the user swaps models often. ROCm + `--mlock` has a known GTT/shared
memory interaction — verify on AMD.

<a name="multi-gpu"></a>
## `--split-mode` / `--tensor-split` — must-measure, interconnect-dependent

`layer` (default): whole layers per GPU; pools VRAM, GPUs take turns → tg ≈ single-GPU
speed. `row`: legacy tensor-split; can help pp with NVLink, usually hurts on consumer
PCIe. If the model fits on one GPU, single-device is fastest. `--tensor-split 3,2`
rebalances asymmetric VRAM (e.g. one card drives displays). Newer tensor-parallel modes
(ik_llama.cpp `-sm graph`) report 3–4× where interconnect allows — fork territory.
Official doc: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md

<a name="speculative"></a>
## Speculative decoding (`-md`) and MTP (`--mtp`) — universal verdicts

Draft-model speculation works because generation is bandwidth-bound: verifying k
drafted tokens ≈ one target pass. Needs a same-family draft with matching vocab,
~10–20× smaller; biggest wins on code/low-entropy text; draft weights + KV come out of
the VRAM budget (`-ngld`/`-devd` place it). **Skip for MoE**: acceptance rates drop
(distribution mismatch) and the margin is small anyway (MoE decode already costs only
its *active* params — a 30B-A3B decodes nearly as cheaply as the draft). For MoE speed,
check MTP instead: `--mtp` (merged May 2026, feature-flagged) uses prediction heads
baked into the checkpoint (Qwen3.5+/DeepSeek V3/R1 lineage, GLM-4.6) — no draft model,
works on MoE, reported 1.7–2.4× decode. Check whether the GGUF has MTP heads before
reaching for `-md`.

<a name="vendors"></a>
## Vendor cheat sheet

- **NVIDIA Ampere+**: everything works; FA on; tg tracks memory bandwidth across cards.
- **NVIDIA Pascal/Turing**: A/B flash attention; expect long-context corner cases.
- **AMD**: benchmark BOTH ROCm and Vulkan — the winner flips by GPU and by month.
  ROCm needs the card on the support matrix + rocWMMA build + symmetric KV types.
  Vulkan runs on nearly everything (Windows included) at typically 80–90% of ROCm.
  Never tell a Windows AMD user to install ROCm first.
- **Intel Arc**: SYCL vs Vulkan also flips (SYCL often faster overall, loses on MoE).
- **Apple Silicon**: `-ngl 99`, FA on, don't over-thread (P-cores only), raise
  `iogpu.wired_limit_mb` for big models, keep total under ~85–90% of unified RAM.
  Speculative decoding works well. No backend choice to make.
- **iGPU/APU**: offloading mostly helps pp (tg is bound by the same shared DRAM);
  "VRAM" is carved from system RAM — don't double-count it in the budget.
- **CPU-only**: quant choice dominates (Q4_K_M/IQ4 class); physical cores; MoE models
  are the headline (bandwidth ÷ *active* params makes 30B-A3B-class usable).
