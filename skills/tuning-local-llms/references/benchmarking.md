# Benchmarking: instruments, sweep commands, hygiene, statistics

Contents: [Choosing an instrument](#instrument) · [llama-bench](#llama-bench) ·
[Server-path validation](#server) · [Without llama-bench](#fallbacks) ·
[Pre-flight hygiene](#hygiene) · [Deciding real vs noise](#statistics) ·
[Recording results](#recording)

<a name="instrument"></a>
## Choosing an instrument

1. **llama-bench** — the default. Native parameter sweeps, repetitions with stddev,
   machine-readable output. Use for all exploratory stages.
2. **koboldcpp `--benchmark [file.csv]`** — standardized pp/tg test, exits when done.
   Good scripted option when the user runs koboldcpp and won't install llama.cpp.
3. **`scripts/bench-api.py`** — timed HTTP requests. Exact timings against
   llama-server's `/completion`; approximate against any OpenAI-compatible endpoint
   (Ollama, LM Studio, Jan). Use when no bench binary exists, and for final validation
   through the real serving path.
4. **llama-batched-bench** — only for the multi-user goal: sweeps parallel sequences
   (`-npp` prompt sizes, `-ntg` gen sizes, `-npl 1,2,4,8`), reports per-slot and
   aggregate throughput and time-to-first-token.

llama-bench timings exclude tokenization/sampling and the HTTP path — slightly
optimistic, but tuning needs *relative* comparisons, so that's fine. Validate the final
winner through the real path anyway.

<a name="llama-bench"></a>
## llama-bench: the sweep engine

Test types: `pp` (prompt processing, default 512 tokens), `tg` (generation, default
128), `-pg <pp,tg>` (combined, closest to real use).

**Sweep syntax — the key feature:** every parameter takes comma-separated values or
ranges (`-ub 128-2048*2` doubles through the range), and llama-bench runs the **full
cross-product**. One invocation per tuning stage:

```sh
# Stage: MoE expert offload sweep at realistic depth
llama-bench -m model.gguf -ngl 99 --n-cpu-moe 12,16,20,24,30 -fa 1 -d 4096 -r 3 -o jsonl

# Stage: batch-size sweep
llama-bench -m model.gguf <winning layout so far> -ub 256,512,1024,2048 -p 2048 -r 3

# Stage: threads (only if CPU is in the path); P = physical/P-core count
llama-bench -m model.gguf <layout> -t P/2,P-2,P,P+2

# Final confirmation: winner vs default, more reps, at the user's working depth
llama-bench -m model.gguf <winner> -d 0,4096,16384 -r 5 -o jsonl
```

Flags that matter: `-d N` prefills N tokens of KV before measuring — **the only way to
see performance at realistic context depth**; depth-0 numbers overstate everything and
a config that wins at 0 can lose at 32k. `-r` repetitions (default 5; use 2–3
exploring, 5–10 confirming). `-o json|jsonl|csv|md` for parseable output (JSON includes
per-rep samples). `--prio 1..3` raises process priority. `--delay N` seconds between
tests for thermal recovery. `-mmp 0/1`, `-nkvo`, `-sm`, `-ts`, `-ctk/-ctv`, `-fa` are
all sweepable the same way.

Keep exploratory trials short (small `-p`/`-n`, `-r 2`) — save long runs for the final
incumbent-vs-default comparison.

<a name="server"></a>
## Server-path validation (llama-server `/completion`)

Every llama-server completion response carries a `timings` object: `prompt_per_second`
(pp), `predicted_per_second` (tg), `cache_n` (tokens reused from cache). Use this for
final validation, not sweeps (a server restart per config is slow). Hygiene:

- `"n_predict": <fixed N>` AND `"ignore_eos": true` — otherwise the model may stop
  after 20 tokens and you're timing noise.
- `"cache_prompt": false` when measuring pp — prefix reuse silently inflates it
  (check `cache_n` is 0).
- Thinking/reasoning models burn output budget on reasoning tokens — fixed `n_predict`
  + `ignore_eos` sidesteps this; never bench them with small chat `max_tokens`.
- A server that crashed during load can look like "still starting" to a naive health
  poll — also check the process is alive while waiting.
- To see why a config crashes under a wrapper (llama-swap, systemd, docker): run the
  exact command in the foreground; wrappers often hide the upstream stderr.

<a name="fallbacks"></a>
## Benchmarking without llama-bench

**Ollama:** `ollama run <model> --verbose` prints prompt-eval and eval rates — crude
but usable for A/B. `scripts/bench-api.py --url http://localhost:11434` gives
repeatable numbers. Between runs, settings changes need a Modelfile edit or env-var
restart (see runtimes.md). Verify actual layer placement with `ollama ps` (CPU/GPU
split %) — the `num_gpu` you asked for is a request, not a command.

**LM Studio / Jan:** tok/s shown per response in UI; the local server + bench-api.py
is the scriptable path. `lms load --estimate-only` (LM Studio) predicts VRAM fit.

**Any OpenAI-compatible endpoint:** bench-api.py's generic mode measures TTFT (pp
proxy) and streaming inter-token rate (tg). Label these *approximate* — they include
network + tokenization overhead and estimate the prompt token count.

<a name="hygiene"></a>
## Pre-flight hygiene (before every sweep)

1. **Quiet machine**: check for other VRAM consumers (`nvidia-smi`) and CPU load
   (browsers, indexers). CPU-bound generation loses to background load nearly 1:1.
   Laptops: on AC power, performance profile.
2. **Evict resident models**: another model in VRAM turns every candidate into an OOM
   crash. Ollama: `ollama stop <model>`. llama-swap: `curl <host>/unload` (GET).
   LM Studio: eject in UI or `lms unload`.
3. **Warm-up**: one discarded run before the sweep — first load pays mmap page-in from
   disk and cold GPU clocks; measured cold-vs-warm swing is 12–18%. A cold model load
   must never sit inside a timed window (unless load time is itself the metric —
   measure that separately if the user hot-swaps models).
4. **Thermal**: long sweeps heat-soak the GPU and penalize later configs — a
   config-*order* artifact. `--delay 5..30` between tests; short tests; re-run the
   winner and baseline back-to-back at the end. Winner regressed on re-run → the sweep
   was contaminated; redo that stage.

<a name="statistics"></a>
## Deciding real vs. noise

- Expected stddev on a quiet, thermally stable machine: **1–3%** of mean for GPU runs;
  CPU-offloaded runs are noisier. Stddev > ~5% is a red flag: check temps and
  background load, rerun.
- A difference is real when it exceeds ~2×√(s₁²+s₂²) (rough Welch criterion) AND is
  practically meaningful (≥3–5%). Under ~3% on consumer hardware: treat as a tie and
  prefer the simpler/safer config.
- Order effects: at minimum, re-measure winner and baseline back-to-back at the end of
  the session.

<a name="recording"></a>
## Recording results

Persist a machine-readable log next to the final config: model file + quant,
llama.cpp build/commit, GPU driver, full flag set, depth, pp/tg means ± stddev, date.
llama-bench `-o jsonl` output is ideal. The next retune — after any driver, build, or
hardware change — starts by diffing against this file instead of starting from zero.
