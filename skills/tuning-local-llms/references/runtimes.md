# Runtimes: tunability, setting translation, model reuse

Contents: [Tunability matrix](#matrix) · [Ollama](#ollama) · [LM Studio](#lm-studio) ·
[koboldcpp](#koboldcpp) · [Jan](#jan) · [llama-swap](#llama-swap) · [llamafile](#llamafile) ·
[Where GGUFs live on disk](#storage)

<a name="matrix"></a>
## Tunability matrix

| Runtime | Raw llama.cpp flags? | Benchmark? | Verdict |
|---|---|---|---|
| llama.cpp (llama-server/cli) | Yes — it IS llama.cpp | llama-bench | Gold standard; tune here |
| llama-swap | Yes — `cmd:` is a verbatim llama-server command | via llama-server timings | Fully tunable |
| koboldcpp | Mostly (renamed flags) | built-in `--benchmark` | Directly tunable, self-benchmarking |
| LM Studio | No, but exposes the ~8 flags that matter, per model | tok/s in UI; API + bench-api.py | Tunable for the basics; settings map 1:1 |
| Jan | No; per-model overrides for the main knobs | tok/s in UI; API + bench-api.py | Same story as LM Studio |
| Ollama | No | `--verbose` eval rates; API | Partially tunable; auto-offload overrides you |
| llamafile | Yes, but embeds an *older* llama.cpp | timings printed | Treat as "llama.cpp with lag"; new flags may not exist |

**Strategy for locked-down runtimes (Ollama/LM Studio/Jan):** benchmark with a raw
llama.cpp binary (or koboldcpp) against the *same GGUF file*, then port the winning
settings back using the tables below. Be explicit with the user about which winners
cannot be ported. If the user won't install anything, tune the exposed knobs directly
with bench-api.py as the instrument — slower, still worthwhile.

<a name="ollama"></a>
## Ollama

Per-model (Modelfile `PARAMETER` or API `options`): `num_gpu` (≈ `-ngl`, but it's a
*request* — Ollama's estimator makes the final call; verify actual placement with
`ollama ps`), `num_ctx` (≈ `-c`), `num_thread` (≈ `-t`), `num_batch` (≈ `-b`).
Whenever `ollama ps` shows a CPU split (any `N%/M% CPU/GPU`), `num_thread` belongs
in the plan: set it to the physical P-core count — the CPU-resident share is what
the user is waiting on, and default threading over logical/E-cores drags it.

Global only (env vars on the server process): `OLLAMA_FLASH_ATTENTION=1`,
`OLLAMA_KV_CACHE_TYPE=q8_0` (requires FA; applies to ALL models including embedders —
a known quality footgun), `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_NUM_PARALLEL`,
`OLLAMA_KEEP_ALIVE`, `OLLAMA_SCHED_SPREAD`, `OLLAMA_MODELS`.

**Cannot be ported into Ollama:** `-ub`, `--override-tensor` / fine MoE placement,
per-model KV cache types, `--split-mode`, speculative decoding. Its auto-offload
heuristics are known to misestimate MoE and multi-GPU splits (~2× off in reported
cases), silently spilling to host RAM. If the tuned config depends on those, the
honest recommendation is llama-server (or llama-swap for multi-model convenience) for
that model.

**Gotchas:** silent head-truncation past `num_ctx` — the system prompt is dropped with
no error; historically small context defaults; flash attention default varies by model
family and version.

<a name="lm-studio"></a>
## LM Studio

Per-model load settings (UI or `lms load`): GPU offload layers, context length, batch
size, flash attention, KV cache quant (K and V separately), KV offload toggle, mmap,
threads, MoE expert offload; runtime selection (CUDA/Vulkan/ROCm/Metal — on AMD, try
both Vulkan and ROCm runtimes). Names map 1:1 to llama.cpp, so llama-bench winners
port cleanly. `lms load --estimate-only` predicts VRAM fit. Existing per-model configs
(a working starting point if the user already ran the model) live under
`~/.lmstudio/.internal/user-concrete-model-default-config/`.
Not exposed: `-ub` separately from `-b`, `-ot` regexes, speculative decoding drafts
(version-dependent).

<a name="koboldcpp"></a>
## koboldcpp

llama.cpp fork with renamed flags: `--gpulayers` (= `-ngl`; `-1` = auto),
`--contextsize`, `--flashattention`, `--quantkv` (0=f16, 1=q8, 2=q4 — K and V
together), `--tensor_split`, backend pickers `--usecublas/--usevulkan/--useclblast`,
`--threads`, `--blasbatchsize` (≈ `-b`). Built-in `--benchmark [out.csv]` runs a
standardized pp/tg pass and exits — usable as the measurement instrument itself.

<a name="jan"></a>
## Jan

Per-model overrides: ctx size, GPU layers, batch size, flash attention, KV offload;
engine-wide llama.cpp settings. Uses llama.cpp underneath — llama-bench numbers
transfer. Same porting limits as LM Studio.

<a name="llama-swap"></a>
## llama-swap

A model-swapping proxy over llama-server: each model's `cmd:` in `config.yaml` is a
verbatim llama-server command line, so **everything** tunes. Notes: `ttl` controls
swap-out; `persistent: true` only prevents swap-out — loading is always lazy, so an
always-on model additionally needs a startup preload hook; `groups` control which
models may coexist (budget VRAM for the *group*, not one model). Keep mmap ON for
frequently swapped models (2–5s swap-in vs 15s+), and evict via `GET /unload` before
benchmarking on a spare port.

<a name="llamafile"></a>
## llamafile

Accepts llama.cpp-style flags but tracks an older llama.cpp commit — verify a flag
exists (`--help`) before recommending it; don't assume `--n-cpu-moe`, `--fit`, or
new `-fa` semantics. Its thread auto-detection has had hybrid-CPU regressions — set
`-t` explicitly.

<a name="storage"></a>
## Where GGUFs live on disk (reuse, don't re-download)

| Tool | Location | Notes |
|---|---|---|
| Ollama | `~/.ollama/models/blobs/` (Linux service: `/usr/share/ollama/.ollama`; Windows `%USERPROFILE%\.ollama`) | Blobs are plain GGUFs under SHA-256 names — llama.cpp opens them directly. Find the weights: `ollama show <model> --modelfile`, or take the largest blob |
| LM Studio | `~/.lmstudio/models/<publisher>/<repo>/` | Plain GGUFs, human-readable layout; symlinks work |
| HuggingFace cache | `~/.cache/huggingface/hub/` (`HF_HOME`) | Snapshots contain plain GGUFs |
| llama.cpp `-hf` | `~/.cache/llama.cpp/` | Plain GGUFs |
| Jan | Jan data folder (Settings → Advanced) | Plain GGUFs |
| koboldcpp / llamafile | none — you pass a path | Consume anything above |

Importing a local GGUF into Ollama via a Modelfile `FROM /path` creates a hard link on
the same filesystem — no duplication. For cross-tool setups, one shared models
directory + symlinks beats three copies of a 20GB file.
