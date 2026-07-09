# tuning-local-llms

An [Agent Skill](https://agentskills.io) that finds the fastest working
configuration for running a local LLM (GGUF / llama.cpp family) on **your**
hardware — by measuring, not guessing.

Community advice ("just use `-ngl 99 -fa`") is directional at best and often
stale. This skill teaches the agent a benchmark-driven workflow:

1. **Ask your goal** — chat, coding/RAG, max context, min memory — because the
   goal changes what "optimal" means.
2. **Discover the environment** — GPU/VRAM, CPU topology, RAM, runtime
   (llama.cpp, Ollama, LM Studio, koboldcpp, Jan, llama-swap), and the model's
   architecture read straight from the GGUF header.
3. **Compute a memory budget analytically** (KV-cache math, weights, buffers)
   to seed the search — then verify with a real load.
4. **Tune in measured stages** — GPU/expert offload → KV cache policy → batch
   sizes → threads → extras — one variable at a time, with warm-ups, thermal
   hygiene, and a noise threshold, using `llama-bench` sweeps (or the bundled
   HTTP benchmark when llama-bench isn't available).
5. **Deliver the config in your runtime's format**, every flag annotated with
   the measurement that justified it.

Works across NVIDIA / AMD / Apple Silicon / Intel / CPU-only, on Linux, macOS,
Windows, and WSL2 — including the traps that silently ruin tuning (Windows
sysmem fallback, Ollama context truncation, Apple's GPU wired-memory limit).

## Requirements

Hard requirements: none beyond a shell — the skill checks what's present on
your machine and adapts. Recommended for the full experience:

- **Python 3** — for the two bundled scripts (both have documented manual
  fallbacks if you don't have it)
- **llama.cpp binaries** (`llama-bench`/`llama-server`) — for proper parameter
  sweeps; without them the skill benchmarks through your existing runtime's
  HTTP API (Ollama, LM Studio, Jan) or koboldcpp's built-in benchmark
- GPU detection uses whatever your system already has (`nvidia-smi` ships with
  the NVIDIA driver; AMD/Apple/Windows paths need nothing extra)

The skill will tell you if installing something would materially improve the
tuning, and asks before suggesting it.

## Install

**Claude Code (personal):**

```sh
git clone <this-repo> && cp -r <this-repo>/skills/tuning-local-llms ~/.claude/skills/
```

**Claude Code (project):** copy the folder into your repo's `.claude/skills/`.

Other agentskills.io-compatible runtimes: copy the folder into that tool's
skills directory.

## Try it

- "My local model feels slow, can you tune it?"
- "I just downloaded Qwen3-30B Q4_K_M — what settings should I run it with?"
- "This model OOMs on my 12GB card, help"
- "What's the biggest context I can fit at 15+ tok/s?"

## Contents

- `SKILL.md` — the workflow (loaded on activation)
- `references/` — per-flag knowledge, benchmarking methodology, runtime
  translation tables, platform detection & gotchas (loaded on demand)
- `scripts/gguf-meta.py` — architecture facts + KV budget from a GGUF header
  (stdlib-only, reads the header only)
- `scripts/bench-api.py` — pp/tg benchmark over HTTP for runtimes without
  llama-bench (stdlib-only)

## License

MIT — see [LICENSE](LICENSE).
