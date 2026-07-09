# Environment facts (command outputs)

## nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
```
name, memory.total [MiB], memory.free [MiB], driver_version
NVIDIA GeForce RTX 3060, 12288 MiB, 11742 MiB, 570.86.16
```

## lscpu (excerpt)
```
Architecture:            x86_64
Model name:              AMD Ryzen 5 7600 6-Core Processor
Thread(s) per core:      2
Core(s) per socket:      6
CPU max MHz:             5170.0000
```

## free -g
```
               total        used        free      shared  buff/cache   available
Mem:              31           4           2           0          24          26
```

## Model file
Qwen3-30B-A3B-Q4_K_M.gguf, 17.3 GiB on disk.
GGUF header (raw facts): arch qwen3moe, 48 layers, 128 experts (8 used per
token, ~3B active params), 32 attention heads, 4 KV heads, head_dim 128,
trained context 262144.

## command -v llama-server llama-bench
```
/usr/local/bin/llama-server
/usr/local/bin/llama-bench
```
llama.cpp build: b6790 (Jun 2026), CUDA backend.
