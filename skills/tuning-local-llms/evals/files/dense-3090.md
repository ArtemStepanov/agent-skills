# Environment facts (command outputs)

## nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```
name, memory.total [MiB], memory.free [MiB]
NVIDIA GeForce RTX 3090, 24576 MiB, 23801 MiB
```

## Model
Llama-3.1-8B-Instruct Q4_K_M, 4.9 GB on disk.
Dense: 32 layers, 8 KV heads, head_dim 128, trained ctx 131072.

## lscpu (excerpt)
```
Model name:          AMD Ryzen 7 5800X 8-Core Processor
Core(s) per socket:  8
```

## command -v llama-server llama-bench
```
/usr/local/bin/llama-server
/usr/local/bin/llama-bench
```

User goal: interactive chat assistant, typical conversations under 8k
tokens. Quote: "don't spend an hour benchmarking, just give me good
settings and tell me if anything is worth measuring later."
