# Environment facts (command outputs, Windows 11)

## where python && where python3 && where py
```
INFO: Could not find files for the given pattern(s).
```
(no Python installed)

## where llama-bench && where llama-server && where koboldcpp
```
INFO: Could not find files for the given pattern(s).
```

## nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```
name, memory.total [MiB], memory.free [MiB]
NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 7350 MiB
```
(before loading the model)

## Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors
```
Name                                   NumberOfCores NumberOfLogicalProcessors
----                                   ------------- -------------------------
13th Gen Intel(R) Core(TM) i5-13500H   12            16
```
(4 P-cores + 8 E-cores per Intel spec)

## Total RAM
16 GB.

## ollama ps (model loaded)
```
NAME                          ID              SIZE     PROCESSOR          UNTIL
qwen2.5:14b-instruct-q4_K_M   f1c9e3d2a8b7    10 GB    28%/72% CPU/GPU    4 minutes from now
```

## ollama run qwen2.5:14b-instruct-q4_K_M --verbose (tail of a test run)
```
prompt eval count:    512 token(s)
prompt eval rate:     198.77 tokens/s
eval count:           256 token(s)
eval rate:            6.84 tokens/s
```

## env | findstr OLLAMA
```
(no OLLAMA_* variables set)
```

Model facts (from the Ollama library page): Qwen2.5-14B, dense, 48 layers,
8 KV heads, head_dim 128, GGUF Q4_K_M ~9.0 GB.

User is not a developer: no Python, no compilers, no package managers
configured. Uses Ollama with a chat UI. Conversations are ordinary chats,
a few thousand tokens.
