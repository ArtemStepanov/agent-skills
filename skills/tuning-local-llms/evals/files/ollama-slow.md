# Environment facts (command outputs)

## ollama ps
```
NAME                    ID              SIZE      PROCESSOR          UNTIL
qwen3:30b-a3b-q4_K_M    a1b2c3d4e5f6    21 GB     42%/58% CPU/GPU    4 minutes from now
```

## ollama show qwen3:30b-a3b-q4_K_M --modelfile (excerpt)
```
FROM /usr/share/ollama/.ollama/models/blobs/sha256-8f2a...
PARAMETER num_ctx 4096
```

## nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```
name, memory.total [MiB], memory.free [MiB]
NVIDIA GeForce RTX 4070, 12282 MiB, 631 MiB
```
(with the model loaded by Ollama)

## ollama run qwen3:30b-a3b-q4_K_M --verbose (tail of a test run)
```
total duration:       1m2.481s
prompt eval count:    1024 token(s)
prompt eval rate:     311.42 tokens/s
eval count:           412 token(s)
eval rate:            9.83 tokens/s
```

## env | grep OLLAMA
```
(no output — no OLLAMA_* variables set)
```

## command -v llama-bench llama-server koboldcpp
```
(no output — none installed)
```

User notes: runs Open WebUI against Ollama; also noticed the assistant
"forgets" instructions from the start of long conversations.
