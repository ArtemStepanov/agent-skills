# Environment facts (command outputs, macOS 15.5)

## sysctl hw.memsize hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
```
hw.memsize: 68719476736
hw.perflevel0.physicalcpu: 8
hw.perflevel1.physicalcpu: 4
```

## system_profiler SPDisplaysDataType (excerpt)
```
Apple M2 Max:
  Chipset Model: Apple M2 Max
  Type: GPU
  Bus: Built-In
  Total Number of Cores: 38
```

## sysctl iogpu.wired_limit_mb
```
iogpu.wired_limit_mb: 0
```
(0 = macOS default cap, ~70-75% of RAM)

## Model
Llama-3.3-70B-Instruct Q4_K_M, 42.5 GB on disk.
Dense: 80 layers, 8 KV heads, head_dim 128, trained ctx 131072.

## Attempted command and failure
```
llama-server -m Llama-3.3-70B-Instruct-Q4_K_M.gguf -c 16384
...
ggml_metal_buffer_init: error: failed to allocate buffer, size = 10922.67 MiB
llama_model_load: error loading model: unable to allocate Metal buffer
```

User goal: interactive chat, wants at least 16k context. llama.cpp installed
via homebrew (recent build). 64 GB machine otherwise idle (~6 GB in use).
