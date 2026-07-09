# Environment facts (command outputs, Windows 11)

## Get-CimInstance Win32_VideoController | Select Name,AdapterRAM
```
Name                              AdapterRAM
----                              ----------
NVIDIA GeForce RTX 4060 Laptop GPU 4293918720
```

## nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
```
name, memory.total [MiB], memory.free [MiB], driver_version
NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 7411 MiB, 576.02
```

## Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors
```
Name                                    NumberOfCores NumberOfLogicalProcessors
----                                    ------------- -------------------------
13th Gen Intel(R) Core(TM) i7-13620H    10            16
```
(6 P-cores + 4 E-cores per Intel spec sheet)

## Model
Mistral-Small-24B-Instruct Q4_K_M, single GGUF, 13.6 GB on disk.
Dense architecture: 40 layers, 8 KV heads, head_dim 128, trained ctx 32768.

## User's llama-server command (from a YouTube guide "RTX 4060 settings")
```
llama-server -m mistral-small-24b-q4_k_m.gguf -ngl 99 -c 32768 --flash-attn on
```

## Observed behavior
Model loads with no errors. Task Manager shows GPU "Dedicated GPU memory"
pegged at 8.0/8.0 GB during generation. Generation speed: ~1.9 tok/s. The
YouTube guide shows ~28 tok/s on the same GPU with a smaller model.
