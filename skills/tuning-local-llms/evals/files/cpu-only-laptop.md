# Environment facts (command outputs, Linux laptop)

## lspci | grep -Ei 'vga|3d|display'
```
00:02.0 VGA compatible controller: Intel Corporation Alder Lake-P GT2 [Iris Xe Graphics] (rev 0c)
```
(no discrete GPU)

## lscpu (excerpt)
```
Model name:            12th Gen Intel(R) Core(TM) i7-1260P
Thread(s) per core:    2
Core(s) per socket:    12
```
(4 P-cores + 8 E-cores per Intel spec; 16 logical CPUs total)

## free -g
```
               total        used        free      shared  buff/cache   available
Mem:              31           5           3           0          22          25
```
RAM: 32 GB LPDDR5-5200, dual channel (~80 GB/s theoretical).

## Model the user wants
Qwen3-30B-A3B Q4_K_M (MoE, 3B active params), 17.3 GB on disk.
48 layers, 4 KV heads, head_dim 128, trained ctx 262144.

## command -v llama-server llama-bench
```
/usr/bin/llama-server
/usr/bin/llama-bench
```
llama.cpp CPU build (no Vulkan/SYCL compiled in).

User goal: private chat assistant, conversations under 4k tokens, on
battery sometimes. Asks: "is this model even a sane choice without a GPU,
and what settings should I use?"
