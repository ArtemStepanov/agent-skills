# Platforms: hardware detection and per-OS gotchas

Contents: [GPU + VRAM detection](#gpu) · [CPU topology](#cpu) · [RAM and bandwidth](#ram) ·
[Windows](#windows) · [WSL2](#wsl2) · [macOS / Apple Silicon](#macos) · [AMD](#amd) ·
[Integrated GPUs](#igpu) · [CPU contention](#contention) · [Installing tools](#installing)

All commands below run without sudo/admin unless noted.

<a name="gpu"></a>
## GPU model + VRAM

| Vendor / OS | Command |
|---|---|
| NVIDIA (Linux/Windows/WSL2) | `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv` |
| AMD (Linux, ROCm installed) | `rocm-smi --showmeminfo vram` |
| AMD (Linux, no ROCm) | `cat /sys/class/drm/card*/device/mem_info_vram_total` |
| Any vendor, any OS | `vulkaninfo --summary` for device names; full `vulkaninfo` → DEVICE_LOCAL `VkMemoryHeap` size = VRAM. llama.cpp's own device listing at startup is also a reliable probe |
| Linux generic | `lspci | grep -Ei 'vga|3d|display'`; `glxinfo -B` shows "Video memory" |
| macOS | `system_profiler SPDisplaysDataType` (GPU + core count). Apple Silicon has no VRAM number — it's unified memory; use total RAM and see the wired-limit note below |
| Windows | `nvidia-smi` if NVIDIA. Do NOT trust `Win32_VideoController.AdapterRAM` — uint32, caps at 4GB. Real value: registry `HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\00XX\HardwareInformation.qwMemorySize`, or Task Manager "Dedicated GPU memory" |

Always use **free** VRAM, not total: a running desktop session takes 0.5–2 GB, and the
budget must survive the user's normal workload (browser with GPU acceleration, second
monitor), not a freshly rebooted machine.

<a name="cpu"></a>
## CPU topology (the `-t` input)

| OS | Command |
|---|---|
| Linux | `lscpu` and `lscpu -e` — on Intel hybrid, P-cores show higher MAXMHZ; count physical cores, not threads |
| macOS | `sysctl hw.perflevel0.physicalcpu` (P-cores), `hw.perflevel1.physicalcpu` (E-cores) |
| Windows | `Get-CimInstance Win32_Processor | Select NumberOfCores,NumberOfLogicalProcessors`; P/E split needs Sysinternals CoreInfo or the CPU model's spec sheet |

Rule: threads = physical P-cores. Recent llama.cpp auto-detects this
(`common_cpu_get_num_math()` excludes E-cores), but verify — wrappers and older
embedded builds (llamafile) have gotten it wrong.

<a name="ram"></a>
## RAM and memory-bandwidth class

Total RAM: `free -g` (Linux), `sysctl hw.memsize` (macOS),
`Get-CimInstance Win32_ComputerSystem` (Windows).

Bandwidth matters because CPU-side generation is bandwidth-bound
(`tg ≈ bandwidth ÷ active-weight-bytes × ~0.6`). Reading DIMM type needs sudo on Linux
(`dmidecode -t 17`); Windows: `Get-CimInstance Win32_PhysicalMemory | Select
SMBIOSMemoryType,Speed` (26=DDR4, 34=DDR5), no admin. Pragmatic fallback — infer the
class: dual-channel DDR4 ≈ 50 GB/s, dual-channel DDR5 ≈ 80–100 GB/s (Intel 12th-gen+ /
AMD Zen4+), Apple M-series by chip name: base ≈ 100, Pro ≈ 200–273, Max ≈ 400–546,
Ultra ≈ 800 GB/s. This one number sets expectations for any CPU-offloaded config
before you benchmark it.

<a name="windows"></a>
## Windows + NVIDIA: the sysmem-fallback trap

Since driver 536.40, VRAM overflow silently spills to system RAM: the model "loads
fine" and runs ~10× slower — **no OOM error is ever raised**. A tuning loop that only
watches for crashes will mis-tune every Windows NVIDIA box. Detect via a tok/s cliff
between two adjacent offload values. Fix: NVIDIA Control Panel → CUDA — Sysmem
Fallback Policy → "Prefer No Sysmem Fallback", so overflow fails loudly; then tune
offload down until it loads.

<a name="wsl2"></a>
## WSL2

CUDA works (`nvidia-smi` inside WSL). Generation is ~90–100% of native (bandwidth-
bound); prompt processing and transfer-heavy configs can be 5–20% slower — native
Windows llama.cpp builds remove the layer. WSL2 reserves a chunk of host RAM (default
50%) — check `.wslconfig` before budgeting CPU-offloaded experts.

<a name="macos"></a>
## macOS / Apple Silicon

- Unified memory: no `-ngl` drama — use `-ngl 99` and manage the *total* footprint.
- macOS caps GPU-wired memory at ~65–75% of RAM. For big models raise it:
  `sudo sysctl iogpu.wired_limit_mb=N` (leave 8–16 GB for the OS; resets on reboot).
  Check this before concluding a model "doesn't fit" on a 32–128 GB Mac.
- Threads: P-cores only (`hw.perflevel0.physicalcpu`); E-cores actively hurt.
- macOS ships bash 3.2 — any bundled shell script must avoid bash-4isms
  (no `mapfile`, no associative arrays); BSD userland (`sed -i ''`, no `grep -P`).

<a name="amd"></a>
## AMD

Two backends, and the winner flips by GPU and release: **ROCm** (HIP) is Linux-first
with a narrow official card matrix and needs a rocWMMA build for good flash attention;
**Vulkan** runs on nearly everything including Windows, typically at 80–90% of ROCm's
speed. Default to Vulkan unless the card is on the ROCm support list; benchmark both
when both are available. Known iGPU corner cases (e.g. gfx1103/780M rocBLAS crashes)
where Vulkan is the fix. `--mlock` on ROCm has a GTT/shared-memory interaction —
verify before recommending.

<a name="igpu"></a>
## Integrated GPUs / APUs

iGPU "VRAM" is carved from system RAM — never double-count it in the budget.
Offloading to an iGPU mostly helps prompt processing (often 2×); generation stays
bound by the same shared DRAM, so expect tg unchanged. Memory channel count and speed
dominate everything else on these systems. Unified-memory toggles exist for Linux APUs
(`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — integrated only; it *hurts* discrete GPUs).

<a name="installing"></a>
## Installing tools (only when the degraded path isn't enough)

Ask before installing anything; state why the degraded path can't answer.

| Tool | Windows | macOS | Linux |
|---|---|---|---|
| llama.cpp (brings `llama-bench` + `llama-server`) | `winget install llama.cpp` | `brew install llama.cpp` | distro package where available, else prebuilt release binaries from github.com/ggml-org/llama.cpp/releases (pick the build matching the GPU backend: cuda / vulkan / cpu) |
| Python 3 (for bundled scripts) | `winget install Python.Python.3.12` | preinstalled / `brew install python` | preinstalled |
| `vulkaninfo` (cross-vendor VRAM probe) | bundled with GPU drivers or Vulkan SDK | — (use system_profiler) | `apt/dnf/pacman install vulkan-tools` |

Not needed: `nvidia-smi` ships with the NVIDIA driver (nothing to install);
`rocm-smi` is unnecessary for detection (sysfs fallback above); koboldcpp is a
single downloadable binary if the user prefers it over llama.cpp.

<a name="contention"></a>
## CPU contention (why production is slower than the benchmark)

CPU-bound generation loses to concurrent load nearly 1:1 — the IDE, VMs, and the very
browser streaming the model's reply all subtract. So quiet-machine benchmark numbers
are *ceilings*, not guarantees. Before blaming the config, check
`ps -eo pcpu,comm --sort=-pcpu | head` (Linux/macOS) or Task Manager. Mitigations if
inference must win under load: on Linux, run the server in a dedicated cgroup slice
with a high `CPUWeight` (in-process `nice` does NOT help across cgroup boundaries —
cgroup v2 splits CPU by slice weight first). A residual gap under load (SMT-sibling
sharing, DRAM bandwidth) is physics, not configuration.
