#!/usr/bin/env python3
"""Print architecture facts and KV-cache budget from a GGUF header.

Reads only the file header (fast on any file size), stdlib only.

Usage: gguf-meta.py <model.gguf> [more.gguf ...]

Prints per file: architecture, layer count, expert count (MoE vs dense),
attention heads / KV heads, head dim, trained context length, file size,
and the computed KV-cache cost per 1k tokens and at common context sizes
for f16 / q8_0 / q4_0 cache types.

Caveats printed when relevant:
- Sliding-window-attention models (Gemma-class): real KV cost is LOWER than
  computed (window-sized cache on SWA layers) — treat numbers as upper bound.
- MLA models (DeepSeek lineage): compressed latent KV — formula overestimates badly.
"""
import os
import struct
import sys

WANT_SUFFIXES = (
    'general.architecture',
    '.block_count',
    '.expert_count',
    '.expert_used_count',
    '.attention.head_count',
    '.attention.head_count_kv',
    '.attention.key_length',
    '.attention.sliding_window',
    '.context_length',
    '.embedding_length',
)


def read_gguf_meta(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError(f'{path}: not a GGUF file')
        struct.unpack('<I', f.read(4))          # version
        struct.unpack('<Q', f.read(8))          # n_tensors
        n_kv, = struct.unpack('<Q', f.read(8))

        def rs():
            n, = struct.unpack('<Q', f.read(8))
            return f.read(n).decode(errors='replace')

        def rv(t):
            scalars = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i',
                       6: '<f', 7: '<B', 10: '<Q', 11: '<q', 12: '<d'}
            if t in scalars:
                fmt = scalars[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            if t == 8:
                return rs()
            if t == 9:                           # array
                et, = struct.unpack('<I', f.read(4))
                n, = struct.unpack('<Q', f.read(8))
                return [rv(et) for _ in range(n)]
            raise ValueError(f'unknown gguf value type {t}')

        out = {}
        for _ in range(n_kv):
            k = rs()
            t, = struct.unpack('<I', f.read(4))
            v = rv(t)
            if k == 'general.architecture' or k.endswith(WANT_SUFFIXES[1:]):
                out[k] = v
        return out


def get(meta, suffix, default=None):
    for k, v in meta.items():
        if k.endswith(suffix):
            return v
    return default


def fmt_bytes(n):
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TiB'


def report(path):
    meta = read_gguf_meta(path)
    arch = meta.get('general.architecture', '?')
    layers = get(meta, '.block_count')
    heads = get(meta, '.attention.head_count')
    kv_heads = get(meta, '.attention.head_count_kv', heads)
    if isinstance(kv_heads, list):               # per-layer values on some hybrids
        kv_heads = max(v for v in kv_heads if v) if any(kv_heads) else heads
    embed = get(meta, '.embedding_length')
    head_dim = get(meta, '.attention.key_length')
    if not head_dim and embed and heads:
        head_dim = embed // heads
    experts = get(meta, '.expert_count', 0)
    ctx_max = get(meta, '.context_length')
    swa = get(meta, '.attention.sliding_window')
    size = os.path.getsize(path)

    print(f'== {os.path.basename(path)}')
    print(f'   arch {arch} | {layers} layers | '
          + (f'MoE: {experts} experts' if experts else 'dense')
          + f' | {heads} heads / {kv_heads} KV heads | head_dim {head_dim}'
          + f' | trained ctx {ctx_max}')
    print(f'   file size {fmt_bytes(size)} (= weights VRAM if fully offloaded)')

    if arch == 'clip':
        print('   vision projector (mmproj) — no KV cache; budget the file size '
              'as extra VRAM when offloaded (--no-mmproj-offload frees it but '
              'makes image encode very slow).')
        return
    if not all(isinstance(x, int) and x > 0 for x in (layers, kv_heads, head_dim)):
        print('   (missing header fields — cannot compute KV budget)')
        return

    # bytes per token = K and V (2) x layers x kv_heads x head_dim x bytes/element
    per_tok = {t: 2 * layers * kv_heads * head_dim * b
               for t, b in (('f16', 2.0), ('q8_0', 1.0625), ('q4_0', 0.5625))}
    print(f'   KV cache per 1k tokens: '
          + ' | '.join(f'{t} {fmt_bytes(v * 1024)}' for t, v in per_tok.items()))
    ctxs = sorted({8192, 32768, ctx_max if isinstance(ctx_max, int) else 32768})
    for c in ctxs:
        row = ' | '.join(f'{t} {fmt_bytes(v * c)}' for t, v in per_tok.items())
        print(f'   KV @ ctx {c}: {row}')
    if swa:
        print(f'   NOTE: sliding-window attention (window {swa}) — real KV cost is '
              'lower than computed on SWA layers; treat the above as an upper bound.')
    if 'deepseek' in str(arch).lower():
        print('   NOTE: MLA architecture — compressed latent KV; the formula above '
              'overestimates badly. Trust the llama.cpp load log instead.')
    if experts:
        print('   MoE tuning: keep -ngl 99 and sweep --n-cpu-moe instead of '
              'lowering -ngl. Expert weights offloaded to CPU stay in system RAM.')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)
    for p in sys.argv[1:]:
        try:
            report(p)
        except Exception as e:
            print(f'== {p}: ERROR: {e}')
