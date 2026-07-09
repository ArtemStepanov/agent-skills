#!/usr/bin/env python3
"""Benchmark pp (prompt processing) and tg (generation) through an HTTP API.

For when llama-bench isn't available: works against llama-server natively
(exact timings from /completion), or any OpenAI-compatible endpoint
(Ollama, LM Studio, Jan, llama-swap) with approximate timings. Stdlib only.

Usage:
  bench-api.py --url http://localhost:8080 [--model NAME] [-r 3] [--label "cfg A"]

Options:
  --url URL        server base URL (required)
  --model NAME     model name for OpenAI-compatible endpoints (required for those;
                   ignored by native llama-server mode)
  -r N             repetitions per measurement (default 3; first extra warm-up
                   run is always discarded)
  --pp-tokens N    approx prompt length in tokens for the pp test (default 1536)
  --tg-tokens N    tokens to generate for the tg test (default 200)
  --label TEXT     label printed with the result row (default: the URL)

Notes:
- Native mode sets n_predict + ignore_eos + cache_prompt:false — exact numbers
  from the server's own timings.
- OpenAI mode measures wall-clock TTFT (pp proxy) and streaming inter-token
  rate (tg); prompt token count is estimated (~4 chars/token). APPROXIMATE —
  fine for A/B between configs, not comparable with llama-bench numbers.
- Benchmark on a quiet machine; results are mean +/- stddev over -r runs.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# ~8 tokens of generative filler per repeat; long prompt for pp measurement
PP_FILLER = 'The quick brown fox jumps over the lazy dog near the river bank. '
TG_PROMPT = ('Write a detailed essay about the history of computing, covering '
             'mechanical calculators, the transistor, and the internet era.')


def post(url, body, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def detect_native(base):
    """True if this is llama-server with the native /completion endpoint."""
    try:
        with post(base + '/completion',
                  {'prompt': 'Hi', 'n_predict': 1, 'cache_prompt': False},
                  timeout=120) as resp:
            return 'timings' in json.load(resp)
    except Exception:
        return False


def native_run(base, prompt, n_predict):
    body = {'prompt': prompt, 'n_predict': n_predict, 'cache_prompt': False,
            'ignore_eos': True, 'temperature': 0}
    with post(base + '/completion', body) as resp:
        t = json.load(resp)['timings']
    return t['prompt_per_second'], t['predicted_per_second']


_nonce = [0]


def openai_run(base, model, prompt, max_tokens):
    """Returns (approx pp tok/s from TTFT, tg tok/s from streaming rate)."""
    # A unique first line defeats prefix caching (which would otherwise make
    # TTFT — and thus pp — meaninglessly fast on repeated identical prompts).
    _nonce[0] += 1
    prompt = f'Run {_nonce[0]} of an unrelated benchmark series follows.\n{prompt}'
    body = {'model': model, 'stream': True, 'max_tokens': max_tokens,
            'temperature': 0.7, 'cache_prompt': False,
            'messages': [{'role': 'user', 'content': prompt}]}
    t0 = time.monotonic()
    t_first = None
    n_chunks = 0
    with post(base + '/v1/chat/completions', body) as resp:
        for line in resp:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            chunk = json.loads(line[6:])
            delta = chunk.get('choices', [{}])[0].get('delta', {})
            if delta.get('content') or delta.get('reasoning_content'):
                if t_first is None:
                    t_first = time.monotonic()
                n_chunks += 1
    t_end = time.monotonic()
    if t_first is None or n_chunks < 2:
        raise RuntimeError('no streamed tokens received (model returned nothing?)')
    prompt_tokens_est = len(prompt) / 4
    pp = prompt_tokens_est / (t_first - t0) if t_first > t0 else float('inf')
    tg = (n_chunks - 1) / (t_end - t_first)
    return pp, tg


def measure(fn, reps):
    fn()                                        # warm-up, discarded
    pps, tgs = [], []
    for _ in range(reps):
        pp, tg = fn()
        pps.append(pp)
        tgs.append(tg)
    return pps, tgs


def mean_sd(xs):
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f'{m:.1f} +/- {sd:.1f}'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('-r', type=int, default=3, dest='reps')
    ap.add_argument('--pp-tokens', type=int, default=1536)
    ap.add_argument('--tg-tokens', type=int, default=200)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    base = args.url.rstrip('/')
    label = args.label or base

    pp_prompt = PP_FILLER * max(1, args.pp_tokens // 16)  # filler ~16 tokens each

    try:
        if detect_native(base):
            mode = 'native llama-server (exact)'
            pp_fn = lambda: native_run(base, pp_prompt, 8)
            tg_fn = lambda: native_run(base, TG_PROMPT, args.tg_tokens)
        else:
            if not args.model:
                sys.exit('error: --model is required for OpenAI-compatible endpoints '
                         '(list models: GET {}/v1/models)'.format(base))
            mode = 'OpenAI-compatible (approximate)'
            pp_fn = lambda: openai_run(base, args.model, pp_prompt, 8)
            tg_fn = lambda: openai_run(base, args.model, TG_PROMPT, args.tg_tokens)

        pps, _ = measure(pp_fn, args.reps)
        _, tgs = measure(tg_fn, args.reps)
    except urllib.error.URLError as e:
        sys.exit(f'error: cannot reach {base}: {e.reason}')

    print(f'{label} | mode: {mode} | pp {mean_sd(pps)} tok/s | '
          f'tg {mean_sd(tgs)} tok/s | r={args.reps}')
    if max(tgs) > 2000:
        print('WARNING: tg implausibly high — the model likely hit EOS instantly. '
              'Use a generative prompt or a runtime that honors ignore_eos.')
    if max(pps) > 20000:
        print('WARNING: pp implausibly high — the runtime probably served the '
              'prompt from cache; these pp numbers are not trustworthy.')


if __name__ == '__main__':
    main()
