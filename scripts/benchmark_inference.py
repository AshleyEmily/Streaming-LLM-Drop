"""
Inference benchmark for LLM-Drop pruned (or baseline) models.

Measures:
  - Prefill latency (time to first token)
  - Decode throughput (tokens/second)
  - Peak GPU memory during prefill
  - Peak GPU memory during decode
  - KV cache memory (decode peak - model weights)

Usage (standard):
    python scripts/benchmark_inference.py \
        --model_path path/to/checkpoint \
        --prompt_len 512 --gen_len 128 --runs 3

Usage (streaming-LLM with bounded KV cache):
    python scripts/benchmark_inference.py \
        --model_path path/to/checkpoint \
        --streaming --start_size 4 --recent_size 2000 \
        --prompt_len 512 --gen_len 4096 --runs 3

With --streaming the KV cache is evicted each step (physical eviction) and
attention uses position-shifted RoPE, matching live streaming-LLM inference.
Peak memory stays bounded at (start_size + recent_size) regardless of gen_len.
"""

import argparse
import sys
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def get_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2


def get_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def benchmark(
    model_path: str,
    prompt_len: int,
    gen_len: int,
    runs: int,
    device: str,
    streaming: bool = False,
    start_size: int = 4,
    recent_size: int = 2000,
):
    print(f"\nLoading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    torch.cuda.synchronize()

    # Memory occupied by model weights alone
    model_mem_mb = get_mem_mb()
    print(f"Model loaded: {model_mem_mb:.1f} MiB")

    # Optionally enable streaming-LLM inference (physical KV eviction + pos-shift RoPE)
    kv_cache = None
    if streaming:
        from llmtuner.compression.streaming_llm import enable_streaming_llm
        kv_cache = enable_streaming_llm(model, start_size=start_size, recent_size=recent_size)
        print(f"[streaming] start_size={start_size}  recent_size={recent_size}  "
              f"cache_bound={start_size + recent_size} tokens")

    # Build a dummy prompt of the requested length
    dummy_ids = torch.randint(100, 32000, (1, prompt_len), device=device)

    # ── Warmup (untimed) ─────────────────────────────────────────────────────
    with torch.no_grad():
        _out = model(dummy_ids, use_cache=True)
    torch.cuda.synchronize()
    del _out
    torch.cuda.empty_cache()

    prefill_times = []
    decode_times  = []
    prefill_peaks = []
    decode_peaks  = []
    kv_cache_mbs  = []

    for run in range(runs):
        torch.cuda.synchronize()

        # ── Prefill ──────────────────────────────────────────────────────────
        reset_peak()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model(dummy_ids, use_cache=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        prefill_times.append(t1 - t0)
        prefill_peaks.append(get_peak_mb())

        past_key_values = out.past_key_values
        # After prefill, evict to fit within streaming cache bound
        if kv_cache is not None:
            past_key_values = kv_cache(past_key_values)

        kv_cache_mb = get_mem_mb() - model_mem_mb
        kv_cache_mbs.append(kv_cache_mb)

        # ── Decode ───────────────────────────────────────────────────────────
        reset_peak()
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

        t2 = time.perf_counter()
        for _ in range(gen_len):
            # Physical eviction: keep sinks + recent window before each step
            if kv_cache is not None:
                past_key_values = kv_cache.evict_for_space(past_key_values, num_coming=1)

            with torch.no_grad():
                out = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

        torch.cuda.synchronize()
        t3 = time.perf_counter()

        decode_times.append(t3 - t2)
        decode_peaks.append(get_peak_mb())

        del past_key_values
        torch.cuda.empty_cache()

        print(f"  run {run+1}/{runs}: prefill={prefill_times[-1]*1000:.1f}ms  "
              f"decode={gen_len/(t3-t2):.1f} tok/s  "
              f"kv_cache={kv_cache_mb:.1f} MiB")

    # ── Summary ──────────────────────────────────────────────────────────────
    def avg(lst): return sum(lst) / len(lst)

    mode = f"streaming (sink={start_size}, window={recent_size})" if streaming else "standard"
    print(f"\n{'='*60}")
    print(f"  Model          : {model_path}")
    print(f"  Mode           : {mode}")
    print(f"  Prompt length  : {prompt_len} tokens")
    print(f"  Generate length: {gen_len} tokens")
    print(f"  Runs           : {runs}")
    print(f"{'='*60}")
    print(f"  Model weights  : {model_mem_mb:.1f} MiB")
    print(f"  Prefill latency: {avg(prefill_times)*1000:.1f} ms  "
          f"(= time to first token)")
    print(f"  Decode speed   : {gen_len / avg(decode_times):.1f} tok/s")
    print(f"  KV cache (after prefill): {avg(kv_cache_mbs):.1f} MiB  "
          f"(for {prompt_len} prompt tokens)")
    print(f"  Peak mem prefill: {avg(prefill_peaks):.1f} MiB")
    print(f"  Peak mem decode : {avg(decode_peaks):.1f} MiB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--prompt_len",   type=int,   default=512)
    parser.add_argument("--gen_len",      type=int,   default=128)
    parser.add_argument("--runs",         type=int,   default=3)
    parser.add_argument("--device",                   default="cuda")
    # Streaming-LLM options
    parser.add_argument("--streaming",    action="store_true",
                        help="Enable streaming-LLM inference with bounded KV cache")
    parser.add_argument("--start_size",   type=int,   default=4,
                        help="Number of attention sink tokens to keep (default: 4)")
    parser.add_argument("--recent_size",  type=int,   default=2000,
                        help="Rolling window size in tokens (default: 2000)")
    args = parser.parse_args()

    benchmark(
        args.model_path, args.prompt_len, args.gen_len, args.runs, args.device,
        streaming=args.streaming, start_size=args.start_size, recent_size=args.recent_size,
    )
