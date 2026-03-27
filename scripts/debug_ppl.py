"""
Minimal PPL debug script.

Runs three evaluations on 100 tokens each to isolate where the bug is:
  1. Dense attention  — no streaming, standard model.forward()
  2. Streaming        — with enable_streaming_llm()
  3. Streaming (verbose) — prints position_ids seen at each step

Usage:
    python scripts/debug_ppl.py --model_path mistralai/Mistral-7B-v0.1
"""

import argparse
import os
import sys
import types

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llmtuner.compression.streaming_llm import enable_streaming_llm
from llmtuner.compression.streaming_llm.kv_cache import StartRecentKVCache

device = "cuda"
N_TOKENS = 100


def get_text():
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(data["text"])
    return full_text


def eval_ppl(model, tokenizer, input_ids, kv_cache=None, label=""):
    """Run token-by-token PPL on first N_TOKENS of input_ids."""
    loss_fn = CrossEntropyLoss(reduction="none")
    nlls = []
    past_key_values = None

    for idx in range(N_TOKENS):
        token = input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(token, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits.view(-1, model.config.vocab_size)
        past_key_values = outputs.past_key_values
        if kv_cache is not None:
            past_key_values = kv_cache(past_key_values)

        label_tok = input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
        nll = CrossEntropyLoss(reduction="none")(logits, label_tok)
        nlls.append(nll)

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f"  [{label}] PPL over {N_TOKENS} tokens: {ppl:.4f}")
    return ppl


def eval_ppl_verbose(model, tokenizer, input_ids, kv_cache):
    """Same as eval_ppl but prints position_ids seen inside _streaming_model_forward."""
    seen_positions = []

    # Monkey-patch to intercept position_ids
    orig_forward = model.model.forward.__func__ if hasattr(model.model.forward, "__func__") else None
    _orig = model.model.forward

    def _debug_forward(self_or_none, *args, **kwargs):
        # works whether bound or unbound
        if orig_forward is not None:
            result = orig_forward(self_or_none, *args, **kwargs)
        else:
            result = _orig(*args, **kwargs)
        return result

    # Instead, patch at a higher level to just print position_ids argument
    loss_fn = CrossEntropyLoss(reduction="none")
    nlls = []
    past_key_values = None

    for idx in range(min(10, N_TOKENS)):  # only print first 10 steps
        token = input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            # Manually compute what position_ids the streaming forward should use
            if past_key_values is not None and isinstance(past_key_values, (list, tuple)):
                cache_len = past_key_values[0][0].shape[2]
            else:
                cache_len = 0

            outputs = model(token, past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values
        if kv_cache is not None:
            past_key_values = kv_cache(past_key_values)

        # Check cache length after eviction
        if past_key_values is not None and isinstance(past_key_values, (list, tuple)):
            actual_cache_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
        else:
            actual_cache_len = "non-list"

        logits = outputs.logits.view(-1, model.config.vocab_size)
        label_tok = input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
        nll = loss_fn(logits, label_tok)
        nlls.append(nll)

        pred_token = logits.argmax(dim=-1).item()
        print(f"    step={idx:3d}  expected_pos={cache_len}  "
              f"cache_after_evict={actual_cache_len}  "
              f"nll={nll.item():.2f}  "
              f"pred={tokenizer.decode([pred_token])!r:10s}  "
              f"true={tokenizer.decode([label_tok.item()])!r}")

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f"  [verbose streaming] PPL over {min(10, N_TOKENS)} tokens: {ppl:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    args = parser.parse_args()

    print(f"Loading: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=torch.float16, device_map=device
    )
    model.eval()

    if hasattr(model, "_use_streamllm"):
        model._use_streamllm = False

    text = get_text()
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    print(f"\nFirst 20 tokens: {tokenizer.decode(input_ids[0, :20])!r}")
    print(f"Total tokens: {input_ids.size(1):,}")
    print()

    # ── Test 1: Dense (no streaming) ───────────────────────────────────────────
    print("=== Test 1: Dense (no streaming, past_key_values accumulates) ===")
    with torch.no_grad():
        dense_logits_step0 = model(input_ids[:, :1].to(device)).logits
    eval_ppl(model, tokenizer, input_ids, kv_cache=None, label="dense")

    # ── Patch verification ─────────────────────────────────────────────────────
    print("\n=== Patch verification ===")
    kv_cache = enable_streaming_llm(model, start_size=args.start_size, recent_size=args.recent_size)

    first_attn = model.model.layers[0].self_attn
    fwd = first_attn.forward
    fwd_name = getattr(getattr(fwd, "__func__", fwd), "__name__", str(fwd))
    print(f"model.model.layers[0].self_attn.forward = {fwd_name}")
    print(f"model.model.forward name = {getattr(getattr(model.model.forward, '__func__', model.model.forward), '__name__', '?')}")

    # Compare step-0 logits before and after patching — should differ if patched
    with torch.no_grad():
        streaming_logits_step0 = model(input_ids[:, :1].to(device)).logits
    max_diff = (dense_logits_step0 - streaming_logits_step0).abs().max().item()
    print(f"Step-0 logit max diff (dense vs streaming): {max_diff:.6f}")
    print("  (0.0 = patch has NO effect; non-zero = patch IS active)")

    # ── Cache inspection after step 0 ─────────────────────────────────────────
    print("\n=== Cache inspection after step 0 ===")
    with torch.no_grad():
        out0 = model(input_ids[:, :1].to(device), use_cache=True)
    pkv = out0.past_key_values
    print(f"past_key_values type : {type(pkv)}")
    print(f"past_key_values len  : {len(pkv)}")
    entry = pkv[0]
    print(f"pkv[0] type          : {type(entry)}")
    k0, v0 = entry[0], entry[1]
    print(f"pkv[0][0] (k) shape  : {k0.shape}")
    print(f"pkv[0][1] (v) shape  : {v0.shape}")

    # Now do step 1 and compare streaming logits vs dense reference
    print("\n=== Step-1 logit comparison (dense ref vs streaming) ===")
    # Dense reference: run model in a single 2-token forward (full attention)
    with torch.no_grad():
        ref_logits = model.__class__.forward(
            model,
            input_ids=input_ids[:, :2].to(device),
            use_cache=False,
        ).logits[:, -1, :]  # logits for predicting token 2 from tokens 0-1

    # Streaming step 1: use the cache from step 0
    pkv_evicted = kv_cache(pkv)
    with torch.no_grad():
        stream_out1 = model(input_ids[:, 1:2].to(device), past_key_values=pkv_evicted, use_cache=True)
    stream_logits1 = stream_out1.logits.view(-1, model.config.vocab_size)

    diff1 = (ref_logits - stream_logits1).abs().max().item()
    pred_ref   = tokenizer.decode([ref_logits.argmax(-1).item()])
    pred_stream = tokenizer.decode([stream_logits1.argmax(-1).item()])
    true_tok   = tokenizer.decode([input_ids[0, 2].item()])
    print(f"Step-1 max logit diff: {diff1:.4f}")
    print(f"  dense ref pred  : {pred_ref!r}  (true: {true_tok!r})")
    print(f"  streaming pred  : {pred_stream!r}")

    # ── Test 2: Streaming ──────────────────────────────────────────────────────
    print("\n=== Test 2: Streaming ===")
    eval_ppl(model, tokenizer, input_ids, kv_cache=kv_cache, label="streaming")

    # ── Test 3: Verbose streaming (inspect positions + predictions) ────────────
    print("\n=== Test 3: Streaming verbose (first 10 steps) ===")
    eval_ppl_verbose(model, tokenizer, input_ids, kv_cache=kv_cache)

    # ── Test 4: Streaming generation (qualitative check) ───────────────────────
    print("\n=== Test 4: Streaming generation (qualitative) ===")
    prompt = "Robert Boulter is an English actor who"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = prompt_ids.clone()
    past_key_values = None

    with torch.no_grad():
        # Prefill
        out = model(prompt_ids, past_key_values=None, use_cache=True)
        past_key_values = kv_cache(out.past_key_values)
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=1)

        # Decode 100 more tokens
        for _ in range(99):
            out = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = kv_cache(out.past_key_values)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

    print(f"Prompt : {prompt!r}")
    print(f"Generated:\n{tokenizer.decode(generated[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
