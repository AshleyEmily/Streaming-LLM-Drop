"""
Perplexity evaluation using native sliding window attention (SWA).

Evaluates token-by-token with a growing KV cache, letting Mistral's built-in
SWA attention mask handle context windowing as the model was trained. No
StreamingLLM or physical KV eviction — this is the model's natural behaviour.

Usage:
    python scripts/eval_standard_ppl.py \
        --model_path mistralai/Mistral-7B-v0.1 \
        --num_eval_tokens 400000 \
        --output_dir outputs/ppl_native
"""

import argparse
import os
import sys
import time

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"


def get_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2


def get_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--task", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_eval_tokens", type=int, default=4000,
                        help="Number of tokens to score (default: 4000)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force_mistral", action="store_true",
                        help="Load with MistralForCausalLM directly (preserves SWA for pruned checkpoints)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.force_mistral:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from llmtuner.compression.prune.models.modeling_dropped_mistral import (
            MistralForCausalLM as DroppedMistralForCausalLM,
        )
        model = DroppedMistralForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device,
        )
    model.eval()
    torch.cuda.synchronize()
    model_mem_mb = get_mem_mb()
    print(f"Model loaded: {model_mem_mb:.1f} MiB")

    data = load_dataset(args.dataset_name, args.task, split=args.split)
    full_text = "\n\n".join(data["text"])
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)
    print(f"Total tokens in dataset: {seq_len:,}")
    print(f"Evaluating first {args.num_eval_tokens} tokens")

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    kv_log_file = open(os.path.join(args.output_dir, "kv_cache_growth.csv"), "w")
    kv_log_file.write("token,allocated_mib\n")

    KV_SAMPLE_INTERVAL = 1000

    loss_fn = CrossEntropyLoss(reduction="none")
    nlls = []
    num_eval_tokens = 0
    past_key_values = None

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    pbar = tqdm(range(seq_len - 1))
    for idx in pbar:
        input_token = input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(input_token, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits.view(-1, model.config.vocab_size)
        past_key_values = outputs.past_key_values  # native SWA mask handles windowing

        label = input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
        neg_log_likelihood = loss_fn(logits, label)
        nlls.append(neg_log_likelihood)

        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=log_file, flush=True)

        num_eval_tokens += 1

        if num_eval_tokens % KV_SAMPLE_INTERVAL == 0:
            kv_bytes = sum(
                entry[0].nbytes + entry[1].nbytes
                for entry in past_key_values
                if entry is not None and entry[0] is not None
            )
            kv_log_file.write(f"{num_eval_tokens},{kv_bytes / 1024**2:.2f}\n")
            kv_log_file.flush()

        if num_eval_tokens >= args.num_eval_tokens:
            break

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    log_file.close()
    kv_log_file.close()

    elapsed = t_end - t_start
    throughput = num_eval_tokens / elapsed
    peak_mem_mb = get_peak_mb()
    kv_cache_mb = peak_mem_mb - model_mem_mb

    ppl = torch.exp(torch.stack(nlls).mean())

    print("")
    print("=" * 60)
    print(f"  Model          : {args.model_path}")
    print(f"  Eval mode      : native SWA (token-by-token)")
    print(f"  Tokens scored  : {num_eval_tokens}")
    print(f"  Perplexity     : {ppl.item():.4f}")
    print(f"  Throughput     : {throughput:.1f} tok/s")
    print(f"  Elapsed        : {elapsed:.1f} s")
    print(f"  Model weights  : {model_mem_mb:.1f} MiB")
    print(f"  Peak GPU mem   : {peak_mem_mb:.1f} MiB")
    print(f"  KV cache mem   : {kv_cache_mb:.1f} MiB")
    print("=" * 60)

    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"ppl={ppl.item():.4f}\n")
        f.write(f"throughput_tok_per_s={throughput:.2f}\n")
        f.write(f"elapsed_s={elapsed:.2f}\n")
        f.write(f"model_mem_mib={model_mem_mb:.1f}\n")
        f.write(f"peak_mem_mib={peak_mem_mb:.1f}\n")
        f.write(f"kv_cache_mem_mib={kv_cache_mb:.1f}\n")
        f.write(f"num_eval_tokens={num_eval_tokens}\n")


if __name__ == "__main__":
    main()
