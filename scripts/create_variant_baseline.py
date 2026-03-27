"""
Create an attention-variant baseline checkpoint: full Mistral-7B with a given
attention variant applied but NO layers dropped. Used to isolate the effect of
each variant alone vs variant + pruning.

Usage:
    python scripts/create_variant_baseline.py \
        --attention_variant streamllm \
        --output_dir ../results_prune/mistral-base-streamllm-nodrop/checkpoint

    python scripts/create_variant_baseline.py \
        --attention_variant ntk_rope --ntk_rope_factor 4.0 \
        --output_dir ../results_prune/mistral-base-ntk_rope-nodrop/checkpoint

    python scripts/create_variant_baseline.py \
        --attention_variant gqa --gqa_num_kv_heads 4 \
        --output_dir ../results_prune/mistral-base-gqa-nodrop/checkpoint
"""

import argparse
import shutil
import sys
import os
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llmtuner.compression.prune.utils import auto_map, CUSTOM_FILE
from llmtuner.compression.prune.attention_variants import patch_attention_variant


class _PruningArgs:
    """Minimal stand-in for PruningArguments with all variant fields."""
    def __init__(self, args):
        self.attention_variant  = args.attention_variant
        self.streamllm_n_init   = args.streamllm_n_init
        self.streamllm_n_local  = args.streamllm_n_local
        self.ntk_rope_factor    = args.ntk_rope_factor
        self.gqa_num_kv_heads   = args.gqa_num_kv_heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--attention_variant",  required=True, choices=["streamllm", "ntk_rope", "gqa"])
    parser.add_argument("--output_dir",         required=True)
    parser.add_argument("--streamllm_n_init",   type=int,   default=128)
    parser.add_argument("--streamllm_n_local",  type=int,   default=4096)
    parser.add_argument("--ntk_rope_factor",    type=float, default=4.0)
    parser.add_argument("--gqa_num_kv_heads",   type=int,   default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",   # keep on CPU — only need to patch config + save weights
    )

    # Apply variant patch via the existing dispatcher
    pruning_args = _PruningArgs(args)
    patch_attention_variant(model, pruning_args)
    print(f"Variant applied: {args.attention_variant}")

    # Build output config — mirrors post_layers_drop exactly
    model_type = model.config.model_type   # "mistral"
    out_cfg = deepcopy(model.config)
    out_cfg.auto_map       = auto_map[model_type]
    out_cfg.drop_attn_list = [False] * out_cfg.num_hidden_layers  # no drops
    out_cfg.drop_mlp_list  = [False] * out_cfg.num_hidden_layers

    # Save weights, tokenizer, config, and custom modeling files
    print(f"Saving to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    out_cfg.save_pretrained(args.output_dir)
    shutil.copy(CUSTOM_FILE[model_type]["config"], args.output_dir)
    shutil.copy(CUSTOM_FILE[model_type]["model"],  args.output_dir)

    print("Done.")
    print(f"  Checkpoint : {args.output_dir}")


if __name__ == "__main__":
    main()
