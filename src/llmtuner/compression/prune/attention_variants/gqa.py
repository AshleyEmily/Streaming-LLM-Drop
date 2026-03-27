# coding=utf-8
# GQA (Grouped-Query Attention) variant for LLM-Drop-v2 pruning.
#
# Reduces num_key_value_heads by averaging neighbouring KV head weights.
# Both the weight matrices and model.config.num_key_value_heads are updated,
# so the change persists in the saved checkpoint and is correctly loaded at
# eval time (no re-patching needed).
#
# Example: Mistral-7B has 8 KV heads. Setting gqa_num_kv_heads=4 merges
# pairs of adjacent heads by averaging, producing a 4-KV-head model.
#
# Constraint: current_kv_heads must be divisible by target_kv_heads.

import torch
import torch.nn as nn

from .streamllm import _is_attention_module


def _merge_kv_heads(
    module: nn.Module,
    current_kv: int,
    target_kv: int,
) -> None:
    """Merge k_proj and v_proj weight matrices from current_kv to target_kv heads.

    For each projection (k_proj, v_proj):
      - weight shape: (current_kv * head_dim, hidden_size)
      - Reshape to (target_kv, ratio, head_dim, hidden_size)
      - Average over the ratio dimension
      - Reshape back to (target_kv * head_dim, hidden_size)

    Bias terms (if present) are handled the same way.
    """
    if current_kv % target_kv != 0:
        raise ValueError(
            f"current_kv_heads ({current_kv}) must be divisible by "
            f"target_kv_heads ({target_kv})."
        )
    ratio = current_kv // target_kv
    head_dim = module.head_dim

    for proj_name in ("k_proj", "v_proj"):
        proj: nn.Linear = getattr(module, proj_name)

        # Weight: (current_kv * head_dim, hidden_size)
        w = proj.weight.data  # (current_kv * head_dim, hidden)
        hidden = w.shape[1]
        w = w.view(target_kv, ratio, head_dim, hidden)  # (T, R, D, H)
        w = w.mean(dim=1)                               # (T, D, H)
        proj.weight.data = w.view(target_kv * head_dim, hidden)

        # Bias (optional)
        if proj.bias is not None:
            b = proj.bias.data  # (current_kv * head_dim,)
            b = b.view(target_kv, ratio, head_dim).mean(dim=1)  # (T, D)
            proj.bias.data = b.view(target_kv * head_dim)


def patch(model: nn.Module, pruning_args) -> None:
    """Reduce num_key_value_heads to pruning_args.gqa_num_kv_heads.

    Iterates all attention modules and merges their KV projection weights.
    Updates model.config.num_key_value_heads so the change is serialised
    into config.json and loaded correctly at eval time.
    """
    target_kv: int = pruning_args.gqa_num_kv_heads
    patched = 0
    skipped = 0

    for module in model.modules():
        if not _is_attention_module(module):
            continue
        # transformers 5.x moved num_key_value_heads to self.config
        if hasattr(module, "num_key_value_heads"):
            current_kv: int = module.num_key_value_heads
        elif hasattr(module, "config"):
            current_kv: int = module.config.num_key_value_heads
        else:
            continue
        if current_kv <= target_kv:
            skipped += 1
            continue
        _merge_kv_heads(module, current_kv, target_kv)
        if hasattr(module, "num_key_value_heads"):
            module.num_key_value_heads = target_kv
        patched += 1

    # Update config so checkpoint reflects the new KV head count
    model.config.num_key_value_heads = target_kv

    if patched == 0 and skipped == 0:
        raise RuntimeError(
            "GQA patch found no attention modules. "
            "Check that the model architecture is supported."
        )
    print(f"[GQA] merged KV heads in {patched} modules "
          f"(target={target_kv}, skipped={skipped} already at/below target)")
