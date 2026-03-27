# coding=utf-8
# NTK-aware RoPE variant for LLM-Drop-v2 pruning.
#
# Applies dynamic NTK-aware RoPE scaling post-hoc after model load.
# The scale factor is written to model.config.rope_scaling so it is
# serialised into config.json — at eval time HuggingFace reads it and
# initialises the rotary embeddings correctly from scratch.
#
# Supported transformers API versions:
#   - >=4.46 (older): RotaryEmbedding takes (dim, max_position_embeddings, base)
#   - >=5.0  (newer): RotaryEmbedding takes (config, device)
#
# The patch tries the newer API first and falls back to manually updating the
# inv_freq buffer for the older API.

import torch
import torch.nn as nn


def patch(model: nn.Module, pruning_args) -> None:
    """Apply NTK-aware dynamic RoPE scaling to all rotary embedding modules.

    Sets model.config.rope_scaling so the saved checkpoint automatically
    uses this scaling when loaded by HuggingFace at eval time (no re-patching
    needed in the custom modeling files).
    """
    factor: float = pruning_args.ntk_rope_factor

    # Persist scaling config in the model config (survives save/load).
    # transformers 5.x uses rope_parameters; 4.x uses rope_scaling.
    if hasattr(model.config, "rope_parameters") and isinstance(model.config.rope_parameters, dict):
        model.config.rope_parameters = {
            **model.config.rope_parameters,
            "rope_type": "dynamic",
            "factor": factor,
        }
    else:
        model.config.rope_scaling = {"type": "dynamic", "factor": factor}

    patched = 0
    for module in model.modules():
        if not hasattr(module, "rotary_emb"):
            continue
        rotary_emb = module.rotary_emb
        if not hasattr(rotary_emb, "inv_freq"):
            continue

        # Determine the device from existing buffers
        device = rotary_emb.inv_freq.device

        # ── Try newer transformers API (>=5.0): __init__(config, device) ──────
        reinit_ok = False
        try:
            rotary_emb.__init__(model.config, device=device)
            reinit_ok = True
        except (TypeError, AttributeError, KeyError):
            pass

        if not reinit_ok:
            # ── Fallback: manually update inv_freq buffer ────────────────────
            # NTK-aware scaling adjusts the base per dimension:
            #   new_base = base * factor ^ (dim / (dim - 2))
            # which is equivalent to applying NTK interpolation across freqs.
            try:
                inv_freq_old = rotary_emb.inv_freq
                dim = inv_freq_old.shape[0] * 2
                base = getattr(model.config, "rope_theta", 10000.0)
                new_base = base * (factor ** (dim / (dim - 2)))
                inv_freq_new = 1.0 / (
                    new_base ** (
                        torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
                    )
                )
                rotary_emb.register_buffer("inv_freq", inv_freq_new, persistent=False)
                reinit_ok = True
            except Exception as e:
                print(f"[NTK-RoPE] Warning: could not update inv_freq for "
                      f"{type(rotary_emb).__name__}: {e}")

        if reinit_ok:
            patched += 1

    if patched == 0:
        print("[NTK-RoPE] Warning: no rotary_emb modules found — "
              "rope_scaling was written to config but in-memory buffers unchanged.")
    else:
        print(f"[NTK-RoPE] updated {patched} rotary_emb modules (factor={factor})")
