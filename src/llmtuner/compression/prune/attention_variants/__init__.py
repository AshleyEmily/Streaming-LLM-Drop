# coding=utf-8
# Pluggable attention variant system for LLM-Drop-v2 pruning.
#
# Each variant implements a `patch(model, pruning_args) -> None` function that
# modifies the model in-place before calibration scoring runs.
#
# Adding a new variant: create a new .py file in this package that exports
# `patch(model, pruning_args)`, then register it in _REGISTRY below.

from .streamllm import patch as _streamllm_patch
from .ntk_rope import patch as _ntk_rope_patch
from .gqa import patch as _gqa_patch

_REGISTRY = {
    "streamllm": _streamllm_patch,
    "ntk_rope": _ntk_rope_patch,
    "gqa": _gqa_patch,
}


def patch_attention_variant(model, pruning_args) -> None:
    """Apply the selected attention variant to the model in-place.

    Called from workflow.py after model load, before calibration scoring.
    Each variant may also write fields to model.config so they persist in
    the saved checkpoint and are re-applied at eval time.
    """
    variant = getattr(pruning_args, "attention_variant", None)
    if not variant:
        return
    if variant not in _REGISTRY:
        raise ValueError(
            f"Unknown attention_variant '{variant}'. "
            f"Choices: {sorted(_REGISTRY)}"
        )
    _REGISTRY[variant](model, pruning_args)
