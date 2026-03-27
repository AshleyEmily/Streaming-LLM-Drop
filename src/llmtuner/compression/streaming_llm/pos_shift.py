"""Position-shifted RoPE helpers for streaming inference.

The key insight from StreamingLLM (Xiao et al., 2023): keys stored in the
rolling cache should be encoded with their *cache-relative* positions (0 ..
kv_seq_len-1) rather than their original text positions.  Queries still use
actual text positions.  This decoupling lets the model attend correctly despite
the gap created by evicting middle tokens.

Adapted for modern transformers (4.x / 5.x) where cos/sin may be pre-computed
at the model level and passed in as a (cos, sin) tuple.
"""

import torch
from typing import Tuple


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_pos_shift_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_pos_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE with standard positions to Q and cache-relative positions to K.

    Parameters
    ----------
    q, k : (B, H, L, head_dim)
        Query and key tensors *before* RoPE.
    cos, sin : broadcastable with q/k
        Pre-computed for actual sequence positions 0 .. L-1.
        The sequence axis must be dim -2 (second-to-last).
        Typical shapes: (1, 1, L, head_dim) or (1, L, head_dim).
    k_pos_ids : LongTensor of shape (L,)
        Cache-relative position index for each key token.  Produced by
        ``get_streaming_position_ids()``.  All values must be < L so they are
        valid indices into cos/sin.

    Returns
    -------
    q_embed, k_embed : same shape as q, k
    """
    # Q: standard RoPE — cos/sin already computed for positions [0 .. L-1]
    q_embed = (q * cos) + (_rotate_half(q) * sin)

    # K: re-indexed RoPE — look up cos/sin at each token's cache position
    # The sequence axis is always dim -2 in the broadcasted cos/sin tensor.
    seq_dim = cos.dim() - 2
    k_cos = cos.index_select(seq_dim, k_pos_ids)
    k_sin = sin.index_select(seq_dim, k_pos_ids)
    k_embed = (k * k_cos) + (_rotate_half(k) * k_sin)

    return q_embed, k_embed


def apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to a single tensor using arbitrary position_ids.

    Used during streaming inference to apply sequential cache positions to
    the full concatenated key cache each decoding step.

    Parameters
    ----------
    x : (B, H, S, head_dim)  — queries or keys
    cos, sin : (1, max_pos, head_dim) or similar — full position table
    position_ids : LongTensor of shape (S,) — positions to index

    Returns
    -------
    x_embed : same shape as x
    """
    seq_dim = cos.dim() - 2
    x_cos = cos.index_select(seq_dim, position_ids)
    x_sin = sin.index_select(seq_dim, position_ids)
    return (x * x_cos) + (_rotate_half(x) * x_sin)
