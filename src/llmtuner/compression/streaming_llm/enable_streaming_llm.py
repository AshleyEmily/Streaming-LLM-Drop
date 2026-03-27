"""Patch a model for streaming inference with attention sinks + rolling KV cache.

This module provides ``enable_streaming_llm(model, start_size, recent_size)``
which:
  1. Replaces every attention module's forward with a streaming-aware version
     that stores raw (pre-RoPE) keys and re-applies position-shifted RoPE at
     each decode step.
  2. Patches ``model.model.forward`` to manage the raw-key KV cache and pass
     the correct position embeddings for the full cache length.
  3. Returns a ``StartRecentKVCache`` for the caller to evict tokens before
     each generation step.

Supported architectures: llama, mistral (and any model sharing the same
attention attribute names: q_proj / k_proj / v_proj / o_proj / num_heads /
num_key_value_heads / head_dim).

Usage (in benchmark_inference.py)::

    kv_cache = enable_streaming_llm(model, start_size=4, recent_size=2000)
    past_key_values = None
    for _ in range(gen_len):
        past_key_values = kv_cache.evict_for_space(past_key_values, num_coming=1)
        out = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
"""

import types
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .kv_cache import StartRecentKVCache
from .pos_shift import _rotate_half, apply_rope_single


# ── Supported architectures ───────────────────────────────────────────────────

_SUPPORTED_MODEL_TYPES = {"llama", "mistral", "llama2", "llama3"}


def _model_type(model) -> str:
    mt = model.config.model_type.lower()
    for supported in _SUPPORTED_MODEL_TYPES:
        if supported in mt:
            return "llama"   # all share the same attention layout
    raise ValueError(
        f"enable_streaming_llm: unsupported model_type '{model.config.model_type}'. "
        f"Supported: {_SUPPORTED_MODEL_TYPES}"
    )


# ── Per-step attention forward (raw-key cache + position-shifted RoPE) ────────

def _streaming_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = True,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """Attention forward that stores raw (pre-RoPE) keys and recomputes
    position-shifted RoPE for the full cache at every decoding step.

    ``past_key_value`` is a plain (k_raw, v) tuple — NOT a transformers Cache
    object.  The caller (``_streaming_model_forward``) accumulates these and
    ``StartRecentKVCache.evict_for_space`` operates on the same format.
    """
    B, L, _ = hidden_states.shape

    num_heads    = self.num_heads if hasattr(self, "num_heads") else self.config.num_attention_heads
    num_kv_heads = self.num_key_value_heads if hasattr(self, "num_key_value_heads") else self.config.num_key_value_heads
    head_dim     = self.head_dim

    # 1. Project — raw key (no RoPE yet)
    q      = self.q_proj(hidden_states).view(B, L, num_heads,    head_dim).transpose(1, 2)
    k_raw  = self.k_proj(hidden_states).view(B, L, num_kv_heads, head_dim).transpose(1, 2)
    v      = self.v_proj(hidden_states).view(B, L, num_kv_heads, head_dim).transpose(1, 2)

    # 2. Concatenate with cached raw keys.
    # In transformers 5.x the decoder layer may not forward past_key_value as a
    # kwarg, so fall back to the side channel set by _streaming_model_forward.
    if past_key_value is None:
        past_key_value = getattr(self, "_streaming_past_kv", None)
    if past_key_value is not None:
        k_raw = torch.cat([past_key_value[0], k_raw], dim=2)
        v     = torch.cat([past_key_value[1], v],     dim=2)

    kv_seq_len = k_raw.shape[2]

    # 3. Compute cos/sin for the full cache length (positions 0..kv_seq_len-1)
    cache_pos_ids = torch.arange(kv_seq_len, device=hidden_states.device).unsqueeze(0)  # (1, S)
    rotary_emb = getattr(self, "_streaming_rotary_emb", None) or getattr(self, "rotary_emb", None)
    if rotary_emb is not None:
        cos_full, sin_full = rotary_emb(v, position_ids=cache_pos_ids)
    elif position_embeddings is not None:
        # position_embeddings is for the current token only (length L); re-use
        # it as-is only when the cache is just L long (first decode step after
        # prefill).  For subsequent steps we need a proper full-length table.
        # Fall back: index into position_embeddings if large enough.
        cos_full, sin_full = position_embeddings
        if cos_full.shape[-2] < kv_seq_len:
            raise RuntimeError(
                "enable_streaming_llm: position_embeddings too short for kv_seq_len. "
                "Ensure model.model.rotary_emb is accessible (transformers >= 4.x)."
            )
    else:
        raise RuntimeError(
            "enable_streaming_llm: no rotary_emb found on attention module or model. "
            "Attach model.model.rotary_emb via enable_streaming_llm() before inference."
        )

    # 4. Apply RoPE:
    #    - Q: use cache-relative position = kv_seq_len - L .. kv_seq_len - 1
    #    - K: use sequential cache positions 0..kv_seq_len-1 (re-applied every step)
    #
    # We do NOT use the passed-in position_ids for Q. MistralDecoderLayer.forward
    # recomputes position_ids from cache_position (defaulting to [0]) and passes
    # that to self.self_attn, overriding the correct value set by
    # _streaming_model_forward. Deriving q_pos from kv_seq_len is always correct:
    # keys are encoded at positions [0..kv_seq_len-1], so the L new query tokens
    # occupy positions [kv_seq_len-L .. kv_seq_len-1].
    q_pos = torch.arange(kv_seq_len - L, kv_seq_len, device=hidden_states.device)
    k_pos = cache_pos_ids.squeeze(0)  # (kv_seq_len,)

    q = apply_rope_single(q, cos_full, sin_full, q_pos)
    k = apply_rope_single(k_raw, cos_full, sin_full, k_pos)

    # 5. Save unexpanded v for the cache BEFORE GQA expansion.
    # v currently has shape (B, num_kv_heads, S, head_dim). After repeat_interleave
    # it becomes (B, num_heads, S, head_dim). Storing v[:, :num_kv_heads] of the
    # expanded tensor would only capture the first few original heads (wrong for GQA).
    v_for_cache = v  # (B, num_kv_heads, S, head_dim) — unexpanded

    # 6. GQA expansion for attention
    if num_heads != num_kv_heads:
        g = num_heads // num_kv_heads
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    # 7. Causal attention  (is_causal only needed during prefill L > 1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=(L > 1), dropout_p=0.0)
    out = out.transpose(1, 2).contiguous().view(B, L, num_heads * head_dim)

    # 7. Store raw-key cache as side channel and return 2 values.
    # Transformers 5.x decoder layers unpack exactly 2 values from self_attn;
    # returning 3 raises ValueError.  _streaming_model_forward reads the cache
    # from self._streaming_new_past after each decoder_layer call.
    new_past = (k_raw, v_for_cache) if use_cache else None
    self._streaming_new_past = new_past

    return self.o_proj(out), None


# ── Model-level forward that manages the raw-key past_key_values ──────────────

def _streaming_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    """LlamaModel / MistralModel forward that manages raw-key past_key_values.

    Returns past_key_values as a list of (k_raw, v) tuples compatible with
    ``StartRecentKVCache.evict_for_space``.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast

    output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict
    use_cache            = use_cache            if use_cache            is not None else self.config.use_cache

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    seq_length = inputs_embeds.shape[1]

    # Always recompute position_ids from our cache — do NOT trust what was passed in.
    # MistralForCausalLM.forward() in newer transformers computes position_ids based on
    # DynamicCache.get_seq_length(), but our past_key_values is a plain list-of-tuples
    # (not a Cache), so it sees past_seen_tokens=0 and passes position_ids=[[0]] for
    # every token. Using that would make every query have RoPE at position 0, destroying
    # attention quality. We own the cache so we always know the true position.
    device = inputs_embeds.device
    # Handle DynamicCache passed in on the first step (None → empty DynamicCache).
    if past_key_values is not None and not isinstance(past_key_values, (list, tuple)):
        cache_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else 0
        if cache_len == 0:
            past_key_values = None  # empty DynamicCache on first step — treat as no cache
    cache_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    position_ids = torch.arange(cache_len, cache_len + seq_length, device=device).unsqueeze(0)

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    new_past_key_values: List = []

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_kv = past_key_values[i] if past_key_values is not None else None

        # Pass input past KV directly on the attention module so it's accessible
        # inside _streaming_attn_forward regardless of whether transformers 5.x
        # forwards the past_key_value kwarg through the decoder layer.
        # self_attn is None for dropped attention layers — skip those.
        attn = getattr(decoder_layer, "self_attn", None)
        if attn is not None:
            attn._streaming_past_kv = past_kv

        result = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=past_kv,
            use_cache=use_cache,
        )

        # Read new past KV from side channel set by _streaming_attn_forward.
        # Transformers 5.x unpacks only 2 values from self_attn so we can't
        # rely on result[2] — the cache is stored on the module instance instead.
        # Dropped layers have self_attn=None and contribute no KV cache entry.
        hidden_states = result[0] if not isinstance(result, torch.Tensor) else result
        new_past = getattr(attn, "_streaming_new_past", None)

        if use_cache:
            new_past_key_values.append(new_past)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    past_out = new_past_key_values if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, past_out, all_hidden_states] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_out,
        hidden_states=all_hidden_states,
        attentions=None,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def enable_streaming_llm(model, start_size: int = 4, recent_size: int = 2000):
    """Patch *model* for streaming inference and return a KV cache manager.

    After calling this function:
    - All attention modules use position-shifted RoPE with raw-key caching.
    - ``model.model.forward`` manages the cache as plain (k_raw, v) tuples.
    - The returned ``StartRecentKVCache`` should be used to evict tokens before
      each generation step:

        past_key_values = kv_cache.evict_for_space(past_key_values, num_coming=1)

    Parameters
    ----------
    model : PreTrainedModel
    start_size : int
        Number of attention sink tokens (initial tokens) to always keep.
    recent_size : int
        Number of most-recent tokens to keep in the rolling window.

    Returns
    -------
    kv_cache : StartRecentKVCache
    """
    _model_type(model)  # validates architecture

    # Attach model-level rotary_emb to each attention module so it's accessible
    # inside _streaming_attn_forward (needed for transformers 5.x where
    # attention modules no longer have their own rotary_emb).
    model_rotary = getattr(model.model, "rotary_emb", None)

    patched = 0
    for module in model.modules():
        if _is_llama_attn(module):
            if model_rotary is not None and not hasattr(module, "rotary_emb"):
                module._streaming_rotary_emb = model_rotary
            module.forward = types.MethodType(_streaming_attn_forward, module)
            patched += 1

    if patched == 0:
        raise RuntimeError("enable_streaming_llm: no attention modules found.")

    # Patch model.model.forward to manage raw-key past_key_values
    model.model.forward = types.MethodType(_streaming_model_forward, model.model)

    print(f"[enable_streaming_llm] patched {patched} attention modules "
          f"(start_size={start_size}, recent_size={recent_size})")

    # k_seq_dim=2, v_seq_dim=2 for (B, H, S, head_dim) tensors
    return StartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=2,
        v_seq_dim=2,
    )


def _is_llama_attn(module) -> bool:
    # num_heads moved to config in transformers 5.x — check projections only
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
        and (hasattr(module, "num_heads") or hasattr(module, "config"))
    )
