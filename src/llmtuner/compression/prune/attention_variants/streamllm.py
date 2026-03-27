# coding=utf-8
# StreamLLM attention variant for LLM-Drop-v2 pruning.
#
# Implements attention sinks + sliding local window (StreamLLM) using a
# mask-based approach.  Works for full-sequence forward passes (calibration
# scoring and lm_eval loglikelihood evaluation) without a sliding KV cache.
#
# Supported models: Llama, Mistral (and any model whose attention module
# exposes q_proj / k_proj / v_proj / o_proj / num_heads /
# num_key_value_heads / head_dim / rotary_emb).
#
# Attention mask pattern for sequence length L:
#   position i attends to position j  iff
#     j < n_init          (sink tokens, always visible)
#     OR j >= i - n_local + 1  AND j <= i   (local causal window)
#
# No dependency on InfLLM-v2; _is_attention_module logic adapted from
# InfLLM-v2/inf_llm_v2/patch.py lines 38-91.

import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Attention module detection ────────────────────────────────────────────────

_ATTN_CLASS_SUBSTRINGS = ("Attention", "SelfAttention")

_KNOWN_ATTN_CLASSES: Tuple[str, ...] = (
    "LlamaAttention", "LlamaSdpaAttention", "LlamaFlashAttention2",
    "MistralAttention", "MistralSdpaAttention", "MistralFlashAttention2",
    "Qwen2Attention", "Qwen2SdpaAttention", "Qwen2FlashAttention2",
    "Phi3Attention",
    "Gemma2Attention", "Gemma2SdpaAttention",
    "CohereAttention",
    "FalconAttention",
)


def _is_attention_module(module: nn.Module) -> bool:
    """Return True if module looks like a self-attention layer."""
    cls_name = type(module).__name__
    if cls_name in _KNOWN_ATTN_CLASSES:
        return True
    if any(s in cls_name for s in _ATTN_CLASS_SUBSTRINGS):
        return (
            hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and hasattr(module, "o_proj")
        )
    return False


# ── Self-contained RoPE helpers ───────────────────────────────────────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k."""
    # cos/sin: (1, 1, L, head_dim) or (B, 1, L, head_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ── StreamLLM attention forward ───────────────────────────────────────────────

def _get_num_heads(module: nn.Module) -> int:
    """Get num_heads, handling both transformers 4.x (attr) and 5.x (config)."""
    if hasattr(module, "num_heads"):
        return module.num_heads
    if hasattr(module, "config"):
        return module.config.num_attention_heads
    raise AttributeError(f"{type(module).__name__} has no num_heads or config attribute")


def _get_num_kv_heads(module: nn.Module) -> int:
    """Get num_key_value_heads, handling both transformers 4.x and 5.x."""
    if hasattr(module, "num_key_value_heads"):
        return module.num_key_value_heads
    if hasattr(module, "config"):
        return module.config.num_key_value_heads
    raise AttributeError(f"{type(module).__name__} has no num_key_value_heads or config attribute")


def streamllm_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,        # transformers 4.x name (ignored — mask-based)
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """StreamLLM mask-based attention forward (no KV cache).

    Each position i attends only to:
      - sink tokens: positions 0 .. n_init-1
      - local window: positions max(n_init, i - n_local + 1) .. i  (causal)

    Works on full-sequence forward passes (calibration and lm_eval
    loglikelihood). Does NOT support incremental generation with a KV cache.

    Compatible with transformers 4.x (num_heads attr, rotary_emb module) and
    5.x (config-based head counts, position_embeddings always passed in).
    """
    B, L, _ = hidden_states.shape
    n_init: int = self._streamllm_n_init
    n_local: int = self._streamllm_n_local

    num_heads    = _get_num_heads(self)
    num_kv_heads = _get_num_kv_heads(self)
    head_dim     = self.head_dim

    # 1. Linear projections + reshape to (B, heads, L, head_dim)
    q = self.q_proj(hidden_states).view(B, L, num_heads,    head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, L, num_kv_heads, head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, L, num_kv_heads, head_dim).transpose(1, 2)

    # 2. Apply position-shifted RoPE
    #    Q uses actual text positions [0..L-1]; K uses cache-relative positions
    #    that simulate where each key would sit in the rolling streaming cache.
    #    This matches the RoPE assignment in live streaming inference (original
    #    streaming-llm: keys stored raw, re-encoded with arange(0,kv_seq_len)).
    if position_embeddings is not None:
        cos, sin = position_embeddings
    elif hasattr(self, "rotary_emb"):
        cos, sin = self.rotary_emb(v, position_ids)
    else:
        raise RuntimeError(
            "streamllm_forward: position_embeddings not provided and module has no "
            "rotary_emb (transformers 5.x always passes position_embeddings from the "
            "decoder layer — this error should not occur in normal use)."
        )
    from ...streaming_llm import apply_pos_shift_rope, get_streaming_position_ids
    k_pos_ids = get_streaming_position_ids(L, n_init, n_local, device=q.device)
    q, k = apply_pos_shift_rope(q, k, cos, sin, k_pos_ids)

    # 3. GQA expansion: repeat K/V heads to match Q heads
    if num_heads != num_kv_heads:
        g = num_heads // num_kv_heads
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    # 4. Build StreamLLM causal attention bias  (L, L)
    #    allowed[i, j] = True  iff position i may attend to position j
    col = torch.arange(L, device=hidden_states.device).unsqueeze(0)   # (1, L)
    row = torch.arange(L, device=hidden_states.device).unsqueeze(1)   # (L, 1)
    causal   = col <= row                                              # standard causal
    is_sink  = col < n_init                                            # first n_init tokens
    is_local = col >= (row - n_local + 1)                              # local window
    allowed  = causal & (is_sink | is_local)

    bias = torch.zeros(1, 1, L, L, dtype=q.dtype, device=q.device)
    bias.masked_fill_(~allowed, float("-inf"))

    # 5. Scaled dot-product attention + output projection
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
    out = out.transpose(1, 2).contiguous().view(B, L, num_heads * head_dim)
    return self.o_proj(out), None


# ── Model-level forward (skips sliding-window mask) ───────────────────────────

def _streamllm_model_forward(
    self,
    input_ids=None,
    attention_mask=None,        # accepted but ignored — StreamLLM builds its own mask
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
    """Lightweight MistralModel / LlamaModel forward that skips mask computation.

    The standard forward calls create_sliding_window_causal_mask which builds
    (B, H, L, L) index tensors — at L=8192 this costs ~17 GB on an A100 and
    OOMs before the attention layers are even reached.  StreamLLM builds its
    own sink+local-window mask inside streamllm_forward, so the model-level
    mask is unused.  We skip it entirely and pass attention_mask=None to each
    decoder layer.

    Still computes position_embeddings via self.rotary_emb: transformers 5.x
    moved rotary embedding computation to the model level and attention modules
    no longer have their own rotary_emb, so the (cos, sin) tuple must be
    passed down explicitly.

    Works for both MistralModel and LlamaModel (same attribute names).
    Only installed when StreamLLM is active — has no effect on other variants.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast

    output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    seq_length = inputs_embeds.shape[1]

    if position_ids is None:
        device = inputs_embeds.device
        start = int(cache_position[0].item()) if cache_position is not None else 0
        position_ids = torch.arange(start, start + seq_length, device=device).unsqueeze(0)

    # ── Skip create_sliding_window_causal_mask / create_causal_mask ───────────

    hidden_states = inputs_embeds

    # position_embeddings: required by transformers 5.x decoder layers
    position_embeddings = None
    if hasattr(self, "rotary_emb"):
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns    = () if output_attentions    else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_kwargs = dict(
            attention_mask=None,    # StreamLLM builds its own mask inside streamllm_forward
            position_ids=position_ids,
            use_cache=False,        # StreamLLM does not maintain a KV cache
        )
        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings

        result = decoder_layer(hidden_states, **layer_kwargs)
        hidden_states = result if isinstance(result, torch.Tensor) else result[0]

        if output_attentions:
            all_self_attns += (None,)   # streamllm_forward always returns None for attn weights

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# ── Patch entry point ─────────────────────────────────────────────────────────

def patch(model: nn.Module, pruning_args) -> None:
    """Replace self_attn.forward on all attention layers with streamllm_forward.
    Also replaces model.model.forward with _streamllm_model_forward to skip the
    sliding-window causal mask computation (which OOMs at long sequence lengths).

    Writes streamllm_n_init / streamllm_n_local to model.config so they are
    serialised into config.json and the custom modeling class can re-apply both
    patches when the checkpoint is loaded at eval time.
    """
    n_init: int = pruning_args.streamllm_n_init
    n_local: int = pruning_args.streamllm_n_local

    # Persist parameters in config so the checkpoint includes them
    model.config.streamllm_n_init = n_init
    model.config.streamllm_n_local = n_local

    patched = 0
    for module in model.modules():
        if _is_attention_module(module):
            module._streamllm_n_init = n_init
            module._streamllm_n_local = n_local
            module.forward = types.MethodType(streamllm_forward, module)
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "StreamLLM patch found no attention modules. "
            "Check that the model architecture is supported."
        )

    # Patch model.model.forward to skip sliding-window causal mask computation.
    # The standard MistralModel/LlamaModel forward builds (B,H,L,L) index tensors
    # for the mask — at L≥8192 this OOMs before reaching the attention layers.
    model.model._streamllm_old_forward = model.model.forward
    model.model.forward = types.MethodType(_streamllm_model_forward, model.model)

    print(f"[StreamLLM] patched {patched} attention modules "
          f"(n_init={n_init}, n_local={n_local})")
    print(f"[StreamLLM] patched model.model.forward to skip sliding-window mask")
