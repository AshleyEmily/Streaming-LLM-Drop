# coding=utf-8
# LLM-Drop v2 — compact Gemma2 modeling with attention/MLP layer dropping.
#
# Subclasses the current installed Gemma2ForCausalLM and patches individual
# decoder layers to skip dropped attention and/or MLP blocks.
#
# Gemma2-specific notes:
#   - MLP block uses TWO layernorms: pre_feedforward_layernorm (before mlp)
#     and post_feedforward_layernorm (after mlp), unlike Llama's single norm.
#   - The original v1 file imported is_flash_attn_greater_or_equal_2_10 and
#     is_torchdynamo_compiling, both removed/moved in newer transformers.
#     This v2 subclass approach avoids those imports entirely.
#
# Compatibility: transformers >= 4.46.0

import types

from transformers import Gemma2ForCausalLM as _BaseGemma2ForCausalLM

from .configuration_dropped_gemma2 import Gemma2Config


def _dropped_decoder_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Forward for a Gemma2 decoder layer with optional attention/MLP dropping.

    drop_attn=True  -> skip input_layernorm + self_attn + post_attention_layernorm
    drop_mlp=True   -> skip pre_feedforward_layernorm + mlp + post_feedforward_layernorm
    """
    if not self.drop_attn:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        if position_embeddings is not None:
            attn_kwargs["position_embeddings"] = position_embeddings

        attn_out = self.self_attn(**attn_kwargs)
        hidden_states = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        # Gemma2 applies a post-attention layernorm before the residual add
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

    if not self.drop_mlp:
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

    return hidden_states


def _patch_layer(layer, drop_attn: bool, drop_mlp: bool):
    """Patch a Gemma2DecoderLayer instance to skip dropped sub-modules."""
    if not drop_attn and not drop_mlp:
        return

    layer.drop_attn = drop_attn
    layer.drop_mlp = drop_mlp

    if drop_attn:
        layer.self_attn = None
        layer.input_layernorm = None
        layer.post_attention_layernorm = None
    if drop_mlp:
        layer.mlp = None
        layer.pre_feedforward_layernorm = None
        layer.post_feedforward_layernorm = None

    layer.forward = types.MethodType(_dropped_decoder_forward, layer)


class Gemma2ForCausalLM(_BaseGemma2ForCausalLM):
    """Gemma2ForCausalLM with support for dropping attention and/or MLP layers.

    Reads drop_attn_list and drop_mlp_list from config and patches each
    affected decoder layer on init.
    """

    config_class = Gemma2Config

    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        n = config.num_hidden_layers
        drop_attn = list(getattr(config, "drop_attn_list", None) or [False] * n)
        drop_mlp  = list(getattr(config, "drop_mlp_list",  None) or [False] * n)
        drop_attn = (drop_attn + [False] * n)[:n]
        drop_mlp  = (drop_mlp  + [False] * n)[:n]
        for i, layer in enumerate(self.model.layers):
            _patch_layer(layer, bool(drop_attn[i]), bool(drop_mlp[i]))
        # Compress KV cache indices so dropped attention layers don't waste cache slots.
        compressed_idx = 0
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn.layer_idx = compressed_idx
                compressed_idx += 1
        self._n_active_attn_layers = compressed_idx

    def forward(self, *args, **kwargs):
        use_cache = kwargs.get("use_cache")
        effective_use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (effective_use_cache
                and kwargs.get("past_key_values") is None
                and self._n_active_attn_layers < self.config.num_hidden_layers):
            from transformers.cache_utils import DynamicCache
            kwargs["past_key_values"] = DynamicCache()
        return super().forward(*args, **kwargs)
