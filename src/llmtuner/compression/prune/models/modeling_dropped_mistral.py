# coding=utf-8
# LLM-Drop v2 — compact Mistral modeling with attention/MLP layer dropping.
#
# Subclasses the current installed MistralForCausalLM and patches individual
# decoder layers to skip dropped attention and/or MLP blocks.
#
# Compatibility: transformers >= 4.46.0
#   - Handles position_embeddings kwarg
#   - No dependency on removed utilities
#   - Works with DynamicCache

import types

from transformers import MistralForCausalLM as _BaseMistralForCausalLM

from .configuration_dropped_mistral import MistralConfig


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
    """Forward for a Mistral decoder layer with optional attention/MLP dropping.

    drop_attn=True  -> skip input_layernorm + self_attn
    drop_mlp=True   -> skip post_attention_layernorm + mlp
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
        hidden_states = residual + hidden_states

    if not self.drop_mlp:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

    return hidden_states


def _patch_layer(layer, drop_attn: bool, drop_mlp: bool):
    """Patch a MistralDecoderLayer instance to skip dropped sub-modules."""
    if not drop_attn and not drop_mlp:
        return

    layer.drop_attn = drop_attn
    layer.drop_mlp = drop_mlp

    if drop_attn:
        layer.self_attn = None
        layer.input_layernorm = None
    if drop_mlp:
        layer.mlp = None
        layer.post_attention_layernorm = None

    layer.forward = types.MethodType(_dropped_decoder_forward, layer)


class MistralForCausalLM(_BaseMistralForCausalLM):
    """MistralForCausalLM with support for dropping attention and/or MLP layers.

    Reads drop_attn_list and drop_mlp_list from config and patches each
    affected decoder layer on init.
    """

    config_class = MistralConfig

    def __init__(self, config: MistralConfig):
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

        # Re-apply StreamLLM forward patch if config carries the parameters.
        # This restores the patch after the checkpoint is loaded at eval time.
        self._use_streamllm = bool(getattr(config, "streamllm_n_init", None))
        if self._use_streamllm:
            import types as _types
            from llmtuner.compression.prune.attention_variants.streamllm import (
                _is_attention_module as _is_attn,
                streamllm_forward as _slm_fwd,
                _streamllm_model_forward as _slm_model_fwd,
            )
            for layer in self.model.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None and _is_attn(attn):
                    attn._streamllm_n_init = config.streamllm_n_init
                    attn._streamllm_n_local = config.streamllm_n_local
                    attn.forward = _types.MethodType(_slm_fwd, attn)
            # Also patch model.model.forward to skip sliding-window mask at eval time
            self.model.forward = _types.MethodType(_slm_model_fwd, self.model)

    def forward(self, *args, **kwargs):
        # StreamLLM uses mask-based full-sequence attention and does not
        # maintain a KV cache. Force use_cache=False to prevent the base
        # class from trying to collect cache entries that are never returned.
        if self._use_streamllm:
            kwargs["use_cache"] = False
            kwargs.pop("past_key_values", None)

        return super().forward(*args, **kwargs)
