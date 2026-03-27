# coding=utf-8
# LLM-Drop v2 — LlamaConfig subclass with drop_attn_list / drop_mlp_list fields.
#
# Instead of maintaining a full copy of the transformers LlamaConfig source,
# this file subclasses the currently installed version and adds the two extra
# fields needed by LLM-Drop.  This makes it forward-compatible with any
# transformers version that provides LlamaConfig.
#
# Compatibility: transformers >= 4.46.0

from transformers import LlamaConfig as _BaseLlamaConfig


class LlamaConfig(_BaseLlamaConfig):
    """LlamaConfig extended with layer-dropping support.

    Extra fields (both default to all-False lists of length num_hidden_layers):
        drop_attn_list: list[bool]  – True at index i means attention sub-block
                                      of layer i is skipped.
        drop_mlp_list:  list[bool]  – True at index i means MLP sub-block of
                                      layer i is skipped.

    Accepts both bool lists and lists of integer layer indices for backward
    compatibility with the v1 format.
    """

    def __init__(
        self,
        drop_attn_list=None,
        drop_mlp_list=None,
        streamllm_n_init=None,
        streamllm_n_local=None,
        **kwargs,
    ):
        # Resolve num_hidden_layers before calling super so we can build the
        # drop lists; super() will set self.num_hidden_layers.
        n = kwargs.get("num_hidden_layers", 32)

        # Normalise each list: accept bool lists or lists of int layer indices.
        def _to_bool_list(raw, n):
            if raw is None:
                return [False] * n
            # If entries are ints, interpret as indices of dropped layers.
            if raw and isinstance(raw[0], (int,)) and not isinstance(raw[0], bool):
                as_set = set(raw)
                return [i in as_set for i in range(n)]
            # Bool list — extend / truncate to length n.
            raw = list(raw)
            return (raw + [False] * n)[:n]

        self.drop_attn_list = _to_bool_list(drop_attn_list, n)
        self.drop_mlp_list  = _to_bool_list(drop_mlp_list,  n)
        self.streamllm_n_init  = streamllm_n_init
        self.streamllm_n_local = streamllm_n_local

        super().__init__(**kwargs)

        # Re-normalise after super() sets the authoritative num_hidden_layers.
        n = self.num_hidden_layers
        self.drop_attn_list = _to_bool_list(self.drop_attn_list, n)
        self.drop_mlp_list  = _to_bool_list(self.drop_mlp_list,  n)
