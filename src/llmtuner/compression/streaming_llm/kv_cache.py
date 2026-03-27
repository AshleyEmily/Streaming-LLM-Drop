# Ported from mit-han-lab/streaming-llm (2023).
# Original: streaming_llm/kv_cache.py
# Adapted for LLM-Drop-v2: no changes to logic; just moved into this package.

import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    """KV cache that retains attention sinks (first start_size tokens) and the
    most recent recent_size tokens, evicting everything in between.

    Each entry in past_key_values is a (k, v) pair of tensors with the
    sequence dimension at k_seq_dim / v_seq_dim (typically 2 for Llama/Mistral).
    """

    def __init__(self, start_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def _seq_len(self, past_key_values):
        """Return seq_len from the first non-None entry (dropped layers are None)."""
        for entry in past_key_values:
            if entry is not None:
                return entry[0].size(self.k_seq_dim)
        return 0

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = self._seq_len(past_key_values)
        if seq_len <= self.cache_size:
            return past_key_values
        return self.evict_range(past_key_values, self.start_size, seq_len - self.recent_size)

    def evict_for_space(self, past_key_values, num_coming):
        """Evict tokens so that seq_len + num_coming <= cache_size."""
        if past_key_values is None:
            return None
        seq_len = self._seq_len(past_key_values)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return self.evict_range(past_key_values, self.start_size, seq_len - self.recent_size + num_coming)

    def evict_range(self, past_key_values, start, end):
        """Remove tokens in the index range [start, end) from the cache.
        None entries (dropped attention layers) are passed through unchanged."""
        if past_key_values is None:
            return None
        seq_len = self._seq_len(past_key_values)
        assert start <= end <= seq_len
        result = []
        for entry in past_key_values:
            if entry is None:
                result.append(None)
            else:
                k, v = entry
                result.append([
                    torch.cat([self.k_slice(k, 0, start), self.k_slice(k, end, seq_len)], dim=self.k_seq_dim),
                    torch.cat([self.v_slice(v, 0, start), self.v_slice(v, end, seq_len)], dim=self.v_seq_dim),
                ])
        return result
