import torch


def get_streaming_position_ids(
    seq_len: int,
    n_init: int,
    n_local: int,
    device: torch.device,
) -> torch.Tensor:
    """Return cache-relative key position IDs for a calibration sequence.

    During live streaming inference, keys are stored without RoPE and RoPE is
    re-applied each step using sequential cache positions (0 .. kv_seq_len-1).
    This function computes, for each token index j in a full sequence of length
    seq_len, the cache position that key j *would* occupy under steady-state
    streaming:

      - Sink tokens  j < n_init      → cache position j   (never evicted)
      - Rolling window  j >= n_init  → cache position n_init + (j - n_init) % n_local

    When seq_len <= n_init + n_local (cache not yet full) the formula reduces to
    the identity mapping, so calibration matches standard attention exactly.

    Returns
    -------
    pos : LongTensor of shape (seq_len,)
        Cache-relative position for each token.  All values are in
        [0, n_init + n_local), so they are always valid indices into a cos/sin
        tensor of length seq_len (since seq_len >= n_init + n_local whenever
        the streaming window is active).
    """
    pos = torch.arange(seq_len, device=device, dtype=torch.long)
    sink_mask = pos < n_init
    rolling_pos = n_init + ((pos - n_init).clamp(min=0) % n_local)
    return torch.where(sink_mask, pos, rolling_pos)
