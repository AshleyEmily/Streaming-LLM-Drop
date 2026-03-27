#!/usr/bin/env python3
"""
StreamLLM attention variant tests.

Groups:
  A — Mask shape and pattern correctness
  B — Short-sequence equivalence: StreamLLM == full causal when L <= n_local
  C — Long-sequence divergence: StreamLLM mask blocks mid-range tokens
  D — Forward pass: correct output shape, no NaN/Inf, returns (tensor, None)
  E — _is_attention_module detection
  F — patch() wires all attention layers on a tiny Llama model

Run from the repo root:
    python test_streamllm.py
"""
import sys, os, importlib, importlib.util, types as _types

# ---------------------------------------------------------------------------
# Direct module loading (no peft required)
# ---------------------------------------------------------------------------
_SRC      = os.path.join(os.path.dirname(__file__), "src")
_MODELS   = os.path.join(_SRC, "llmtuner", "compression", "prune", "models")
_PRUNE    = os.path.join(_SRC, "llmtuner", "compression", "prune")
_ATTN_DIR = os.path.join(_PRUNE, "attention_variants")

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Stub out llmtuner package hierarchy so imports inside the modules resolve
for _pkg in [
    "llmtuner",
    "llmtuner.compression",
    "llmtuner.compression.prune",
    "llmtuner.compression.prune.models",
    "llmtuner.compression.prune.attention_variants",
]:
    if _pkg not in sys.modules:
        m = _types.ModuleType(_pkg)
        m.__path__ = [_ATTN_DIR if "attention_variants" in _pkg else _PRUNE]
        sys.modules[_pkg] = m

_slm = _load_module(
    "llmtuner.compression.prune.attention_variants.streamllm",
    os.path.join(_ATTN_DIR, "streamllm.py"),
)

_cfg_llama = _load_module(
    "llmtuner.compression.prune.models.configuration_dropped_llama",
    os.path.join(_MODELS, "configuration_dropped_llama.py"),
)
_mdl_llama = _load_module(
    "llmtuner.compression.prune.models.modeling_dropped_llama",
    os.path.join(_MODELS, "modeling_dropped_llama.py"),
)

import torch
import torch.nn as nn
import torch.nn.functional as F

streamllm_forward   = _slm.streamllm_forward
_is_attention_module = _slm._is_attention_module
patch               = _slm.patch
_rotate_half        = _slm._rotate_half
_apply_rope         = _slm._apply_rope
DroppedConfig       = _cfg_llama.LlamaConfig
DroppedLlama        = _mdl_llama.LlamaForCausalLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results = []

def check(name, condition, detail=""):
    ok = bool(condition)
    tag = PASS if ok else FAIL
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((name, ok))

def _build_streamllm_mask(L, n_init, n_local, dtype=torch.float32):
    """Reference mask: same logic as streamllm_forward."""
    col = torch.arange(L).unsqueeze(0)
    row = torch.arange(L).unsqueeze(1)
    causal   = col <= row
    is_sink  = col < n_init
    is_local = col >= (row - n_local + 1)
    allowed  = causal & (is_sink | is_local)
    bias = torch.zeros(L, L, dtype=dtype)
    bias.masked_fill_(~allowed, float("-inf"))
    return allowed, bias

VOCAB  = 64
HIDDEN = 64
INTER  = 128
HEADS  = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS
N_LAYERS = 2

def _tiny_cfg(**kw):
    return DroppedConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        max_position_embeddings=512,
        **kw,
    )

# ---------------------------------------------------------------------------
# Group A — Mask shape and pattern correctness
# ---------------------------------------------------------------------------
print("\n=== A: Mask shape and pattern ===")

def _check_mask(L, n_init, n_local):
    allowed, bias = _build_streamllm_mask(L, n_init, n_local)

    # 1. Shape
    check(f"A-shape L={L}", bias.shape == (L, L))

    # 2. Causal: upper triangle must be -inf
    for i in range(L):
        for j in range(i+1, L):
            if not (bias[i, j] == float("-inf")):
                check(f"A-causal L={L} i={i} j={j}", False, "future token visible")
                return
    check(f"A-causal L={L}", True)

    # 3. Sink tokens (j < n_init) always reachable from every position i >= j
    sink_ok = all(
        allowed[i, j].item()
        for i in range(L)
        for j in range(min(n_init, L))
        if i >= j
    )
    check(f"A-sinks L={L} n_init={n_init}", sink_ok)

    # 4. Local window: i attends to j in [i-n_local+1, i]
    local_ok = True
    for i in range(L):
        for j in range(max(0, i - n_local + 1), i + 1):
            if not allowed[i, j].item():
                local_ok = False
    check(f"A-local L={L} n_local={n_local}", local_ok)

    # 5. Mid-range tokens (not sink, not local) must be blocked when L > n_init+n_local
    if L > n_init + n_local:
        i = L - 1
        j = n_init  # first non-sink token
        blocked = not allowed[i, j].item()
        check(f"A-midrange-blocked L={L}", blocked,
              f"position {i} should NOT see position {j}")

_check_mask(L=16,  n_init=2,  n_local=4)
_check_mask(L=32,  n_init=4,  n_local=8)
_check_mask(L=128, n_init=8,  n_local=16)

# ---------------------------------------------------------------------------
# Group B — Short sequence: StreamLLM == full causal attention
# ---------------------------------------------------------------------------
print("\n=== B: Short-sequence equivalence (L <= n_local) ===")

def _full_causal_mask(L, dtype=torch.float32):
    mask = torch.zeros(L, L, dtype=dtype)
    mask.masked_fill_(torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1), float("-inf"))
    return mask

for L in [4, 8, 16]:
    n_init, n_local = 2, 32  # n_local >> L: every token is in the local window
    allowed_slm, bias_slm = _build_streamllm_mask(L, n_init, n_local)
    full = _full_causal_mask(L)
    check(f"B-equiv L={L} n_local={n_local}",
          torch.allclose(bias_slm, full),
          "StreamLLM mask should equal full causal when L <= n_local")

# n_local exactly equals L
L, n_local = 8, 8
n_init = 2
allowed_slm, bias_slm = _build_streamllm_mask(L, n_init, n_local)
full = _full_causal_mask(L)
check(f"B-equiv L=n_local={L}", torch.allclose(bias_slm, full))

# ---------------------------------------------------------------------------
# Group C — Long sequence: sink-only tokens are blocked mid-range
# ---------------------------------------------------------------------------
print("\n=== C: Long-sequence: mid-range tokens blocked ===")

L, n_init, n_local = 64, 4, 8
allowed, _ = _build_streamllm_mask(L, n_init, n_local)

# last row should see: 0..3 (sinks) and 56..63 (local window of 8)
i = L - 1
expected_visible = set(range(n_init)) | set(range(i - n_local + 1, i + 1))
actually_visible = {j for j in range(L) if allowed[i, j].item()}
check("C-last-row sinks visible", set(range(n_init)).issubset(actually_visible))
check("C-last-row local visible", set(range(i - n_local + 1, i + 1)).issubset(actually_visible))
check("C-last-row no mid-range", actually_visible == expected_visible,
      f"visible={sorted(actually_visible)}")

# First n_init rows: all within the local window anyway, so causal == full
for i in range(n_init):
    row_visible = {j for j in range(L) if allowed[i, j].item()}
    expected    = set(range(i + 1))
    check(f"C-sink-row-{i} correct", row_visible == expected)

# ---------------------------------------------------------------------------
# Group D — Forward pass: shape, no NaN/Inf, return type
# ---------------------------------------------------------------------------
print("\n=== D: Forward pass correctness ===")

class _MockRotaryEmb(nn.Module):
    """Identity RoPE: returns cos=1, sin=0 for any input."""
    def forward(self, x, position_ids):
        B, H, L, D = x.shape  # x is value tensor
        cos = torch.ones(1, 1, L, D, dtype=x.dtype, device=x.device)
        sin = torch.zeros(1, 1, L, D, dtype=x.dtype, device=x.device)
        return cos, sin

class _MockAttention(nn.Module):
    """Minimal attention module with the attributes StreamLLM needs.

    Simulates transformers 4.x style (num_heads attr + rotary_emb module).
    """
    def __init__(self, hidden, heads, kv_heads):
        super().__init__()
        assert hidden % heads == 0
        self.num_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.rotary_emb = _MockRotaryEmb()
        # StreamLLM params (set by patch)
        self._streamllm_n_init = 4
        self._streamllm_n_local = 8
        # Bind the forward
        import types
        self.forward = types.MethodType(streamllm_forward, self)

torch.manual_seed(42)
attn = _MockAttention(HIDDEN, HEADS, KV_HEADS)
attn.eval()

for B, L in [(1, 4), (1, 16), (2, 32), (1, 64)]:
    with torch.no_grad():
        x = torch.randn(B, L, HIDDEN)
        out, cache = attn(x)

    check(f"D-shape B={B} L={L}", out.shape == (B, L, HIDDEN))
    check(f"D-no-nan B={B} L={L}", not torch.isnan(out).any().item())
    check(f"D-no-inf B={B} L={L}", not torch.isinf(out).any().item())
    check(f"D-cache-None B={B} L={L}", cache is None)

# position_embeddings kwarg path (transformers 5.x style — no rotary_emb on module)
with torch.no_grad():
    x = torch.randn(1, 16, HIDDEN)
    L = 16
    cos = torch.ones(1, 1, L, HEAD_DIM)
    sin = torch.zeros(1, 1, L, HEAD_DIM)
    out2, _ = attn(x, position_embeddings=(cos, sin))
check("D-position_embeddings kwarg", out2.shape == (1, 16, HIDDEN))

# GQA expansion path: heads > kv_heads
attn_gqa = _MockAttention(HIDDEN, HEADS, 1)   # 1 KV head, 4 query heads
attn_gqa._streamllm_n_init = 2
attn_gqa._streamllm_n_local = 8
attn_gqa.eval()
with torch.no_grad():
    out_gqa, _ = attn_gqa(torch.randn(1, 16, HIDDEN))
check("D-gqa-expansion shape", out_gqa.shape == (1, 16, HIDDEN))

# config-based heads (transformers 5.x style: no num_heads attr, only config)
class _Cfg:
    num_attention_heads = HEADS
    num_key_value_heads = KV_HEADS

class _MockAttentionV5(nn.Module):
    """Simulates transformers 5.x: config-based heads, no rotary_emb, no num_heads attr."""
    def __init__(self, hidden, cfg):
        super().__init__()
        self.config = cfg
        self.head_dim = hidden // cfg.num_attention_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self._streamllm_n_init = 4
        self._streamllm_n_local = 8
        import types
        self.forward = types.MethodType(streamllm_forward, self)

attn_v5 = _MockAttentionV5(HIDDEN, _Cfg())
attn_v5.eval()
with torch.no_grad():
    x = torch.randn(1, 16, HIDDEN)
    cos = torch.ones(1, 1, 16, HEAD_DIM)
    sin = torch.zeros(1, 1, 16, HEAD_DIM)
    out_v5, _ = attn_v5(x, position_embeddings=(cos, sin))
check("D-v5-config-heads shape", out_v5.shape == (1, 16, HIDDEN))
check("D-v5-no-nan", not torch.isnan(out_v5).any().item())

# ---------------------------------------------------------------------------
# Group E — _is_attention_module detection
# ---------------------------------------------------------------------------
print("\n=== E: _is_attention_module detection ===")

# Should return True
check("E-MockAttention (heuristic)", _is_attention_module(_MockAttention(HIDDEN, HEADS, KV_HEADS)))

class LlamaAttention(nn.Module):
    q_proj = k_proj = v_proj = o_proj = None

check("E-LlamaAttention (known name)", _is_attention_module(LlamaAttention()))

class MistralSdpaAttention(nn.Module):
    q_proj = k_proj = v_proj = o_proj = None

check("E-MistralSdpaAttention (known name)", _is_attention_module(MistralSdpaAttention()))

class SomeCustomSelfAttention(nn.Module):
    q_proj = k_proj = v_proj = o_proj = None  # has all projections + "Attention" in name

check("E-heuristic match", _is_attention_module(SomeCustomSelfAttention()))

# Should return False
class FeedForward(nn.Module):
    pass

check("E-FeedForward rejected", not _is_attention_module(FeedForward()))

class AttentionLike(nn.Module):  # name matches but missing projections
    pass

check("E-AttentionLike-no-proj rejected", not _is_attention_module(AttentionLike()))

# ---------------------------------------------------------------------------
# Group F — patch() wires all attention layers on a tiny Llama model
# ---------------------------------------------------------------------------
print("\n=== F: patch() on tiny Llama model ===")

cfg = _tiny_cfg()
with torch.no_grad():
    model = DroppedLlama(cfg)
    model.eval()

class _FakeArgs:
    attention_variant = "streamllm"
    streamllm_n_init = 8
    streamllm_n_local = 16

args = _FakeArgs()

try:
    patch(model, args)
    patch_ok = True
except Exception as e:
    patch_ok = False
    print(f"    patch() raised: {e}")

check("F-patch-no-error", patch_ok)

# Config written
check("F-config-n_init",  model.config.streamllm_n_init == 8)
check("F-config-n_local", model.config.streamllm_n_local == 16)

# All attention layers have the patched forward
patched_layers = 0
for layer in model.model.layers:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        continue
    has_params = hasattr(attn, "_streamllm_n_init") and hasattr(attn, "_streamllm_n_local")
    is_slm     = attn.__dict__.get("forward") is not None  # bound method override
    if has_params and is_slm:
        patched_layers += 1

check("F-all-layers-patched", patched_layers == N_LAYERS,
      f"expected {N_LAYERS}, got {patched_layers}")

# Forward pass through patched model produces correct shape and no NaN
with torch.no_grad():
    ids = torch.randint(0, VOCAB, (1, 20))
    out = model(ids)

check("F-model-forward-shape", out.logits.shape == (1, 20, VOCAB))
check("F-model-forward-no-nan", not torch.isnan(out.logits).any().item())
check("F-model-forward-no-inf", not torch.isinf(out.logits).any().item())

# ---------------------------------------------------------------------------
# Group G — model.model.forward patch: skips mask, correct shape, no NaN
# ---------------------------------------------------------------------------
print("\n=== G: _streamllm_model_forward patch ===")

from llmtuner.compression.prune.attention_variants.streamllm import _streamllm_model_forward

cfg2 = _tiny_cfg()
with torch.no_grad():
    model2 = DroppedLlama(cfg2)
    model2.eval()

args2 = _FakeArgs()
args2.streamllm_n_init = 4
args2.streamllm_n_local = 8

patch(model2, args2)

# model.model.forward should be replaced
check("G-model-forward-replaced",
      model2.model.__dict__.get("forward") is not None,
      "forward should be instance-bound (in __dict__)")

# Old forward preserved
check("G-old-forward-saved", hasattr(model2.model, "_streamllm_old_forward"))

# Forward pass with longer sequence (> n_init + n_local = 12)
with torch.no_grad():
    ids2 = torch.randint(0, VOCAB, (1, 64))
    out2 = model2(ids2)

check("G-long-seq-shape", out2.logits.shape == (1, 64, VOCAB))
check("G-long-seq-no-nan", not torch.isnan(out2.logits).any().item())
check("G-long-seq-no-inf", not torch.isinf(out2.logits).any().item())

# Batch size > 1
with torch.no_grad():
    ids3 = torch.randint(0, VOCAB, (2, 32))
    out3 = model2(ids3)
check("G-batch2-shape", out3.logits.shape == (2, 32, VOCAB))
check("G-batch2-no-nan", not torch.isnan(out3.logits).any().item())

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
passed = sum(1 for _, ok in _results if ok)
total  = len(_results)
label  = PASS if passed == total else FAIL
print(f"  [{label}] {passed}/{total} tests passed")
if passed < total:
    print("\n  Failed tests:")
    for name, ok in _results:
        if not ok:
            print(f"    - {name}")
print("=" * 50)
sys.exit(0 if passed == total else 1)
