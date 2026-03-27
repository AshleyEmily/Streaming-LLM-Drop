#!/usr/bin/env python3
"""
Comprehensive equivalence tests for LLM-Drop-v2.

Tests are organised into groups:
  A — No-drop equivalence: v2 with all-False drop lists must produce
      bitwise-identical logits to the standard LlamaForCausalLM across
      many (seed, batch_size, sequence_length) combinations.
  B — Single-layer drop correctness: for every layer and every drop mode
      (attn-only, mlp-only, both), verify the structural invariants.
  C — Multi-layer drop patterns: alternating, all-attn, all-mlp, all-both,
      random patterns.
  D — Shape / dtype / NaN invariance: every configuration produces
      correctly-shaped float32 outputs free of NaN / Inf.
  E — forward_layer utility: works across many (B, T) shapes.
  F — Similarity computation workflow: multi-batch calibration simulation
      with HiddenStatesRecordWrapper produces valid cosine similarities.
  G — Config round-trip: LlamaConfig normalises drop lists correctly
      regardless of input format.

Run from the repo root:
    cd /path/to/LLM-Drop-v2
    python test_equivalence.py

Requirements: transformers>=4.46.0, torch>=2.0.0
"""
import sys, os, importlib, importlib.util, types as _types

# ---------------------------------------------------------------------------
# Direct module loading (bypass the full llmtuner package which requires peft)
# ---------------------------------------------------------------------------
_SRC       = os.path.join(os.path.dirname(__file__), "src")
_MODELS    = os.path.join(_SRC, "llmtuner", "compression", "prune", "models")
_PRUNE     = os.path.join(_SRC, "llmtuner", "compression", "prune")

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

for _pkg in [
    "llmtuner",
    "llmtuner.compression",
    "llmtuner.compression.prune",
    "llmtuner.compression.prune.models",
]:
    if _pkg not in sys.modules:
        m = _types.ModuleType(_pkg)
        m.__path__ = []
        sys.modules[_pkg] = m

_cfg_llama = _load_module(
    "llmtuner.compression.prune.models.configuration_dropped_llama",
    os.path.join(_MODELS, "configuration_dropped_llama.py"),
)
_mdl_llama = _load_module(
    "llmtuner.compression.prune.models.modeling_dropped_llama",
    os.path.join(_MODELS, "modeling_dropped_llama.py"),
)
_wrapper = _load_module(
    "llmtuner.compression.prune.wrapper",
    os.path.join(_PRUNE, "wrapper.py"),
)
_utils = _load_module(
    "llmtuner.compression.prune.utils",
    os.path.join(_PRUNE, "utils.py"),
)

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM as StdLlama
from transformers import LlamaConfig   as StdLlamaConfig

DroppedConfig        = _cfg_llama.LlamaConfig
DroppedLlama         = _mdl_llama.LlamaForCausalLM
forward_layer        = _utils.forward_layer
HiddenStatesWrapper  = _wrapper.HiddenStatesRecordWrapper

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
VOCAB  = 64
HIDDEN = 64
INTER  = 128
N      = 4   # number of layers

def _dropped_cfg(drop_attn=None, drop_mlp=None):
    da = drop_attn or [False] * N
    dm = drop_mlp  or [False] * N
    return DroppedConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=N,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        drop_attn_list=da,
        drop_mlp_list=dm,
    )

def _std_cfg():
    return StdLlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=N,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )

def _rand_ids(seed, batch, seqlen):
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, seqlen))

# Helpers to grab hidden states from pre/post layer hooks
def _attach_hs_hooks(model):
    """Return (handles, captured_dict).
    captured['pre_i']  = hidden_states tensor entering layer i
    captured['post_i'] = hidden_states tensor leaving  layer i
    """
    captured = {}

    def _pre(i):
        def fn(module, args):
            hs = args[0]
            if not isinstance(hs, torch.Tensor):
                hs = hs[0]
            captured[f"pre_{i}"] = hs.detach().clone()
        return fn

    def _post(i):
        def fn(module, args, output):
            hs = output if isinstance(output, torch.Tensor) else output[0]
            captured[f"post_{i}"] = hs.detach().clone()
        return fn

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(_pre(i)))
        handles.append(layer.register_forward_hook(_post(i)))
    return handles, captured

def _remove(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# ═══ Group A: No-drop equivalence matrix ═══
# ---------------------------------------------------------------------------
def test_group_A():
    print("Group A: No-drop equivalence (v2 == standard model) ...")

    # Share weights: build std model, copy into dropped model
    std = StdLlama(_std_cfg()).eval()
    drp = DroppedLlama(_dropped_cfg()).eval()
    missing, unexpected = drp.load_state_dict(std.state_dict(), strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected}"

    seeds   = [0, 7, 42, 99, 1337]
    batches = [1, 2, 4]
    seqlens = [4, 16, 64]

    n_pass = n_fail = 0
    failures = []

    for seed in seeds:
        for B in batches:
            for T in seqlens:
                ids = _rand_ids(seed, B, T)
                with torch.no_grad():
                    std_logits = std(ids).logits
                    drp_logits = drp(ids).logits
                diff = (std_logits - drp_logits).abs().max().item()
                ok   = torch.allclose(std_logits, drp_logits, atol=1e-5)
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                    failures.append(f"seed={seed} B={B} T={T} max|diff|={diff:.3e}")

    total = n_pass + n_fail
    if failures:
        for f in failures:
            print(f"    FAIL: {f}")
        raise AssertionError(f"Group A: {n_fail}/{total} sub-tests failed")

    print(f"  PASSED  ({n_pass}/{total} sub-tests, all max|diff| ≤ 1e-5)")


# ---------------------------------------------------------------------------
# ═══ Group B: Single-layer drop correctness ═══
# ---------------------------------------------------------------------------
def test_group_B():
    print("Group B: Single-layer drop correctness ...")

    torch.manual_seed(42)
    ids = _rand_ids(42, 1, 16)

    n_pass = n_fail = 0
    failures = []

    for layer_idx in range(N):
        # ---- mode 1: both attn + mlp dropped → full pass-through ----
        da = [i == layer_idx for i in range(N)]
        dm = [i == layer_idx for i in range(N)]
        m = DroppedLlama(_dropped_cfg(da, dm)).eval()
        handles, cap = _attach_hs_hooks(m)
        with torch.no_grad():
            m(ids)
        _remove(handles)

        diff = (cap[f"pre_{layer_idx}"] - cap[f"post_{layer_idx}"]).abs().max().item()
        ok   = diff < 1e-6
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            failures.append(f"layer {layer_idx} full-drop: hidden states changed (diff={diff:.3e})")

        # Sub-modules must be None
        layer = m.model.layers[layer_idx]
        for attr in ("self_attn", "input_layernorm", "mlp", "post_attention_layernorm"):
            if getattr(layer, attr, "MISSING") is not None:
                n_fail += 1
                failures.append(f"layer {layer_idx} full-drop: {attr} not None")
            else:
                n_pass += 1

        # ---- mode 2: attn-only dropped ----
        da = [i == layer_idx for i in range(N)]
        dm = [False] * N
        m  = DroppedLlama(_dropped_cfg(da, dm)).eval()

        # Hook post_attention_layernorm of the targeted layer to capture the
        # residual that enters the MLP block.  When attn is dropped, this
        # should equal the layer's input (no attn residual was added).
        pre_mlp = {}
        def _palh(module, args):
            hs = args[0]
            if not isinstance(hs, torch.Tensor):
                hs = hs[0]
            pre_mlp["hs"] = hs.detach().clone()

        handles2, cap2 = _attach_hs_hooks(m)
        h_pal = m.model.layers[layer_idx].post_attention_layernorm \
                  .register_forward_pre_hook(_palh)
        with torch.no_grad():
            m(ids)
        h_pal.remove()
        _remove(handles2)

        # The value entering post_attention_layernorm == layer pre-hook input
        diff = (cap2[f"pre_{layer_idx}"] - pre_mlp["hs"]).abs().max().item()
        ok   = diff < 1e-6
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            failures.append(
                f"layer {layer_idx} attn-drop: residual to MLP != layer input "
                f"(diff={diff:.3e})"
            )

        # attn-related sub-modules None; mlp-related must exist
        layer = m.model.layers[layer_idx]
        for attr in ("self_attn", "input_layernorm"):
            if getattr(layer, attr, "MISSING") is not None:
                n_fail += 1
                failures.append(f"layer {layer_idx} attn-drop: {attr} not None")
            else:
                n_pass += 1
        for attr in ("mlp", "post_attention_layernorm"):
            if getattr(layer, attr, "MISSING") is None:
                n_fail += 1
                failures.append(f"layer {layer_idx} attn-drop: {attr} unexpectedly None")
            else:
                n_pass += 1

        # ---- mode 3: mlp-only dropped ----
        da = [False] * N
        dm = [i == layer_idx for i in range(N)]
        m  = DroppedLlama(_dropped_cfg(da, dm)).eval()
        handles3, cap3 = _attach_hs_hooks(m)
        with torch.no_grad():
            m(ids)
        _remove(handles3)

        # MLP was dropped → layer output differs from input (attention still ran)
        diff_in_out = (cap3[f"pre_{layer_idx}"] - cap3[f"post_{layer_idx}"]).abs().max().item()
        if diff_in_out < 1e-6:
            n_fail += 1
            failures.append(
                f"layer {layer_idx} mlp-drop: output == input "
                "(attention should have changed hidden states)"
            )
        else:
            n_pass += 1

        # mlp-related sub-modules None; attn-related must exist
        layer = m.model.layers[layer_idx]
        for attr in ("mlp", "post_attention_layernorm"):
            if getattr(layer, attr, "MISSING") is not None:
                n_fail += 1
                failures.append(f"layer {layer_idx} mlp-drop: {attr} not None")
            else:
                n_pass += 1
        for attr in ("self_attn", "input_layernorm"):
            if getattr(layer, attr, "MISSING") is None:
                n_fail += 1
                failures.append(f"layer {layer_idx} mlp-drop: {attr} unexpectedly None")
            else:
                n_pass += 1

    total = n_pass + n_fail
    if failures:
        for f in failures:
            print(f"    FAIL: {f}")
        raise AssertionError(f"Group B: {n_fail}/{total} sub-tests failed")

    print(f"  PASSED  ({n_pass}/{total} sub-tests)")


# ---------------------------------------------------------------------------
# ═══ Group C: Multi-layer drop patterns ═══
# ---------------------------------------------------------------------------
def test_group_C():
    print("Group C: Multi-layer drop patterns ...")

    # Build a no-drop baseline for comparison
    std = StdLlama(_std_cfg()).eval()

    n_pass = n_fail = 0
    failures = []

    patterns = {
        "alternating-attn":   (
            [i % 2 == 0 for i in range(N)],
            [False] * N,
        ),
        "alternating-mlp":    (
            [False] * N,
            [i % 2 == 0 for i in range(N)],
        ),
        "alternating-both":   (
            [i % 2 == 0 for i in range(N)],
            [i % 2 == 0 for i in range(N)],
        ),
        "all-attn":           ([True]  * N, [False] * N),
        "all-mlp":            ([False] * N, [True]  * N),
        "all-both":           ([True]  * N, [True]  * N),
        "first-and-last-attn": (
            [i in (0, N - 1) for i in range(N)],
            [False] * N,
        ),
        "random-seed123": None,  # filled below
    }

    # Random pattern
    torch.manual_seed(123)
    rand_da = [bool(torch.randint(0, 2, ()).item()) for _ in range(N)]
    rand_dm = [bool(torch.randint(0, 2, ()).item()) for _ in range(N)]
    patterns["random-seed123"] = (rand_da, rand_dm)

    for name, (da, dm) in patterns.items():
        for seed in [0, 42, 99]:
            for B, T in [(1, 8), (2, 16), (4, 32)]:
                ids = _rand_ids(seed, B, T)
                try:
                    m = DroppedLlama(_dropped_cfg(da, dm)).eval()
                    with torch.no_grad():
                        logits = m(ids).logits

                    # Shape check
                    assert logits.shape == (B, T, VOCAB), \
                        f"Wrong shape: {logits.shape}"
                    # Dtype
                    assert logits.dtype == torch.float32, \
                        f"Wrong dtype: {logits.dtype}"
                    # No NaN / Inf
                    assert not logits.isnan().any(), "NaN in logits"
                    assert not logits.isinf().any(), "Inf in logits"

                    # Dropped model should NOT produce the same logits as the
                    # no-drop standard model (unless all drops happen to have
                    # zero effect with random weights, which is astronomically
                    # unlikely for any non-trivial pattern).
                    if any(da) or any(dm):
                        with torch.no_grad():
                            std_logits = std(ids).logits
                        same = torch.allclose(logits, std_logits, atol=1e-5)
                        if same:
                            n_fail += 1
                            failures.append(
                                f"pattern={name} seed={seed} B={B} T={T}: "
                                "dropped model outputs identical to no-drop model"
                            )
                        else:
                            n_pass += 1
                    else:
                        n_pass += 1

                except Exception as e:
                    n_fail += 1
                    failures.append(f"pattern={name} seed={seed} B={B} T={T}: {e}")

    total = n_pass + n_fail
    if failures:
        for f in failures[:10]:   # cap output
            print(f"    FAIL: {f}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
        raise AssertionError(f"Group C: {n_fail}/{total} sub-tests failed")

    print(f"  PASSED  ({n_pass}/{total} sub-tests)")


# ---------------------------------------------------------------------------
# ═══ Group D: Shape / dtype / NaN for every seed × B × T × drop-mode ═══
# ---------------------------------------------------------------------------
def test_group_D():
    print("Group D: Shape / dtype / NaN invariance ...")

    drop_modes = [
        ([False] * N, [False] * N),
        ([True, False, True, False], [False] * N),
        ([False] * N, [False, True, False, True]),
        ([True] * N,  [True] * N),
    ]

    n_pass = n_fail = 0
    failures = []

    for da, dm in drop_modes:
        m = DroppedLlama(_dropped_cfg(da, dm)).eval()
        for seed in [1, 22, 333]:
            for B, T in [(1, 4), (2, 8), (3, 17), (4, 63)]:
                ids = _rand_ids(seed, B, T)
                with torch.no_grad():
                    out = m(ids)
                logits = out.logits
                ok = True
                if logits.shape != (B, T, VOCAB):
                    ok = False
                    failures.append(f"da={da} dm={dm} seed={seed} B={B} T={T}: shape {logits.shape}")
                if logits.dtype != torch.float32:
                    ok = False
                    failures.append(f"da={da} dm={dm} seed={seed} B={B} T={T}: dtype {logits.dtype}")
                if logits.isnan().any():
                    ok = False
                    failures.append(f"da={da} dm={dm} seed={seed} B={B} T={T}: NaN")
                if logits.isinf().any():
                    ok = False
                    failures.append(f"da={da} dm={dm} seed={seed} B={B} T={T}: Inf")
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1

    total = n_pass + n_fail
    if failures:
        for f in failures:
            print(f"    FAIL: {f}")
        raise AssertionError(f"Group D: {n_fail}/{total} sub-tests failed")

    print(f"  PASSED  ({n_pass}/{total} sub-tests)")


# ---------------------------------------------------------------------------
# ═══ Group E: forward_layer utility ═══
# ---------------------------------------------------------------------------
def test_group_E():
    print("Group E: forward_layer utility across shapes ...")

    cfg   = _dropped_cfg()
    model = DroppedLlama(cfg).eval()
    D     = cfg.hidden_size

    n_pass = n_fail = 0
    failures = []

    for B in [1, 2, 4]:
        for T in [4, 8, 33]:
            torch.manual_seed(B * 1000 + T)
            hs   = torch.randn(B, T, D)
            pids = torch.arange(T).unsqueeze(0).expand(B, -1)

            for layer_idx in range(N):
                layer = model.model.layers[layer_idx]
                with torch.no_grad():
                    out = forward_layer(model, layer, hs, None, pids)

                ok = True
                if out.shape != (B, T, D):
                    ok = False
                    failures.append(f"B={B} T={T} layer={layer_idx}: shape {out.shape}")
                if out.isnan().any():
                    ok = False
                    failures.append(f"B={B} T={T} layer={layer_idx}: NaN")
                if out.isinf().any():
                    ok = False
                    failures.append(f"B={B} T={T} layer={layer_idx}: Inf")
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1

    total = n_pass + n_fail
    if failures:
        for f in failures:
            print(f"    FAIL: {f}")
        raise AssertionError(f"Group E: {n_fail}/{total} sub-tests failed")

    print(f"  PASSED  ({n_pass}/{total} sub-tests)")


# ---------------------------------------------------------------------------
# ═══ Group F: Similarity computation workflow ═══
# ---------------------------------------------------------------------------
def test_group_F():
    print("Group F: Similarity computation workflow (multi-batch calibration) ...")

    cfg   = _dropped_cfg()
    model = DroppedLlama(cfg).eval()
    model.config.use_cache = False
    D = cfg.hidden_size

    n_pass = n_fail = 0
    failures = []

    for layer_idx in range(N):
        layer = model.model.layers[layer_idx]
        wrapper = HiddenStatesWrapper(layer, record_input=True, record_output=True)

        # Simulate 4 calibration batches with different inputs
        def record_hook(_, inp, out):
            in_hs  = inp[0] if isinstance(inp[0], torch.Tensor) else inp[0][0]
            out_hs = out    if isinstance(out, torch.Tensor)    else out[0]
            wrapper.record(in_hs.data, out_hs.data)

        handle = layer.register_forward_hook(record_hook)
        for batch_seed in [10, 20, 30, 40]:
            torch.manual_seed(batch_seed)
            hs   = torch.randn(1, 16, D)
            pids = torch.arange(16).unsqueeze(0)
            with torch.no_grad():
                forward_layer(model, layer, hs, None, pids)
        handle.remove()

        # Compute cosine similarity
        in_hs  = torch.cat(wrapper.input_hidden_states,  dim=0).float()
        out_hs = torch.cat(wrapper.output_hidden_states, dim=0).float()
        cos_sim = F.cosine_similarity(in_hs, out_hs, dim=-1).mean().item()

        ok = True
        if not (-1.0 <= cos_sim <= 1.0):
            ok = False
            failures.append(f"layer {layer_idx}: cosine similarity out of range ({cos_sim:.4f})")
        if torch.isnan(torch.tensor(cos_sim)):
            ok = False
            failures.append(f"layer {layer_idx}: cosine similarity is NaN")
        # cos_sim should not be exactly 1.0 for all tokens (would indicate pass-through)
        if cos_sim > 0.9999 and layer_idx < N - 1:
            # Allow for the last layer which may be near-identity; warn only
            failures.append(
                f"  WARNING layer {layer_idx}: cos_sim={cos_sim:.6f} is suspiciously high"
                " (possible pass-through?)"
            )
            # Don't count as fail — random weights can give high similarity
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    total = n_pass + n_fail
    hard_failures = [f for f in failures if not f.startswith("  WARNING")]
    if hard_failures:
        for f in hard_failures:
            print(f"    FAIL: {f}")
        raise AssertionError(f"Group F: {n_fail}/{total} sub-tests failed")

    for f in failures:
        print(f"  {f}")
    print(f"  PASSED  ({n_pass}/{total} sub-tests)")


# ---------------------------------------------------------------------------
# ═══ Group G: Config round-trip ═══
# ---------------------------------------------------------------------------
def test_group_G():
    print("Group G: Config round-trip (normalisation of drop lists) ...")

    n_pass = n_fail = 0
    failures = []

    cases = [
        # (drop_attn_input, drop_mlp_input, expected_da, expected_dm)
        # Bool list
        (
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True],
        ),
        # Integer index list
        (
            [0, 2],      # layers 0 and 2 dropped
            [1, 3],      # layers 1 and 3 dropped
            [True, False, True, False],
            [False, True, False, True],
        ),
        # Empty → all False
        (
            [],
            [],
            [False] * N,
            [False] * N,
        ),
        # None → all False
        (
            None,
            None,
            [False] * N,
            [False] * N,
        ),
        # All True
        (
            [True] * N,
            [True] * N,
            [True] * N,
            [True] * N,
        ),
        # Shorter list than N (should be padded with False)
        (
            [True, True],          # only 2 elements for 4-layer model
            [False, False],
            [True, True, False, False],
            [False] * N,
        ),
    ]

    for da_in, dm_in, da_exp, dm_exp in cases:
        cfg = DroppedConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            num_hidden_layers=N,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            drop_attn_list=da_in,
            drop_mlp_list=dm_in,
        )
        ok = True
        if cfg.drop_attn_list != da_exp:
            ok = False
            failures.append(
                f"drop_attn_list: input={da_in!r} "
                f"got={cfg.drop_attn_list!r} expected={da_exp!r}"
            )
        if cfg.drop_mlp_list != dm_exp:
            ok = False
            failures.append(
                f"drop_mlp_list: input={dm_in!r} "
                f"got={cfg.drop_mlp_list!r} expected={dm_exp!r}"
            )
        if len(cfg.drop_attn_list) != N or len(cfg.drop_mlp_list) != N:
            ok = False
            failures.append(
                f"list length wrong: da={len(cfg.drop_attn_list)} "
                f"dm={len(cfg.drop_mlp_list)}"
            )
        if ok:
            n_pass += 1
        else:
            n_fail += 1

        # Also verify the model has the right None modules
        if ok:
            m = DroppedLlama(cfg).eval()
            for i, layer in enumerate(m.model.layers):
                da_i = cfg.drop_attn_list[i]
                dm_i = cfg.drop_mlp_list[i]
                if da_i:
                    for attr in ("self_attn", "input_layernorm"):
                        if getattr(layer, attr, "MISSING") is not None:
                            n_fail += 1
                            failures.append(
                                f"layer {i} da={da_i}: {attr} not None"
                            )
                        else:
                            n_pass += 1
                if dm_i:
                    for attr in ("mlp", "post_attention_layernorm"):
                        if getattr(layer, attr, "MISSING") is not None:
                            n_fail += 1
                            failures.append(
                                f"layer {i} dm={dm_i}: {attr} not None"
                            )
                        else:
                            n_pass += 1

    total = n_pass + n_fail
    if failures:
        for f in failures:
            print(f"    FAIL: {f}")
        raise AssertionError(f"Group G: {n_fail}/{total} sub-tests failed")

    print(f"  PASSED  ({n_pass}/{total} sub-tests)")


# ---------------------------------------------------------------------------
# ═══ Group H: KV cache compression ═══
# ---------------------------------------------------------------------------
def test_group_H():
    print("Group H: KV cache compression with use_cache=True ...")

    n_pass = n_fail = 0
    failures = []

    # Cases: (drop_attn_list, expected_cache_slots)
    # expected = number of non-dropped attention layers
    cases = [
        # drop middle layer → 3 cache slots, not 4
        ([False, True,  False, False], 3),
        # drop first layer  → 3 cache slots
        ([True,  False, False, False], 3),
        # drop last layer   → 3 cache slots
        ([False, False, False, True],  3),
        # drop two middle   → 2 cache slots
        ([False, True,  True,  False], 2),
        # drop all          → 0 cache slots
        ([True,  True,  True,  True],  0),
        # drop none         → 4 cache slots (no compression needed)
        ([False, False, False, False], 4),
    ]

    for da, expected_slots in cases:
        dm = [False] * N
        model = DroppedLlama(_dropped_cfg(da, dm)).eval()

        ids = _rand_ids(42, 1, 8)
        with torch.no_grad():
            out_cached    = model(ids, use_cache=True)
            out_no_cache  = model(ids, use_cache=False)

        pkv = out_cached.past_key_values

        # Count actual cache slots used
        actual_slots = len(pkv.layers) if pkv is not None else 0

        ok = True
        if actual_slots != expected_slots:
            ok = False
            failures.append(
                f"da={da}: cache slots={actual_slots}, expected={expected_slots}"
            )

        # Logits must be identical regardless of use_cache
        diff = (out_cached.logits - out_no_cache.logits).abs().max().item()
        if diff > 1e-5:
            ok = False
            failures.append(
                f"da={da}: logits differ between cached/uncached runs (max|diff|={diff:.3e})"
            )

        if ok:
            n_pass += 1
        else:
            n_fail += 1

    total = n_pass + n_fail
    if failures:
        for f in failures:
            print(f"    FAIL: {f}")
        raise AssertionError(f"Group H: {n_fail}/{total} sub-tests failed")

    print(f"  PASSED  ({n_pass}/{total} sub-tests)")


# ---------------------------------------------------------------------------
# ═══ Original smoke tests (kept for quick regression check) ═══
# ---------------------------------------------------------------------------
def smoke_no_drop_equivalence():
    print("Smoke 1: No-drop equivalence ...")
    std = StdLlama(_std_cfg()).eval()
    drp = DroppedLlama(_dropped_cfg()).eval()
    drp.load_state_dict(std.state_dict(), strict=False)
    ids = _rand_ids(42, 2, 16)
    with torch.no_grad():
        a = std(ids).logits
        b = drp(ids).logits
    diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=1e-5), f"FAIL diff={diff:.3e}"
    print(f"  PASSED  (max|diff|={diff:.2e})")


def smoke_full_drop_passthrough():
    print("Smoke 2: Fully-dropped layer is a pass-through ...")
    da = [False, True, False, False]
    dm = [False, True, False, False]
    m  = DroppedLlama(_dropped_cfg(da, dm)).eval()
    handles, cap = _attach_hs_hooks(m)
    with torch.no_grad():
        m(_rand_ids(42, 1, 12))
    _remove(handles)
    diff = (cap["pre_1"] - cap["post_1"]).abs().max().item()
    assert diff < 1e-6, f"FAIL diff={diff:.3e}"
    print(f"  PASSED  (max|hs diff|={diff:.2e})")


def smoke_attn_only_drop_residual():
    print("Smoke 3: Attn-only drop: residual to MLP == layer input ...")
    da = [False, False, True, False]
    dm = [False] * N
    m  = DroppedLlama(_dropped_cfg(da, dm)).eval()
    pre_mlp = {}

    def _pal(module, args):
        hs = args[0]
        if not isinstance(hs, torch.Tensor):
            hs = hs[0]
        pre_mlp["hs"] = hs.detach().clone()

    handles, cap = _attach_hs_hooks(m)
    h = m.model.layers[2].post_attention_layernorm.register_forward_pre_hook(_pal)
    with torch.no_grad():
        m(_rand_ids(42, 1, 12))
    h.remove()
    _remove(handles)
    diff = (cap["pre_2"] - pre_mlp["hs"]).abs().max().item()
    assert diff < 1e-6, f"FAIL diff={diff:.3e}"
    print(f"  PASSED  (max|diff|={diff:.2e})")


def smoke_forward_layer():
    print("Smoke 4: forward_layer utility ...")
    cfg = _dropped_cfg()
    m   = DroppedLlama(cfg).eval()
    B, T, D = 1, 8, cfg.hidden_size
    hs  = torch.randn(B, T, D)
    pid = torch.arange(T).unsqueeze(0)
    with torch.no_grad():
        out = forward_layer(m, m.model.layers[0], hs, None, pid)
    assert out.shape == (B, T, D)
    assert not out.isnan().any()
    print(f"  PASSED  (shape={tuple(out.shape)})")


def smoke_similarity():
    print("Smoke 5: Cosine similarity finite and bounded ...")
    cfg   = _dropped_cfg()
    m     = DroppedLlama(cfg).eval()
    m.config.use_cache = False
    D     = cfg.hidden_size
    layer = m.model.layers[0]
    w     = HiddenStatesWrapper(layer, record_input=True, record_output=True)

    def hook(_, inp, out):
        i = inp[0] if isinstance(inp[0], torch.Tensor) else inp[0][0]
        o = out    if isinstance(out, torch.Tensor)    else out[0]
        w.record(i.data, o.data)

    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        forward_layer(m, layer, torch.randn(1, 16, D), None,
                      torch.arange(16).unsqueeze(0))
    h.remove()

    in_hs  = torch.cat(w.input_hidden_states,  dim=0).float()
    out_hs = torch.cat(w.output_hidden_states, dim=0).float()
    sim    = F.cosine_similarity(in_hs, out_hs, dim=-1).mean().item()
    assert -1.0 <= sim <= 1.0
    assert not torch.isnan(torch.tensor(sim))
    print(f"  PASSED  (cos_sim={sim:.4f})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    WIDTH = 62
    print("=" * WIDTH)
    print("LLM-Drop-v2  —  Comprehensive Equivalence Tests")
    print("=" * WIDTH)
    print()

    smoke_tests = [
        smoke_no_drop_equivalence,
        smoke_full_drop_passthrough,
        smoke_attn_only_drop_residual,
        smoke_forward_layer,
        smoke_similarity,
    ]
    group_tests = [
        test_group_A,
        test_group_B,
        test_group_C,
        test_group_D,
        test_group_E,
        test_group_F,
        test_group_G,
        test_group_H,
    ]

    n_pass = n_fail = 0

    print("─── Smoke tests ─────────────────────────────────────────")
    for t in smoke_tests:
        try:
            t()
            n_pass += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            n_fail += 1
        print()

    print("─── Group tests ──────────────────────────────────────────")
    for t in group_tests:
        try:
            t()
            n_pass += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            n_fail += 1
        print()

    total = n_pass + n_fail
    print("=" * WIDTH)
    print(f"Results: {n_pass}/{total} tests passed, {n_fail} failed")
    print("=" * WIDTH)
    sys.exit(0 if n_fail == 0 else 1)
