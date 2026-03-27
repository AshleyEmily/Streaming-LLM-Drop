#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-env-test
#SBATCH --partition=gpuqs               # same partition as prune_and_eval.slurm
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ashley.irawan@sjsu.edu
#SBATCH --mail-type=END
#SBATCH --output=logs/env_test_%j.out
#SBATCH --error=logs/env_test_%j.err

set -euo pipefail

# ── Activate your environment ──────────────────────────────────────────────────
# module load cuda/12.1
# module load anaconda/2023.09
# conda activate llmdrop

module load python3/3.12.12
module load ml/torch/2.6

python -m venv ~/venvs/llmdrop
source ~/venvs/llmdrop/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install peft lm_eval trl

cd ~/LLM-Drop-v2
mkdir -p logs

PASS=0; FAIL=0
ok()   { echo "  [PASS] $1"; ((PASS++)) || true; }
fail() { echo "  [FAIL] $1"; ((FAIL++)) || true; }

echo "========================================"
echo "  LLM-Drop environment test"
echo "  Node : ${SLURM_NODELIST:-local}"
echo "  Job  : ${SLURM_JOB_ID:-n/a}"
echo "========================================"

# ── 1. Python ──────────────────────────────────────────────────────────────────
echo ""
echo "[1] Python"
PY_VER=$(python --version 2>&1)
if [[ "$PY_VER" == "Python 3.12"* ]]; then
    ok "Python version: $PY_VER"
else
    fail "Python 3.12 required, found: $PY_VER"
fi

# ── 2. Key packages ────────────────────────────────────────────────────────────
echo ""
echo "[2] Package versions"

check_pkg() {
    local pkg=$1 import=$2
    VER=$(python -c "import $import; print($import.__version__)" 2>&1) \
        && ok "$pkg $VER" \
        || fail "$pkg not importable"
}

check_pkg torch          torch
check_pkg transformers   transformers
check_pkg accelerate     accelerate
check_pkg peft           peft
check_pkg datasets       datasets
check_pkg lm_eval        lm_eval

# ── 3. CUDA ────────────────────────────────────────────────────────────────────
echo ""
echo "[3] CUDA"
python - <<'EOF'
import torch, sys

ok   = lambda s: print(f"  [PASS] {s}")
fail = lambda s: (print(f"  [FAIL] {s}"), sys.exit(1))

if torch.cuda.is_available():
    ok(f"CUDA available — {torch.cuda.device_count()} device(s)")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        ok(f"  GPU {i}: {props.name}  ({props.total_memory // 1024**3} GB)")
else:
    fail("CUDA not available — check drivers and CUDA_VISIBLE_DEVICES")
EOF

# ── 4. bf16 support ────────────────────────────────────────────────────────────
echo ""
echo "[4] bf16 support"
python - <<'EOF'
import torch, sys
ok   = lambda s: print(f"  [PASS] {s}")
fail = lambda s: print(f"  [FAIL] {s}")

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8:
        ok(f"bf16 supported (compute capability {cap[0]}.{cap[1]})")
    else:
        fail(f"bf16 NOT supported (compute capability {cap[0]}.{cap[1]} < 8.0) — remove --bf16 flag")
else:
    fail("no GPU — cannot check bf16")
EOF

# ── 5. accelerate multi-GPU smoke test ────────────────────────────────────────
echo ""
echo "[5] accelerate launch (single-GPU smoke test)"
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 - <<'EOF'
from accelerate import Accelerator
import torch
acc = Accelerator()
t = torch.tensor([1.0], device=acc.device)
result = acc.reduce(t, reduction="mean")
print(f"  [PASS] accelerate reduce OK — device: {acc.device}, result: {result.item():.1f}")
EOF

# ── 6. LLM-Drop source importable ─────────────────────────────────────────────
echo ""
echo "[6] LLM-Drop source"
python - <<'EOF'
import sys
sys.path.insert(0, "src")
ok   = lambda s: print(f"  [PASS] {s}")
fail = lambda s: (print(f"  [FAIL] {s}"), sys.exit(1))

try:
    from llmtuner.compression.prune.utils import forward_layer, auto_map, CUSTOM_FILE
    ok("llmtuner.compression.prune.utils imported")
except Exception as e:
    fail(f"llmtuner import failed: {e}")

try:
    from llmtuner.compression.prune.models.modeling_dropped_llama import LlamaForCausalLM
    ok("modeling_dropped_llama imported")
except Exception as e:
    fail(f"modeling_dropped_llama import failed: {e}")

try:
    from llmtuner.compression.prune.models.modeling_dropped_mistral import MistralForCausalLM
    ok("modeling_dropped_mistral imported")
except Exception as e:
    fail(f"modeling_dropped_mistral import failed: {e}")

try:
    from llmtuner.compression.prune.models.modeling_dropped_gemma2 import Gemma2ForCausalLM
    ok("modeling_dropped_gemma2 imported")
except Exception as e:
    fail(f"modeling_dropped_gemma2 import failed: {e}")
EOF

# ── 7. lm_eval task registry ──────────────────────────────────────────────────
echo ""
echo "[7] lm_eval task availability"
python - <<'EOF'
ok   = lambda s: print(f"  [PASS] {s}")
fail = lambda s: print(f"  [FAIL] {s}")

try:
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    for task in ["boolq", "rte", "piqa", "mmlu", "winogrande", "hellaswag", "arc_challenge", "gsm8k", "openbookqa"]:
        try:
            tm.load_task_or_group([task])
            ok(task)
        except Exception as e:
            fail(f"{task}: {e}")
except Exception as e:
    fail(f"TaskManager failed: {e}")
EOF

# ── 8. Output paths writable ──────────────────────────────────────────────────
echo ""
echo "[8] Output directory writable"
TEST_DIR="../results_prune/.write_test_$$"
mkdir -p "$TEST_DIR" && rm -rf "$TEST_DIR" \
    && echo "  [PASS] ../results_prune/ is writable" \
    || echo "  [FAIL] ../results_prune/ is NOT writable"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  PASSED: ${PASS}   FAILED: ${FAIL}"
if [[ $FAIL -gt 0 ]]; then
    echo "  Fix the failures above before submitting prune_and_eval.slurm"
    echo "========================================"
    exit 1
else
    echo "  Environment looks good — ready to run prune_and_eval.slurm"
    echo "========================================"
fi
