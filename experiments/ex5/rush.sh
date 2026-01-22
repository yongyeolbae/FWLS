
set -euo pipefail



ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[KR] Working dir: $ROOT_DIR"
echo "[EN] Python: $PYTHON_BIN"
echo ""


ONLY="${ONLY:-all}"

run_one () {
  local name="$1"
  local script="$2"
  local cfg="$3"

  echo "============================================================"
  echo "[RUN] ${name}"
  echo "  script: ${script}"
  echo "  config: ${cfg}"
  echo "============================================================"
  $PYTHON_BIN "$script" --config "$cfg"
  echo ""
}

if [[ "$ONLY" == "all" || "$ONLY" == "asc" || "$ONLY" == "ascformer" ]]; then
  run_one "ASCFormer-like" \
    "scripts/train_ascformer_like.py" \
    "configs/ascformer_pair_rule.yaml"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "caftb" || "$ONLY" == "caftbnet" ]]; then
  run_one "CAFTB-Net" \
    "scripts/train_caftbnet.py" \
    "configs/caftb_pair_rule.yaml"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "pscc" || "$ONLY" == "psccnet" ]]; then
  run_one "PSCCNet-Lite" \
    "scripts/train_psccnet_lite.py" \
    "configs/psccnet_pair_rule.yaml"
fi

echo "[DONE] exp5_supp_training finished."

