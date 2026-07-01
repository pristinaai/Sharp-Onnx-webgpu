#!/usr/bin/env bash
# Set up Python environment and clone apple/ml-sharp for ONNX export.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHARP_DIR="${SHARP_DIR:-$ROOT/vendor/ml-sharp}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv-export}"

echo "==> Root: $ROOT"
echo "==> SHARP repo: $SHARP_DIR"

if [[ ! -d "$SHARP_DIR/.git" ]]; then
  echo "==> Cloning apple/ml-sharp..."
  mkdir -p "$(dirname "$SHARP_DIR")"
  git clone --depth 1 https://github.com/apple/ml-sharp "$SHARP_DIR"
else
  echo "==> ml-sharp already cloned at $SHARP_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing export dependencies..."
pip install --upgrade pip wheel setuptools

# Core export deps (gsplat is only needed for rendering, not ONNX export)
pip install torch torchvision timm pillow numpy scipy onnx onnxruntime onnxscript

# Install SHARP package without pulling gsplat (rendering dependency)
pip install -e "$SHARP_DIR" --no-deps
pip install click plyfile matplotlib imageio

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Export ONNX:"
echo "  python $ROOT/scripts/export_sharp_onnx.py \\"
echo "    --sharp-repo $SHARP_DIR \\"
echo "    --output $ROOT/web/public/models/sharp_web_predictor.onnx \\"
echo "    --verbose"
