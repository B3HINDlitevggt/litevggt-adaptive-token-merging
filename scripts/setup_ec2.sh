#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/6] Checking GPU visibility"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found. Use a GPU AMI or install NVIDIA drivers first."
fi

echo "[2/6] Installing system packages"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  git \
  wget \
  curl \
  pkg-config \
  cmake \
  ninja-build \
  python3-pip \
  python3-venv

echo "[3/6] Creating virtual environment at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[4/6] Upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel

echo "[5/6] Installing Python dependencies"
python -m pip install -r "$ROOT_DIR/requirements.txt"

echo "[6/6] Installing Transformer Engine"
python -m pip install --no-build-isolation transformer_engine[pytorch]

if [[ -n "${CKPT_URL:-}" ]]; then
  mkdir -p "$ROOT_DIR/checkpoints"
  wget -O "$ROOT_DIR/checkpoints/te_dict.pt" "$CKPT_URL"
fi

echo "Setup complete."
echo "Activate with: source \"$VENV_DIR/bin/activate\""
