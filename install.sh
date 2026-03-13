#!/usr/bin/env bash
# ============================================================================
# VAGEN — Simplified Install Script
# ============================================================================

set -euo pipefail

VAGEN_REPO="https://github.com/mll-lab-nu/VAGEN.git"
INSTALL_DIR="VAGEN"
NO_GPU=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --no-gpu) NO_GPU=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Ensure uv is installed
# ---------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "==> 'uv' not found. Installing via official script..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# 2. Clone VAGEN and initialize submodules
# ---------------------------------------------------------------------------
if [ ! -d "$INSTALL_DIR" ]; then
    echo "==> Cloning VAGEN..."
    git clone "$VAGEN_REPO" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
echo "==> Initializing verl submodule..."
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# 3. Generate pyproject.toml
# ---------------------------------------------------------------------------
echo "==> Writing pyproject.toml..."
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vagen"
version = "26.2.5"
description = "VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents"
requires-python = ">=3.10,<3.13"

dependencies = [
    "numpy>=1.26.0,<2.0.0",
    "pandas>=2.0.0",
    "pyarrow>=19.0.0",
    "transformers[hf_xet]>=4.51.0",
    "accelerate>=1.0.0",
    "datasets>=3.0.0",
    "peft>=0.14.0",
    "huggingface_hub>=0.25.0",
    "hf-transfer>=0.1.0",
    "trl==0.26.2",
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.0",
    "packaging>=20.0",
    "ray[default]>=2.41.0",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "gymnasium>=1.0.0",
    "gymnasium[toy-text]",
    "gym-sokoban",
    "Pillow>=10.0.0",
    "qwen-vl-utils>=0.0.8",
    "wandb>=0.19.0",
    "tensorboard>=2.14.0",
    "tqdm>=4.60.0",
    "dill>=0.3.0",
    "pybind11>=2.10.0",
    "pylatexenc>=2.0",
    "codetiming>=1.4.0",
    "uvicorn<0.41",
    "nvidia-ml-py>=12.560.30",
    "fastapi[standard]>=0.115.0",
    "optree>=0.13.0",
    "pydantic>=2.9",
    "grpcio>=1.62.1"
]

[project.optional-dependencies]
gpu = ["flashinfer-python==0.5.0", "liger-kernel"]
sglang = ["sglang[all]>=0.4.0", "torch-memory-saver"]
vllm = ["vllm>=0.6.0"]
inference = ["vagen[gpu]", "vagen[sglang]"]
EOF

# ---------------------------------------------------------------------------
# 4. Create and activate uv virtual environment
# ---------------------------------------------------------------------------
echo "==> Creating clean uv virtual environment..."
uv venv --python 3.12
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 5. Install PyTorch & Dependencies
# ---------------------------------------------------------------------------
if [ "$NO_GPU" = true ]; then
    echo "==> Installing CPU PyTorch and VAGEN core..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    uv pip install -e "."
else
    echo "==> Installing GPU PyTorch (CUDA 12.4) and VAGEN inference deps..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    uv pip install -e ".[inference]"
    
    echo "==> Attempting to install flash-attn..."
    uv pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn installation failed. Proceeding without it."
fi

# ---------------------------------------------------------------------------
# 6. Install Submodules
# ---------------------------------------------------------------------------
echo "==> Installing verl submodule..."
uv pip install --no-deps -e verl/

echo ""
echo "============================================"
echo "  VAGEN installation complete!"
echo "  Activate the environment with:"
echo "    cd $INSTALL_DIR && source .venv/bin/activate"
echo "============================================"