#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Azure GPU VM setup for MiLo DeepSeek quantization
# Tested on: NC48ads_A100_v4 and NC80ads_H100_v5 (Ubuntu 22.04)
#
# Usage:
#   bash quantization/scripts/setup_azure_gpu.sh
# -----------------------------------------------------------------------
set -euo pipefail

GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "Detected GPU: $GPU_TYPE"

# ------------------------------------------------------------------
# 1. System packages
# ------------------------------------------------------------------
sudo apt-get update -y
sudo apt-get install -y \
    build-essential git wget curl \
    python3-dev python3-pip python3-venv \
    ninja-build cmake \
    libssl-dev zlib1g-dev

# ------------------------------------------------------------------
# 2. CUDA 12.4 toolkit (if not already installed by Azure image)
# ------------------------------------------------------------------
if ! nvcc --version 2>/dev/null | grep -q "12\."; then
    echo "Installing CUDA 12.4 toolkit..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -y
    sudo apt-get install -y cuda-toolkit-12-4
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# ------------------------------------------------------------------
# 3. Conda environment
# ------------------------------------------------------------------
if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda"
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init bash
fi

eval "$(conda shell.bash hook)"
conda create -n milo python=3.11 -y
conda activate milo

# ------------------------------------------------------------------
# 4. PyTorch with CUDA 12.4
# ------------------------------------------------------------------
pip install --upgrade pip
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------------
# 5. Project dependencies
# ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pip install -r "$REPO_ROOT/quantization/requirements.txt"
pip install transformers==4.44.0 accelerate sentencepiece protobuf

# ------------------------------------------------------------------
# 6. Flash Attention 2  (A100 / H100 both supported)
# ------------------------------------------------------------------
pip install flash-attn --no-build-isolation

# ------------------------------------------------------------------
# 7. Optional: lm-evaluation-harness for benchmarking
# ------------------------------------------------------------------
pip install lm-eval

# ------------------------------------------------------------------
# 8. Verify
# ------------------------------------------------------------------
python - <<'EOF'
import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

echo ""
echo "Setup complete.  Activate with:  conda activate milo"
