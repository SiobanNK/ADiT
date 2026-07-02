#!/bin/bash
# ============================================================
# ADiT installation script for Jean-Zay (IDRIS)
# Run this script in an interactive job or a SLURM batch job.
#
# Recommended interactive session to run this:
#   srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
#        --gres=gpu:1 --time=02:00:00 \
#        --account=<YOUR_PROJECT_ACCOUNT> --pty bash
#
# Usage:
#   bash install_adit_jean_zay.sh
# ============================================================

export PIP_CACHE_DIR=$WORK/.cache/pip
export TMPDIR=$WORK/.tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR


set -e

# ------------------------------------------------------------
# 0. Configuration — edit these if needed
# ------------------------------------------------------------
ENV_DIR="$WORK/envs/adit"          # where the venv will live
TORCH_VERSION="2.5.2"
CUDA_VERSION="cu124"
PYTHON_VERSION="3.10.4"               # Python 3.10 recommended

# ------------------------------------------------------------
# 1. Load Jean-Zay modules
#    Jean-Zay provides PyTorch via modules; we load the bare
#    minimum so we can build a clean venv on top.
# ------------------------------------------------------------
module purge
module load python/3.10.4          # adjust version if needed
module load cuda/12.4.0            # CUDA 12.1
module load cudnn/8.9.7.29-cuda    # cuDNN for CUDA 12.x

echo ">>> Modules loaded"
python --version
nvcc --version | head -1

# ------------------------------------------------------------
# 2. Create a virtual environment in $WORK
# ------------------------------------------------------------
if [ ! -d "$ENV_DIR" ]; then
    python -m venv "$ENV_DIR"
    echo ">>> Virtual environment created at $ENV_DIR"
else
    echo ">>> Virtual environment already exists at $ENV_DIR — skipping creation"
fi

source "$ENV_DIR/bin/activate"
pip install --upgrade pip

# ------------------------------------------------------------
# 3. Install PyTorch 2.1.2 + CUDA 12.1
# ------------------------------------------------------------
echo ">>> Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}..."
pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# 4. Install PyG (torch_scatter, torch_cluster, torch_sparse …)
#    Wheels are built for Torch 2.1.0 + cu121 + cp310 on Linux.
#    Note: pyg wheels use torch 2.1.0 in the filename even though
#    they work fine with 2.1.2.
# ------------------------------------------------------------
echo ">>> Downloading PyG wheels..."
WHL_DIR=$(mktemp -d)
BASE_URL="https://data.pyg.org/whl/torch-2.5.1%2Bcu124"

wget -q -P "$WHL_DIR" "${BASE_URL}/pyg_lib-0.4.0%2Bpt21cu121-cp310-cp310-linux_x86_64.whl"
wget -q -P "$WHL_DIR" "${BASE_URL}/torch_scatter-2.1.2%2Bpt21cu121-cp310-cp310-linux_x86_64.whl"
wget -q -P "$WHL_DIR" "${BASE_URL}/torch_spline_conv-1.2.2%2Bpt21cu121-cp310-cp310-linux_x86_64.whl"
wget -q -P "$WHL_DIR" "${BASE_URL}/torch_cluster-1.6.2%2Bpt21cu121-cp310-cp310-linux_x86_64.whl"
wget -q -P "$WHL_DIR" "${BASE_URL}/torch_sparse-0.6.18%2Bpt21cu121-cp310-cp310-linux_x86_64.whl"

echo ">>> Installing PyG wheels..."
pip install "$WHL_DIR"/*.whl
rm -rf "$WHL_DIR"

# ------------------------------------------------------------
# 5. Install remaining ADiT dependencies (batch 1)
# ------------------------------------------------------------
echo ">>> Installing core ADiT dependencies (batch 1)..."
pip install \
    rootutils \
    tqdm \
    biopython \
    foldcomp \
    lightning \
    omegaconf \
    hydra-core \
    pandas \
    dm-tree

# ------------------------------------------------------------
# 6. Install remaining ADiT dependencies (batch 2)
# ------------------------------------------------------------
echo ">>> Installing core ADiT dependencies (batch 2)..."
pip install \
    rich \
    biotite \
    atom3D \
    torchdrug \
    torcheval \
    spyrmsd \
    lifelines

# ------------------------------------------------------------
# 7. Quick smoke test
# ------------------------------------------------------------
echo ">>> Running smoke test..."
python - <<'EOF'
import torch
import torch_scatter, torch_cluster, torch_sparse
import lightning, hydra, biopython, biotite
print(f"PyTorch : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print("All key packages imported successfully.")
EOF

echo ""
echo "============================================================"
echo " Installation complete!"
echo " Activate your environment with:"
echo "   source ${ENV_DIR}/bin/activate"
echo "============================================================"
