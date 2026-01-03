#!/bin/bash
# Setup script for Genie training on a rented server
# This script:
# 1. Clones the repository (if not already present)
# 2. Installs uv (if not present)
# 3. Creates a uv-based Python environment
# 4. Installs all required packages
# 5. Downloads the pong dataset
# 6. Creates necessary directories
# 7. Verifies the setup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/skr3178/Genie_google.git"
REPO_NAME="Genie_google"
WORK_DIR="${1:-$HOME}"  # Allow user to specify working directory, default to $HOME
PROJECT_DIR="${WORK_DIR}/${REPO_NAME}"
ENV_NAME="genie"
PYTHON_VERSION="3.10"

echo -e "${GREEN}=== Genie Training Environment Setup ===${NC}"
echo "Working directory: ${WORK_DIR}"
echo "Project directory: ${PROJECT_DIR}"
echo ""

# Step 0: Clone repository if needed
echo -e "${YELLOW}[0/7] Setting up repository...${NC}"
if [ -d "${PROJECT_DIR}" ] && [ -d "${PROJECT_DIR}/.git" ]; then
    echo -e "${GREEN}✓ Repository already exists at ${PROJECT_DIR}${NC}"
    echo "  Updating repository..."
    cd "${PROJECT_DIR}"
    git pull || echo -e "${YELLOW}  Warning: Could not update repository${NC}"
else
    if [ -d "${PROJECT_DIR}" ]; then
        echo -e "${YELLOW}Directory ${PROJECT_DIR} exists but is not a git repository${NC}"
        read -p "Remove and clone fresh? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "${PROJECT_DIR}"
        else
            echo -e "${RED}Aborting. Please remove ${PROJECT_DIR} or choose a different directory.${NC}"
            exit 1
        fi
    fi
    
    echo "Cloning repository from ${REPO_URL}..."
    cd "${WORK_DIR}"
    git clone "${REPO_URL}" "${REPO_NAME}"
    echo -e "${GREEN}✓ Repository cloned successfully${NC}"
fi

cd "${PROJECT_DIR}"
DATA_DIR="${PROJECT_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/logs"
echo ""

# Step 1: Check and install uv
echo -e "${YELLOW}[1/7] Checking for uv...${NC}"
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Error: Failed to install uv. Please install manually.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ uv installed successfully${NC}"
else
    echo -e "${GREEN}✓ uv is already installed${NC}"
fi
echo ""

# Step 2: Create uv environment
echo -e "${YELLOW}[2/7] Creating uv Python environment...${NC}"
cd "${PROJECT_DIR}"

# Remove existing environment if it exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Create new environment
uv venv --python "${PYTHON_VERSION}" .venv
echo -e "${GREEN}✓ Environment created${NC}"
echo ""

# Step 3: Activate environment and install packages
echo -e "${YELLOW}[3/7] Installing packages from requirements.txt...${NC}"
source .venv/bin/activate

# Upgrade pip first (uv creates venv with pip)
pip install --upgrade pip

# Install packages using uv pip (faster than regular pip)
uv pip install -r requirements.txt

echo -e "${GREEN}✓ All packages installed${NC}"
echo ""

# Step 4: Create necessary directories
echo -e "${YELLOW}[4/7] Creating necessary directories...${NC}"
mkdir -p "${DATA_DIR}"
mkdir -p "${CHECKPOINT_DIR}/tokenizer"
mkdir -p "${CHECKPOINT_DIR}/lam"
mkdir -p "${CHECKPOINT_DIR}/dynamics"
mkdir -p "${LOG_DIR}/tokenizer"
mkdir -p "${LOG_DIR}/lam"
mkdir -p "${LOG_DIR}/dynamics"
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 5: Download pong dataset
echo -e "${YELLOW}[5/7] Downloading pong dataset...${NC}"
if [ ! -f "${DATA_DIR}/pong_frames.h5" ]; then
    echo "Downloading pong dataset from HuggingFace..."
    python download_dataset.py datasets --pattern "*pong*.h5" --out "${DATA_DIR}"
    
    if [ -f "${DATA_DIR}/pong_frames.h5" ]; then
        echo -e "${GREEN}✓ Pong dataset downloaded successfully${NC}"
        # Show dataset size
        if command -v du &> /dev/null; then
            SIZE=$(du -h "${DATA_DIR}/pong_frames.h5" | cut -f1)
            echo "  Dataset size: ${SIZE}"
        fi
    else
        echo -e "${RED}Warning: pong_frames.h5 not found after download${NC}"
        echo "  Please check the download manually"
    fi
else
    echo -e "${GREEN}✓ Pong dataset already exists${NC}"
fi
echo ""

# Step 6: Verify setup
echo -e "${YELLOW}[6/7] Verifying setup...${NC}"

# Check Python version
PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: ${PYTHON_VER}"

# Check critical packages
echo "Checking critical packages..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || echo -e "  ${RED}✗ PyTorch not found${NC}"
python -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')" || echo -e "  ${RED}✗ NumPy not found${NC}"
python -c "import h5py; print(f'  ✓ h5py {h5py.__version__}')" || echo -e "  ${RED}✗ h5py not found${NC}"
python -c "import yaml; print('  ✓ PyYAML')" || echo -e "  ${RED}✗ PyYAML not found${NC}"
python -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" || echo -e "  ${RED}✗ OpenCV not found${NC}"
python -c "import imageio; print(f'  ✓ imageio {imageio.__version__}')" || echo -e "  ${RED}✗ imageio not found${NC}"

# Check dataset
if [ -f "${DATA_DIR}/pong_frames.h5" ]; then
    echo -e "  ${GREEN}✓ Pong dataset found${NC}"
    # Try to read dataset info
    python -c "
import h5py
import sys
try:
    with h5py.File('${DATA_DIR}/pong_frames.h5', 'r') as f:
        if 'frames' in f:
            frames = f['frames']
            if hasattr(frames, 'shape'):
                print(f'    Dataset shape: {frames.shape}')
            elif hasattr(frames, '__len__'):
                print(f'    Dataset length: {len(frames)}')
        print('    ✓ Dataset is readable')
except Exception as e:
    print(f'    ${RED}✗ Error reading dataset: {e}${NC}')
    sys.exit(1)
" || echo -e "    ${RED}✗ Could not read dataset${NC}"
else
    echo -e "  ${RED}✗ Pong dataset not found${NC}"
fi

# Check CUDA availability
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    echo -e "  ${GREEN}✓ CUDA available (version: ${CUDA_VER}, GPUs: ${GPU_COUNT})${NC}"
else
    echo -e "  ${YELLOW}⚠ CUDA not available (will use CPU)${NC}"
fi

echo ""

# Final instructions
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Repository location: ${PROJECT_DIR}"
echo ""
echo "To activate the environment, run:"
echo "  cd ${PROJECT_DIR}"
echo "  source .venv/bin/activate"
echo ""
echo "To start training the tokenizer on pong dataset:"
echo "  cd ${PROJECT_DIR}"
echo "  source .venv/bin/activate"
echo "  python scripts/train_tokenizer.py --dataset pong"
echo ""
echo "To start training LAM on pong dataset:"
echo "  cd ${PROJECT_DIR}"
echo "  source .venv/bin/activate"
echo "  python scripts/train_lam.py --dataset pong"
echo ""
echo "To start training dynamics model (after tokenizer and LAM are trained):"
echo "  cd ${PROJECT_DIR}"
echo "  source .venv/bin/activate"
echo "  python scripts/train_dynamics.py --dataset pong \\"
echo "    --tokenizer_path checkpoints/tokenizer/checkpoint_final.pt \\"
echo "    --lam_path checkpoints/lam/checkpoint_final.pt"
echo ""
echo "Project structure:"
echo "  ${PROJECT_DIR}/"
echo "  ├── data/          - Dataset files"
echo "  ├── checkpoints/   - Model checkpoints"
echo "  │   ├── tokenizer/"
echo "  │   ├── lam/"
echo "  │   └── dynamics/"
echo "  ├── logs/          - Training logs"
echo "  ├── scripts/       - Training scripts"
echo "  └── configs/        - Configuration files"
echo ""
