#!/bin/bash
set -e

echo "=========================================="
echo "Bhili ASR Setup Script"
echo "=========================================="

ENV_NAME="bhili-asr"
PYTHON_VERSION="3.10"

if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo ""
echo "[1/6] Creating conda environment..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ""
echo "[2/6] Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing GPU version..."
    pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU version..."
    pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "[3/6] Extracting and installing NeMo toolkit..."
if [ ! -f "../model/NeMo.zip" ]; then
    echo "Error: NeMo.zip not found in ../model/"
    echo "Make sure you are running this script from the inference/ directory."
    exit 1
fi
unzip -q ../model/NeMo.zip -d .
cd NeMo
pip install -e .[asr]
cd ..

echo ""
echo "[4/6] Installing dependencies..."
pip install -r requirements.txt
pip install huggingface_hub==0.23.5 transformers==4.36.0
pip install numpy==1.26.4 pyarrow==14.0.1 datasets==2.14.0

echo ""
echo "[5/6] Extracting tokenizers..."
cd ../model
tar -xzvf tokenizers.tar.gz
cd ../inference

echo ""
echo "[6/6] Verifying installation..."
NEMO_PATH=$(python -c "import nemo; print(nemo.__file__)")
echo "NeMo path: $NEMO_PATH"

if [[ "$NEMO_PATH" == *"NeMo/nemo"* ]]; then
    echo "NeMo installation: OK (using local folder)"
else
    echo "Warning: NeMo may not be using the local folder!"
fi

python -c "
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
print('NeMo import: OK')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Update model paths:"
echo "     cd ../model"
echo "     python update_paths.py --root_dir \$(pwd)/tokenizers/tokenizers_v3"
echo ""
echo "  2. Run inference:"
echo "     cd ../inference"
echo "     conda activate $ENV_NAME"
echo "     python inference.py --model ../model/bhili_asr_finetune_v1_updated.nemo --audio your_audio.wav"
echo ""
echo "  Or run web interface:"
echo "     python bhili_asr_app.py"
echo ""