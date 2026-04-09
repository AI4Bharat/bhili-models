# Bhili ASR - Inference

Setup and run inference on the Bhili ASR model.

## Setup

### 1. Create Environment

```bash
conda create -n bhili-asr python=3.10 -y
conda activate bhili-asr
```

### 2. Install NeMo Toolkit

> ⚠️ **Important:** This model requires a custom NeMo toolkit with multilingual tokenizer support. Do **NOT** install NeMo from pip or the official NVIDIA repository. Use the `NeMo/` folder provided in this repo.

```bash
cd NeMo
pip install -e .[asr]
cd ..
```

Verify correct NeMo is installed:

```bash
python -c "import nemo; print(nemo.__file__)"
```

Output should point to your local `NeMo/` folder, NOT system packages.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter version errors:

```bash
pip install huggingface_hub==0.23.5 transformers==4.36.0
pip install numpy==1.26.4 pyarrow==14.0.1 datasets==2.14.0
```

### 4. Download Model

**Option A: From this repo**

```bash
cd ../model
tar -xzvf tokenizers.tar.gz
cd ../inference
```

**Option B: From Hugging Face**

```bash
pip install huggingface_hub
huggingface-cli download sanjay73/bhili-asr --local-dir ../model
cd ../model && tar -xzvf tokenizers.tar.gz && cd ../inference
```

### 5. Update Model Paths

```bash
cd ../model
python update_paths.py --root_dir $(pwd)/tokenizers/tokenizers_v3
cd ../inference
```

This creates `bhili_asr_finetune_v1_updated.nemo` with correct paths.

### 6. Verify Setup

Your directory structure should look like:

```
bhili-asr/
├── inference/
│   ├── NeMo/                 # Patched NeMo toolkit
│   ├── inference.py
│   ├── bhili_asr_app.py
│   └── ...
└── model/
    ├── bhili_asr_finetune_v1.nemo
    ├── bhili_asr_finetune_v1_updated.nemo  # Created after step 5
    ├── tokenizers/
    │   └── tokenizers_v3/
    │       ├── as_256/
    │       ├── bn_256/
    │       ├── hi_256/
    │       ├── mr_256/
    │       └── ...
    └── update_paths.py
```

## Usage

### Web Interface

```bash
python bhili_asr_app.py
```

Open http://localhost:7860

### Command Line

```bash
# Single file
python inference.py --model ../model/bhili_asr_finetune_v1_updated.nemo --audio audio.wav

# Multiple files
python inference.py --model ../model/bhili_asr_finetune_v1_updated.nemo --audio file1.wav file2.wav

# Directory of audio files
python inference.py --model ../model/bhili_asr_finetune_v1_updated.nemo --audio_dir ./audio_folder/

# Save results to JSON
python inference.py --model ../model/bhili_asr_finetune_v1_updated.nemo --audio audio.wav --output results.json
```

### Python API

```python
from inference import BhiliASR

asr = BhiliASR("../model/bhili_asr_finetune_v1_updated.nemo")
text = asr.transcribe("audio.wav")
print(text)
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `KeyError: 'dir'` | Run `update_paths.py` with correct absolute path |
| `MultilingualTokenizer not found` | Install NeMo from the provided `NeMo/` folder, not pip |
| `huggingface_hub` errors | `pip install huggingface_hub==0.23.5 transformers==4.36.0` |
| `pyarrow` errors | `pip install numpy==1.26.4 pyarrow==14.0.1 datasets==2.14.0` |
| Wrong NeMo being used | Check `python -c "import nemo; print(nemo.__file__)"` points to local folder |