# Model Files

## Contents

- `bhili_asr_finetune_v1.nemo` - Model checkpoint
- `tokenizers.tar.gz` - Tokenizer files (extract before use)
- `NeMo.zip` - Patched NeMo toolkit (required)
- `update_paths.py` - Script to update tokenizer paths

## Setup

### 1. Extract Tokenizers

```bash
tar -xzvf tokenizers.tar.gz
```

### 2. Update Paths in Model

```bash
python update_paths.py --root_dir $(pwd)/tokenizers/tokenizers_v3
```

This creates `bhili_asr_finetune_v1_updated.nemo`.

### 3. Install NeMo Toolkit

> ⚠️ Use the provided `NeMo.zip`, not pip install.

```bash
cd ../inference
unzip ../model/NeMo.zip -d .
cd NeMo
pip install -e .[asr]
```

### 4. Use Updated Model

Use the generated `bhili_asr_finetune_v1_updated.nemo` for inference.

## Alternative Download

These files are also available on Hugging Face:

```bash
huggingface-cli download ai4bharat/bhili-asr --local-dir .
```