# Bhili ASR

Automatic Speech Recognition (ASR) model for Bhili (भीली), an Indo-Aryan language spoken by the Bhil people in western India.

This is a fine-tuned version of [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), trained on ~100 hours of Bhili Read and Spontaneous speech data. Try out the model [here](https://bhili-asr.ai4bharat.org/)!

## Repository Structure

```
bhili-asr/
├── inference/              # Everything needed to run the model
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.sh
│   ├── inference.py
│   ├── bhili_asr_app.py
│   ├── test_manifest.jsonl
│   ├── test_results.jsonl
│   └── test_audio/
│
├── train/                  # Training scripts
│   └── README.md
│
└── model/                  # Model and tokenizers
    ├── README.md
    ├── update_paths.py
    ├── NeMo.zip            # Patched NeMo toolkit (required)
    ├── bhili_asr_finetune_v1.nemo
    └── tokenizers.tar.gz
```

## Quick Start

See [inference/README.md](./inference/README.md) for detailed setup instructions.

## Model Download

Model files are available in:
- **This repo**: [model/](./model) folder
- **Hugging Face**: [ai4bharat/bhili-asr](https://huggingface.co/ai4bharat/bhili-asr)