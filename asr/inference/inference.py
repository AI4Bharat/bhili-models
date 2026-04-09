"""
Bhili ASR Inference Script
Usage:
    python inference.py --audio file.wav
    python inference.py --audio file1.wav file2.wav
    python inference.py --audio_dir ./audio_folder
    python inference.py --audio file.wav --output results.json
"""

import argparse
import json
import os
import torch
from pathlib import Path

LANGUAGE_ID = "mr"


class BhiliASR:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def load_model(self):
        if self.model is None:
            print(f"Loading model from {self.model_path}...")
            from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
            self.model = EncDecHybridRNNTCTCBPEModel.restore_from(
                self.model_path,
                map_location=self.device
            )
            self.model.eval()
            self.model.freeze()
            print(f"Model loaded on {self.device}")
        return self.model
    
    def transcribe(self, audio_path):
        model = self.load_model()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with torch.no_grad():
            result = model.transcribe(
                [audio_path],
                batch_size=1,
                language_id=LANGUAGE_ID
            )
        
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, list):
            result = result[0] if result else ""
        if hasattr(result, "text"):
            result = result.text
        
        return str(result).strip()
    
    def transcribe_batch(self, audio_paths, batch_size=8):
        model = self.load_model()
        
        valid_paths = []
        for p in audio_paths:
            if os.path.exists(p):
                valid_paths.append(p)
            else:
                print(f"Warning: File not found, skipping: {p}")
        
        if not valid_paths:
            return []
        
        results = []
        for i in range(0, len(valid_paths), batch_size):
            batch = valid_paths[i:i+batch_size]
            
            with torch.no_grad():
                batch_result = model.transcribe(
                    batch,
                    batch_size=len(batch),
                    language_id=LANGUAGE_ID
                )
            
            if isinstance(batch_result, tuple):
                batch_result = batch_result[0]
            
            for r in batch_result:
                if hasattr(r, "text"):
                    r = r.text
                results.append(str(r).strip())
        
        return results


def get_audio_files(path):
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    path = Path(path)
    
    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        files = []
        for ext in extensions:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted([str(f) for f in files])
    else:
        return []


def main():
    parser = argparse.ArgumentParser(description="Bhili ASR Inference")
    parser.add_argument("--model", type=str, default="bhili_asr_finetune_v1.nemo",
                        help="Path to .nemo model file")
    parser.add_argument("--audio", type=str, nargs="+",
                        help="Audio file(s) to transcribe")
    parser.add_argument("--audio_dir", type=str,
                        help="Directory containing audio files")
    parser.add_argument("--output", type=str,
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        help="Device to use (default: auto)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for transcription")
    
    args = parser.parse_args()
    
    audio_files = []
    if args.audio:
        for a in args.audio:
            audio_files.extend(get_audio_files(a))
    if args.audio_dir:
        audio_files.extend(get_audio_files(args.audio_dir))
    
    if not audio_files:
        print("Error: No audio files specified or found.")
        print("Usage: python inference.py --audio file.wav")
        return
    
    print(f"Found {len(audio_files)} audio file(s)")
    
    asr = BhiliASR(
        model_path=args.model,
        device=args.device
    )
    
    results = []
    
    if len(audio_files) == 1:
        text = asr.transcribe(audio_files[0])
        results.append({"file": audio_files[0], "transcription": text})
        print(f"\nTranscription:\n{text}")
    else:
        texts = asr.transcribe_batch(audio_files, batch_size=args.batch_size)
        for f, t in zip(audio_files, texts):
            results.append({"file": f, "transcription": t})
            print(f"\n{os.path.basename(f)}:")
            print(f"  {t}")
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()