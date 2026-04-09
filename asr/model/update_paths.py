"""
Update tokenizer paths in Bhili ASR model.

Usage:
    python update_paths.py --root_dir /path/to/your/tokenizers_v3
    
Example:
    python update_paths.py --root_dir /home/user/bhili-asr/tokenizers/tokenizers_v3
"""

import argparse
import os
import tarfile
import tempfile


def update_paths(model_path, root_dir):
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return False
    
    root_dir = os.path.abspath(root_dir)
    
    if not os.path.exists(root_dir):
        print(f"Warning: Directory does not exist yet: {root_dir}")
        print("Make sure to extract tokenizers there before running inference.")
    
    output_path = model_path.replace(".nemo", "_updated.nemo")
    
    print(f"Model: {model_path}")
    print(f"New tokenizer root: {root_dir}")
    print(f"Output: {output_path}")
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Extracting model...")
        with tarfile.open(model_path, 'r') as tar:
            tar.extractall(tmpdir)
        
        config_path = os.path.join(tmpdir, "model_config.yaml")
        
        if not os.path.exists(config_path):
            print("Error: model_config.yaml not found")
            return False
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Replace old tokenizer paths
        old_path = "/home/bhili_asr/tokenizers/tokenizers_v3"
        content = content.replace(old_path, root_dir)
        
        # Also handle original training paths if present
        old_training_path = "/projects/data/asrteam/speechteam/projects/large_scale_training_yotta/assets/tokenizers"
        content = content.replace(old_training_path, root_dir)
        
        # Update top-level dir if exists
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('dir:') and 'tokenizer' in lines[max(0,i-5):i+1][-1].lower():
                indent = len(line) - len(line.lstrip())
                lines[i] = ' ' * indent + f'dir: {root_dir}'
                break
        content = '\n'.join(lines)
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print("Repacking model...")
        with tarfile.open(output_path, 'w') as tar:
            for item in os.listdir(tmpdir):
                tar.add(os.path.join(tmpdir, item), arcname=item)
    
    print()
    print(f"Done! Updated model saved to: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Extract tokenizers to: {root_dir}")
    print(f"  2. Run inference with: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update tokenizer paths in Bhili ASR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python update_paths.py --root_dir /home/user/tokenizers/tokenizers_v3
    python update_paths.py --root_dir ./tokenizers/tokenizers_v3 --model bhili_asr.nemo
        """
    )
    parser.add_argument(
        "--root_dir", 
        type=str, 
        required=True,
        help="Path to tokenizers_v3 directory (e.g., /home/user/tokenizers/tokenizers_v3)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="bhili_asr_finetune_v1.nemo",
        help="Path to .nemo model file (default: bhili_asr_finetune_v1.nemo)"
    )
    
    args = parser.parse_args()
    
    success = update_paths(args.model, args.root_dir)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()